"""Telegram service: alerts out, commands in. No inbound port, no webhook.

- Long polling (``getUpdates``); only Ray's chat ID is answered, everything else is dropped.
- Alerts are rendered from ledger rows (order, fill, close, block, fault). Each row is
  claimed in ``telegram_sent`` before it is sent, so a restart or a second poll never
  sends the same row twice; a failed send releases the claim and is retried.
- Commands (/pause /resume /status /flatten) are written to the ``commands`` table through
  ``Mailbox`` and applied by the core on its next bar. This process cannot append to the
  ledger chain. /flatten needs a confirm tap that expires after 60 s.
- Quiet hours 23:00-07:30 Asia/Singapore: trade alerts are silent, faults are always loud.
"""
from __future__ import annotations

import json
import logging
import secrets as _rand
import time as _time
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

from tradex.core.inbox import Mailbox

log = logging.getLogger("tradex.telegram")

SCHEMA = """
CREATE TABLE IF NOT EXISTS telegram_sent (
    seq      INTEGER PRIMARY KEY,       -- events.seq of the alerted row
    sent_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS telegram_kv (k TEXT PRIMARY KEY, v TEXT NOT NULL);
"""
ALERT_KINDS = ("order", "fill", "close", "veto", "verdict", "health")
CONFIRM_TTL_S = 60
FAULT_REPEAT_S = 6 * 3600          # an identical fault is alerted once per this long (every row stays in the ledger)
QUIET_START, QUIET_END = time(23, 0), time(7, 30)
try:
    from zoneinfo import ZoneInfo
    SGT: Any = ZoneInfo("Asia/Singapore")
except Exception:  # noqa: BLE001 - no tzdata on the box: Singapore has no DST, a fixed offset is exact
    SGT = timezone(timedelta(hours=8))


class Transport(Protocol):
    def call(self, method: str, payload: dict[str, Any], timeout: float = 15) -> dict[str, Any]: ...


class RequestsTransport:
    """Bot API over HTTPS with ``requests``. Returns the ``result`` field; raises on ok=false."""

    def __init__(self, token: str):
        self._url = f"https://api.telegram.org/bot{token}/"

    def call(self, method: str, payload: dict[str, Any], timeout: float = 15) -> dict[str, Any]:
        import requests
        try:
            r = requests.post(self._url + method, json=payload, timeout=timeout)
            body = r.json()
        except Exception as exc:  # noqa: BLE001 - never put the token-bearing URL in an error
            raise RuntimeError(f"telegram {method} failed: {type(exc).__name__}") from None
        if not body.get("ok"):
            raise RuntimeError(f"telegram {method} refused: {body.get('description', r.status_code)}")
        return body.get("result")


def is_quiet(now: datetime) -> bool:
    t = now.astimezone(SGT).time()
    return t >= QUIET_START or t < QUIET_END


def render(kind: str, d: dict[str, Any]) -> tuple[str, bool] | None:
    """(text, is_fault) for a ledger row, or None when the row is not alert-worthy."""
    did = d.get("decision_id", "")
    if kind == "order":
        side = "BUY" if d["side"] > 0 else "SELL"
        px = f" @ {d['price']}" if d.get("price") is not None else ""
        return f"ORDER {side} {d['qty']:g} {d['symbol']} {d['order_type']}{px} ({d['purpose']}) [{did}]", False
    if kind == "fill":
        side = "bought" if d["side"] > 0 else "sold"
        return f"FILL {side} {d['qty']:g} {d['symbol']} @ {d['price']} fees ${d['fees_usd']:.2f} [{did}]", False
    if kind == "close":
        return (f"CLOSE {d['symbol']} {d['qty']:g} @ {d['exit_price']} pnl ${d['net_pnl_usd']:+.2f} "
                f"({d['r_multiple']:+.2f}R) {d['reason']} [{did}]"), False
    if kind == "veto":
        return f"BLOCKED by {d['source']}: {d['reason']} [{did}]", False
    if kind == "verdict" and d.get("outcome") == "rejected":
        return f"BLOCKED by risk gate: {'; '.join(d.get('reasons', [])) or 'rejected'} [{did}]", False
    if kind == "health" and not d.get("ok", True):
        return f"FAULT {d['check']}: {d.get('detail', '')}", True
    return None


class TelegramService:
    def __init__(self, ledger_path: str | Path, token: str | None = None, chat_id: int | str | None = None,
                 transport: Transport | None = None, now: Callable[[], datetime] | None = None,
                 poll_timeout: int = 10, backfill: bool = True):
        if transport is None:
            if token is None:
                from tradex import secrets  # lazy: tests inject everything
                token = secrets.get("telegram_bot_token")
            transport = RequestsTransport(token)
        if chat_id is None:
            from tradex import secrets
            chat_id = secrets.get("telegram_chat_id")
        self.chat_id = int(chat_id)
        self.t = transport
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.poll_timeout = poll_timeout
        self.mb = Mailbox(ledger_path, extra_schema=SCHEMA, extra_writable=("telegram_sent", "telegram_kv"))
        self.pending_flatten: dict[str, float] = {}      # token -> expiry (monotonic-free: epoch seconds)
        self.dropped = 0
        self._fault_seen: dict[str, list] = {}           # fault text -> [last sent (epoch s), repeats held back]
        if not backfill:
            self._baseline()

    # --- alerts ------------------------------------------------------------------------

    def _baseline(self) -> None:
        with self.mb._lock, self.mb.db:
            self.mb.db.execute("INSERT OR IGNORE INTO telegram_sent (seq, sent_at) SELECT seq, 'baseline' FROM events")

    def send_alerts(self) -> int:
        marks = ",".join("?" * len(ALERT_KINDS))
        rows = self.mb.read(f"SELECT seq, kind, payload FROM events WHERE kind IN ({marks}) AND seq NOT IN "
                            "(SELECT seq FROM telegram_sent) ORDER BY seq", ALERT_KINDS)
        sent = 0
        for r in rows:
            out = render(r["kind"], json.loads(r["payload"]))
            with self.mb._lock, self.mb.db:
                cur = self.mb.db.execute("INSERT OR IGNORE INTO telegram_sent (seq, sent_at) VALUES (?,?)",
                                         (r["seq"], self.now().isoformat()))
            if cur.rowcount == 0:
                continue                                  # another poller claimed it
            if out is None:
                continue                                  # claimed so it is never re-rendered
            text, fault = out
            key = text
            if fault:
                text = self._dedupe_fault(key)
                if text is None:
                    continue                              # same fault already sent recently; the ledger keeps every row
            try:
                self.t.call("sendMessage", {"chat_id": self.chat_id, "text": text,
                                            "disable_notification": (not fault) and is_quiet(self.now())})
                sent += 1
                if fault:
                    self._fault_seen[key] = [self.now().timestamp(), 0]
            except Exception as exc:  # noqa: BLE001 - release the claim, retry next cycle
                log.warning("alert %s not sent: %s", r["seq"], exc)
                with self.mb._lock, self.mb.db:
                    self.mb.db.execute("DELETE FROM telegram_sent WHERE seq=?", (r["seq"],))
                break
        return sent

    def _dedupe_fault(self, text: str) -> str | None:
        """The same fault text is sent once per FAULT_REPEAT_S; the next one after that says how many it covers."""
        now = self.now().timestamp()
        seen = self._fault_seen.get(text)
        if seen is not None and now - seen[0] < FAULT_REPEAT_S:
            seen[1] += 1
            return None
        held = seen[1] if seen else 0
        return text + (f"  (repeated {held} more time{'s' if held != 1 else ''} since the last alert)" if held else "")

    # --- commands ----------------------------------------------------------------------

    def _reply(self, text: str, **extra: Any) -> None:
        self.t.call("sendMessage", {"chat_id": self.chat_id, "text": text, **extra})

    def _status(self) -> str:
        snap = self.mb.read("SELECT payload FROM events WHERE kind='snapshot' ORDER BY seq DESC LIMIT 1")
        pend = self.mb.read("SELECT COUNT(*) n FROM commands WHERE applied_at IS NULL")[0]["n"]
        last = self.mb.read("SELECT time FROM events ORDER BY seq DESC LIMIT 1")
        if not snap:
            return f"No snapshot yet. Pending commands: {pend}."
        d = json.loads(snap[0]["payload"])
        lim = d.get("limits", {})
        return (f"Equity ${d['equity_usd']:,.2f}, open risk ${d['open_risk_usd']:,.2f}, "
                f"{len(d['positions'])} positions, tier {lim.get('tier')}, "
                f"{'PAUSED' if lim.get('paused') else 'trading'}. Pending commands: {pend}. "
                f"Last ledger row: {last[0]['time'] if last else '-'}")

    def _cmd(self, name: str) -> None:
        stamp = self.now().isoformat()
        if name in ("pause", "resume"):
            cid = self.mb.add_command(stamp, "telegram", name)
            self._reply(f"{name} queued (command {cid}); the core applies it on its next bar.")
        elif name == "status":
            self._reply(self._status())
        elif name == "flatten":
            tok = _rand.token_hex(4)
            self.pending_flatten[tok] = self.now().timestamp() + CONFIRM_TTL_S
            kb = {"inline_keyboard": [[{"text": "Confirm flatten", "callback_data": f"flatten:{tok}"},
                                       {"text": "Cancel", "callback_data": f"cancel:{tok}"}]]}
            self._reply(f"Close ALL positions and pause? Confirm within {CONFIRM_TTL_S} s.", reply_markup=kb)

    def _callback(self, cq: dict[str, Any]) -> None:
        stamp = self.now()
        action, _, tok = (cq.get("data") or "").partition(":")
        exp = self.pending_flatten.pop(tok, None)
        if exp is None or stamp.timestamp() > exp:
            text = "Expired or already used. Send /flatten again."
        elif action == "flatten":
            cid = self.mb.add_command(stamp.isoformat(), "telegram", "flatten")
            text = f"Flatten queued (command {cid}); the core closes positions on its next bar."
        else:
            text = "Cancelled."
        self.t.call("answerCallbackQuery", {"callback_query_id": cq["id"], "text": text})
        msg = cq.get("message") or {}
        if msg.get("message_id") is not None:
            self.t.call("editMessageText", {"chat_id": self.chat_id, "message_id": msg["message_id"], "text": text})

    def handle(self, upd: dict[str, Any]) -> None:
        cq = upd.get("callback_query")
        if cq is not None:
            chat = ((cq.get("message") or {}).get("chat") or {}).get("id")
            if chat != self.chat_id or (cq.get("from") or {}).get("id") != self.chat_id:
                self.dropped += 1
                return
            self._callback(cq)
            return
        msg = upd.get("message") or {}
        if (msg.get("chat") or {}).get("id") != self.chat_id or (msg.get("from") or {}).get("id") != self.chat_id:
            self.dropped += 1                             # not Ray: no reply, no trace of the bot's features
            return
        text = (msg.get("text") or "").strip()
        if text.startswith("/"):
            name = text.split()[0][1:].split("@")[0].lower()
            if name in ("pause", "resume", "status", "flatten"):
                self._cmd(name)

    def poll_once(self) -> int:
        row = self.mb.read("SELECT v FROM telegram_kv WHERE k='offset'")
        offset = int(row[0]["v"]) if row else 0
        ups = self.t.call("getUpdates", {"offset": offset, "timeout": self.poll_timeout,
                                         "allowed_updates": ["message", "callback_query"]},
                          timeout=self.poll_timeout + 10) or []
        for u in ups:
            try:
                self.handle(u)
            except Exception as exc:  # noqa: BLE001 - one bad update must not wedge the queue
                log.warning("update %s failed: %s", u.get("update_id"), exc)
            offset = max(offset, int(u["update_id"]) + 1)
        if ups:
            with self.mb._lock, self.mb.db:
                self.mb.db.execute("INSERT OR REPLACE INTO telegram_kv (k, v) VALUES ('offset', ?)", (str(offset),))
        return len(ups)

    def run(self, should_stop: Callable[[], bool] = lambda: False) -> None:
        backoff = 1.0
        while not should_stop():
            try:
                self.send_alerts()
                self.poll_once()
                backoff = 1.0
            except Exception as exc:  # noqa: BLE001 - network blips: back off, keep the service alive
                log.warning("telegram cycle failed: %s", exc)
                _time.sleep(backoff)
                backoff = min(backoff * 2, 60)
