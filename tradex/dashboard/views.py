"""Everything the dashboard shows, as plain dicts computed from the ledger (read-only).

No function here writes, calls a broker or invents a number: with an empty ledger each view
returns ``empty: True`` and zero counts, and the page shows an empty state. Broker data
(Oanda, moomoo) appears only as the rows the core already wrote into the ledger.

Run labelling: the ledger does not record the run mode, so ``run_id`` is the label. A run id
starting with ``paper`` or ``live`` is a real run; ``replay``/``parity``/``dry`` are not. Anything
else is "unlabelled" and is never counted as readiness evidence (see ``tradex.readiness``).
"""
from __future__ import annotations

import json
import math
import re
import sqlite3
import threading
import time as _time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml

from tradex.core.ledger import Ledger

REAL_PREFIXES = ("paper", "live")
NON_REAL_PREFIXES = ("replay", "parity", "dry", "test", "backtest")
STALE_AFTER_S = 6 * 3600           # a paper or live core writes a row every bar; this long silent means it stopped
WEEKEND_GAP_MAX_S = 80 * 3600      # the market-closed weekend can legitimately leave a gap this long
MIN_TRADES = 30                    # same bar as config/gates/readiness.yaml (paper_trades_per_venue)
SYMBOL_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,24}$")
TIMELINE_KINDS = ("plan", "veto", "verdict", "order", "fill", "close", "health", "agent_output", "exit_change")


class LedgerUnavailable(Exception):
    pass


@dataclass
class Sources:
    ledger: Path = Path("data/ledger/live.sqlite")
    strategies_dir: Path = Path("strategies")
    reports_dir: Path = Path("reports")
    bars_dir: Path = Path("data/cache")
    accounts_path: Path = Path("config/accounts.yaml")
    runtime_config: Path = Path("config/runtime.yaml")
    readiness_path: Path = Path("config/gates/readiness.yaml")
    state_dir: Path = Path("data/state")          # ops scripts drop backup_ok / healthcheck_ok stamps here
    tz: str = "Asia/Singapore"


def clean(v: Any) -> Any:
    """JSON-safe copy: NaN and infinity become None (browsers reject them)."""
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    return v


def parse_ts(s: Any) -> datetime | None:
    try:
        d = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def venue_of(symbol: str) -> str:
    return "oanda" if "_" in (symbol or "") else "moomoo"


def side_word(direction: int) -> str:
    return "Buy" if direction > 0 else "Sell" if direction < 0 else "Flat"


def pretty_symbol(sym: str) -> str:
    return sym.replace("_", "/") if "_" in (sym or "") else sym


def run_kind(run_id: str) -> str:
    r = (run_id or "").lower()
    if r.startswith(REAL_PREFIXES):
        return "live" if r.startswith("live") else "paper"
    if r.startswith(NON_REAL_PREFIXES):
        return "replay"
    return "unlabelled"


def _signed_usd(v: float) -> str:
    return f"{'-' if v < 0 else '+'}${abs(v):,.2f}"


def describe(kind: str, d: dict[str, Any]) -> str:
    """One plain sentence per ledger row, for the Today timeline."""
    sym = pretty_symbol(d.get("symbol", ""))
    if kind == "plan":
        return (f"Plan: {side_word(d.get('direction', 0))} {sym}, expected {d.get('ev_r', 0):+.2f}R "
                f"after costs, reward/risk {d.get('reward_risk', 0):.1f}")
    if kind == "veto":
        return f"Blocked by {d.get('source', '?')}: {d.get('reason', '')}"
    if kind == "verdict":
        if d.get("outcome") == "accepted":
            return f"Risk gate accepted: size {d.get('qty', 0):g}, risking ${d.get('risk_usd', 0):,.2f}"
        return "Risk gate declined: " + "; ".join(map(str, d.get("reasons", [])[:2]))
    if kind == "order":
        return f"Order: {side_word(d.get('side', 0))} {d.get('qty', 0):g} {sym} ({d.get('purpose', '')}, {d.get('order_type', '')})"
    if kind == "fill":
        return f"Filled {side_word(d.get('side', 0))} {d.get('qty', 0):g} {sym} at {d.get('price', 0):g}"
    if kind == "close":
        return f"Closed {sym}: {d.get('r_multiple', 0):+.2f}R, {_signed_usd(d.get('net_pnl_usd', 0))} ({d.get('reason', '')})"
    if kind == "exit_change":
        return f"Moved {d.get('field_name', '')} from {d.get('old')} to {d.get('new')}: {d.get('reason', '')}"
    if kind == "health":
        return f"{'OK' if d.get('ok') else 'FAULT'} {d.get('check', '')}: {d.get('detail', '')}".strip()
    if kind == "agent_output":
        body = d.get("body") or {}
        shadow = " (shadow: not applied)" if body.get("shadow") or body.get("applied") is False else ""
        return f"Agent {d.get('agent', '')} {d.get('action', '')} on {d.get('target', '')}{shadow}"
    return kind


class Views:
    def __init__(self, src: Sources):
        self.src = src
        self.tz = ZoneInfo(src.tz)
        self._chain: dict[str, Any] = {}
        self._chain_lock = threading.Lock()

    # --- plumbing ---------------------------------------------------------------------

    def open(self) -> Ledger:
        p = self.src.ledger
        if not p.exists():
            raise LedgerUnavailable(f"no ledger file at {p}")
        try:
            led = Ledger(p, read_only=True, git_commit="")
            led.db.execute("SELECT 1 FROM events LIMIT 1")
            return led
        except sqlite3.Error as exc:
            raise LedgerUnavailable(f"ledger at {p} cannot be read: {exc}") from exc

    @staticmethod
    def q(led: Ledger, sql: str, args: tuple | list = ()) -> list[dict[str, Any]]:
        return [dict(r) for r in led.db.execute(sql, tuple(args))]

    @staticmethod
    def payloads(led: Ledger, sql: str, args: tuple | list = ()) -> list[dict[str, Any]]:
        out = []
        for r in led.db.execute(sql, tuple(args)):
            d = json.loads(r["payload"])
            d["_seq"] = r["seq"]
            out.append(d)
        return out

    def _try_table(self, led: Ledger, sql: str, args: tuple | list = ()) -> list[dict[str, Any]]:
        try:
            return self.q(led, sql, args)
        except sqlite3.Error:
            return []

    def _empty(self, reason: str, **extra: Any) -> dict[str, Any]:
        return {"empty": True, "reason": reason, **extra}

    def now(self) -> datetime:
        return datetime.now(timezone.utc)

    # --- meta + change token -----------------------------------------------------------

    def meta(self) -> dict[str, Any]:
        base = {"tz": self.src.tz, "generated_at": self.now().isoformat(), "ledger_path": str(self.src.ledger)}
        try:
            led = self.open()
        except LedgerUnavailable as exc:
            return base | {"empty": True, "source": "none", "real": False, "reason": str(exc),
                           "banner": "No ledger yet. Start the core (or a replay) and this page fills in.",
                           "last_seq": 0, "last_event_time": None}
        try:
            return base | self._meta(led)
        finally:
            led.close()

    def _meta(self, led: Ledger) -> dict[str, Any]:
        last = self.q(led, "SELECT seq, time FROM events ORDER BY seq DESC LIMIT 1")
        runs = [r["run_id"] for r in self.q(led, "SELECT DISTINCT run_id FROM events")]
        kinds = {run_kind(r) for r in runs}
        if not last:
            source, banner = "empty", "The ledger is empty: nothing has been recorded yet."
        elif kinds <= {"paper", "live"}:
            source = "live" if "live" in kinds else "paper"
            banner = ""
        elif kinds == {"replay"}:
            source = "replay"
            banner = "This ledger is from a replay over recorded bars. It is not paper or live trading and is never readiness evidence."
        else:
            source = "mixed" if len(kinds) > 1 else "unlabelled"
            banner = ("This ledger has runs that are not labelled paper or live (run ids: "
                      + ", ".join(sorted(runs)[:4]) + "). They do not count as real evidence.")
        cmd = self._try_table(led, "SELECT COUNT(*) n FROM commands WHERE applied_at IS NULL")
        snap = self.q(led, "SELECT payload FROM events WHERE kind='snapshot' AND book='ensemble' ORDER BY seq DESC LIMIT 1")
        limits = json.loads(snap[0]["payload"]).get("limits", {}) if snap else {}
        return {"empty": not last, "source": source, "real": source in ("paper", "live"), "banner": banner,
                "run_ids": runs, "last_seq": last[0]["seq"] if last else 0,
                "last_event_time": last[0]["time"] if last else None,
                "paused": bool(limits.get("paused")), "tier": limits.get("tier"),
                "pending_commands": cmd[0]["n"] if cmd else 0, "agents_mode": self._agents_mode()}

    def _agents_mode(self) -> str | None:
        try:
            doc = yaml.safe_load(self.src.runtime_config.read_text()) or {}
            return (doc.get("agents") or {}).get("mode")
        except (OSError, yaml.YAMLError):
            return None

    def change_token(self) -> str:
        try:
            led = self.open()
        except LedgerUnavailable:
            return "none"
        try:
            a = self.q(led, "SELECT COALESCE(MAX(seq),0) n FROM events")[0]["n"]
            b = self._try_table(led, "SELECT COALESCE(MAX(id),0) n, COUNT(applied_at) m FROM commands")
            c = self._try_table(led, "SELECT COALESCE(MAX(id),0) n FROM agent_calls")
            return f"{a}.{b[0]['n'] if b else 0}.{b[0]['m'] if b else 0}.{c[0]['n'] if c else 0}"
        finally:
            led.close()

    # --- Today -------------------------------------------------------------------------

    def day_bounds(self, now: datetime | None = None) -> tuple[datetime, datetime]:
        now = (now or self.now()).astimezone(self.tz)
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return start.astimezone(timezone.utc), (start + timedelta(days=1)).astimezone(timezone.utc)

    def today(self, now: datetime | None = None) -> dict[str, Any]:
        meta = self.meta()
        if meta.get("empty"):
            return self._empty(meta.get("reason") or "no rows yet", meta=meta, headline=meta.get("banner", ""),
                               attention=[], timeline=[], counts={}, health=[])
        led = self.open()
        try:
            start, end = self.day_bounds(now)
            cushion = (start - timedelta(hours=1)).isoformat()
            rows = []
            for r in led.db.execute(
                    "SELECT seq, kind, time, decision_id, book, symbol, payload FROM events WHERE time>=? ORDER BY seq",
                    (cushion,)):
                t = parse_ts(r["time"])
                if t is not None and start <= t < end:
                    rows.append((r, json.loads(r["payload"])))
            counts = {k: 0 for k in ("plan", "veto", "accepted", "declined", "fill", "close")}
            pnl, closed_r = 0.0, 0.0
            timeline = []
            for r, d in rows:
                k = r["kind"]
                is_main = (r["book"] or "ensemble") == "ensemble"
                if k in ("plan", "veto", "fill", "close") and is_main:
                    counts[k] += 1
                if k == "verdict":
                    counts["accepted" if d.get("outcome") == "accepted" else "declined"] += 1
                if k == "close" and is_main:
                    pnl += d.get("net_pnl_usd", 0.0)
                    closed_r += d.get("r_multiple", 0.0)
                if k in TIMELINE_KINDS and (is_main or k in ("health", "agent_output", "veto", "verdict", "exit_change")):
                    timeline.append({"seq": r["seq"], "time": r["time"], "kind": k, "decision_id": r["decision_id"] or None,
                                     "text": describe(k, d), "bad": (k == "health" and not d.get("ok", True))})
            timeline = timeline[-40:][::-1]
            snap = self._latest_snapshot(led)
            prev = self._snapshot_before(led, start)
            health = self._health_latest(led)
            attention = self._attention(meta, health, led, now)
            pos = (snap or {}).get("positions", [])
            lim = (snap or {}).get("limits", {})
            eq = (snap or {}).get("equity_usd")
            return {
                "empty": False, "meta": meta, "headline": self._headline(meta, snap, counts, attention),
                "equity": {"now": eq, "since_yesterday": (eq - prev["equity_usd"]) if (snap and prev) else None,
                           "as_of": (snap or {}).get("time"), "cash": (snap or {}).get("cash_usd")},
                "risk": {"open_risk_usd": (snap or {}).get("open_risk_usd"), "heat_cap_usd": lim.get("heat_cap_usd"),
                         "open_positions": len(pos), "tier": lim.get("tier"), "paused": bool(lim.get("paused"))},
                "closed_today": {"trades": counts["close"], "net_pnl_usd": round(pnl, 2), "total_r": round(closed_r, 2)},
                "counts": counts, "attention": attention, "health": health, "timeline": timeline,
                "day": {"start": start.isoformat(), "end": end.isoformat()},
            }
        finally:
            led.close()

    def _headline(self, meta, snap, counts, attention) -> str:
        if snap is None:
            return "Nothing has traded yet. The ledger has rows but no account snapshot."
        bits = []
        if meta.get("paused"):
            bits.append("Paused: no new entries")
        else:
            bits.append("Trading normally" if not attention else "Running, but something needs a look")
        n = len(snap.get("positions", []))
        bits.append(f"{n} open position{'s' if n != 1 else ''}")
        bits.append(f"{counts['close']} closed today")
        return ". ".join(bits) + "."

    def stale_age_s(self, meta: dict[str, Any], now: datetime | None = None) -> float | None:
        """Seconds since the last ledger row when a paper or live run has gone quiet, else None.

        The core writes at least one row per bar, so hours of silence mean it stopped (Mac asleep, crash). Quiet
        is expected over the weekend market gap (Friday 21:00 UTC to Monday 04:00 UTC); replays are never
        judged, since their rows are old by design."""
        if not meta.get("real"):
            return None
        last = parse_ts(meta.get("last_event_time"))
        if last is None:
            return None
        now = (now or self.now()).astimezone(timezone.utc)
        age = (now - last).total_seconds()
        if age <= STALE_AFTER_S:
            return None
        weekend_gap = ((now.weekday() == 4 and now.hour >= 21) or now.weekday() in (5, 6)
                       or (now.weekday() == 0 and now.hour < 4))
        if weekend_gap and age < WEEKEND_GAP_MAX_S:
            return None
        return age

    def _attention(self, meta, health, led, now: datetime | None = None) -> list[dict[str, str]]:
        out = []
        stale = self.stale_age_s(meta, now)
        if stale is not None:
            hours = stale / 3600
            out.append({"level": "fault", "text": "No ledger activity for "
                        + (f"{hours:.0f} hours" if hours < 48 else f"{hours / 24:.0f} days")
                        + ". The core may have stopped (Mac asleep, or the run crashed).", "time": meta.get("last_event_time")})
        for h in health:
            if not h["ok"]:
                out.append({"level": "fault", "text": f"{h['check']}: {h['detail'] or 'failing'}", "time": h["time"]})
        if meta.get("paused"):
            out.append({"level": "info", "text": "Entries are paused (pause command applied)."})
        if meta.get("pending_commands"):
            out.append({"level": "info", "text": f"{meta['pending_commands']} command(s) waiting for the core to apply."})
        chain = self.chain_status(led, wait=False)
        if chain.get("ok") is False:
            out.append({"level": "fault", "text": f"Ledger hash chain broken at row {chain.get('bad_seq')}.", "time": None})
        if meta.get("banner"):
            out.append({"level": "info", "text": meta["banner"]})
        return out

    def _health_latest(self, led: Ledger) -> list[dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for d in self.payloads(led, "SELECT seq, payload FROM events WHERE kind='health' ORDER BY seq"):
            out[d["check"]] = {"check": d["check"], "ok": bool(d.get("ok")), "detail": d.get("detail", ""), "time": d.get("time")}
        return sorted(out.values(), key=lambda h: (h["ok"], h["check"]))

    def _latest_snapshot(self, led: Ledger, book: str = "ensemble") -> dict[str, Any] | None:
        r = led.db.execute("SELECT payload FROM events WHERE kind='snapshot' AND book=? ORDER BY seq DESC LIMIT 1",
                           (book,)).fetchone()
        return json.loads(r["payload"]) if r else None

    def _snapshot_before(self, led: Ledger, t: datetime, book: str = "ensemble") -> dict[str, Any] | None:
        r = led.db.execute("SELECT payload FROM events WHERE kind='snapshot' AND book=? AND time<? "
                           "ORDER BY time DESC, seq DESC LIMIT 1", (book, t.isoformat())).fetchone()
        return json.loads(r["payload"]) if r else None

    # --- Positions & exposure -----------------------------------------------------------

    def positions(self, at: str | None = None, book: str = "ensemble") -> dict[str, Any]:
        try:
            led = self.open()
        except LedgerUnavailable as exc:
            return self._empty(str(exc), positions=[], exposure=[])
        try:
            snap = led.snapshot_at(at, book) if at else self._latest_snapshot(led, book)
            books = [r["book"] for r in self.q(led, "SELECT DISTINCT book FROM events WHERE kind='snapshot' AND book IS NOT NULL ORDER BY book")]
            if snap is None:
                return self._empty("No account snapshot has been recorded yet." if not at else "No snapshot at or before that time.",
                                   positions=[], exposure=[], books=books, book=book)
            rows = []
            for p in snap.get("positions", []):
                entry, stop, tgt, mark = p.get("entry"), p.get("stop"), p.get("target"), p.get("mark")
                risk = abs(entry - stop) if entry is not None and stop is not None else None
                ur = ((mark - entry) * p["direction"] / risk) if (risk and mark is not None) else None
                rows.append({**p, "symbol_label": pretty_symbol(p["symbol"]), "side": side_word(p["direction"]),
                             "venue": venue_of(p["symbol"]), "unrealised_r": ur,
                             "to_stop_pct": (abs(mark - stop) / mark * 100) if (mark and stop is not None) else None,
                             "to_target_pct": (abs(tgt - mark) / mark * 100) if (mark and tgt is not None) else None,
                             "has_stop": stop is not None})
            lim = snap.get("limits", {})
            cap = lim.get("heat_cap_usd")
            exposure = sorted(({"currency": k, "usd": v} for k, v in (snap.get("exposure_by_currency") or {}).items()),
                              key=lambda e: -abs(e["usd"]))
            return {"empty": False, "as_of": snap.get("time"), "book": book, "books": books, "positions": rows,
                    "equity_usd": snap.get("equity_usd"), "cash_usd": snap.get("cash_usd"),
                    "heat": {"open_risk_usd": snap.get("open_risk_usd"), "cap_usd": cap,
                             "used_pct": (snap["open_risk_usd"] / cap * 100) if cap else None},
                    "limits": lim, "exposure": exposure, "looking_back": bool(at)}
        finally:
            led.close()

    # --- Decisions ----------------------------------------------------------------------

    def _by_decision(self, led: Ledger, ids: list[str], kinds: tuple[str, ...]) -> dict[str, dict[str, list[dict[str, Any]]]]:
        out: dict[str, dict[str, list[dict[str, Any]]]] = {i: {} for i in ids}
        for i in range(0, len(ids), 400):
            chunk = ids[i:i + 400]
            sql = (f"SELECT payload FROM events WHERE decision_id IN ({','.join('?' * len(chunk))}) "
                   f"AND kind IN ({','.join('?' * len(kinds))}) ORDER BY seq")
            for r in led.db.execute(sql, [*chunk, *kinds]):
                d = json.loads(r["payload"])
                out[d["decision_id"]].setdefault(d["kind"], []).append(d)
        return out

    @staticmethod
    def _outcome(g: dict[str, list[dict[str, Any]]]) -> str:
        if "close" in g:
            return "closed"
        if "fill" in g:
            return "open"
        if "veto" in g:
            return "blocked"
        v = g.get("verdict")
        if v:
            return "accepted" if v[-1].get("outcome") == "accepted" else "declined"
        return "pending"

    def decisions(self, limit: int = 100, outcome: str | None = None, symbol: str | None = None,
                  book: str = "ensemble") -> dict[str, Any]:
        try:
            led = self.open()
        except LedgerUnavailable as exc:
            return self._empty(str(exc), decisions=[])
        try:
            sql, args = "SELECT payload FROM events WHERE kind='plan' AND book=?", [book]
            if symbol:
                sql += " AND symbol=?"
                args.append(symbol)
            fetch = max(1, min(limit, 500)) * (4 if outcome else 1)
            plans = [json.loads(r["payload"]) for r in led.db.execute(sql + " ORDER BY seq DESC LIMIT ?", [*args, fetch])]
            if not plans:
                return self._empty("No plans have been recorded yet.", decisions=[])
            g = self._by_decision(led, [p["decision_id"] for p in plans], ("veto", "verdict", "fill", "close", "counterfactual"))
            out = []
            for p in plans:
                grp = g[p["decision_id"]]
                oc = self._outcome(grp)
                if outcome and oc != outcome:
                    continue
                c = grp.get("close", [None])[-1]
                blocked = (grp.get("veto") or [None])[0]
                ver = (grp.get("verdict") or [None])[-1]
                why = (blocked["source"] + ": " + blocked["reason"]) if blocked else (
                    "; ".join(ver.get("reasons", [])[:2]) if ver and ver.get("outcome") != "accepted" else "")
                out.append({"decision_id": p["decision_id"], "time": p["time"], "symbol": p["symbol"],
                            "symbol_label": pretty_symbol(p["symbol"]), "side": side_word(p["direction"]),
                            "score": p.get("score"), "ev_r": p.get("ev_r"), "reward_risk": p.get("reward_risk"),
                            "strategies": p.get("strategies", []), "outcome": oc, "why": why,
                            "qty": ver.get("qty") if ver else None,
                            "r_multiple": c.get("r_multiple") if c else None,
                            "net_pnl_usd": c.get("net_pnl_usd") if c else None,
                            "counterfactual_r": (grp.get("counterfactual") or [{}])[-1].get("r_multiple")})
                if len(out) >= limit:
                    break
            return {"empty": not out, "reason": "No decisions match that filter." if not out else "", "decisions": out}
        finally:
            led.close()

    def decision(self, decision_id: str) -> dict[str, Any] | None:
        try:
            led = self.open()
        except LedgerUnavailable:
            return None
        try:
            rows = led.why(decision_id)
            if not rows:
                return None
            by: dict[str, list[dict[str, Any]]] = {}
            for d in rows:
                by.setdefault(d["kind"], []).append(d)
            plan = (by.get("plan") or [None])[0]
            if plan is None:                                       # a vote-only or stray ID
                return None
            sym = plan["symbol"]
            end = parse_ts((by.get("close") or [{}])[-1].get("time")) or (parse_ts(plan["time"]) or self.now()) + timedelta(days=1)
            lo = ((parse_ts(plan["time"]) or end) - timedelta(days=1)).isoformat()
            opinions = [d for d in self.payloads(
                led, "SELECT seq, payload FROM events WHERE kind='agent_output' AND time>=? ORDER BY seq", (lo,))
                if d.get("target") == decision_id or (d.get("target") == sym and (parse_ts(d["time"]) or end) <= end)]
            for d in opinions:
                b = d.get("body") or {}
                d["status"] = ("shadow: recorded, not applied" if b.get("shadow") or b.get("applied") is False
                               else "applied")
            verdict = (by.get("verdict") or [None])[-1]
            checks = []
            if verdict:
                for name, val in (verdict.get("checks") or {}).items():
                    checks.append({"name": name, "detail": val if isinstance(val, dict) else {"value": val}})
            timeline = [{"seq": d["_seq"], "time": d["time"], "kind": d["kind"], "text": describe(d["kind"], d)}
                        for d in rows if d["kind"] != "vote"]
            return {"decision_id": decision_id, "symbol": sym, "symbol_label": pretty_symbol(sym),
                    "side": side_word(plan["direction"]), "outcome": self._outcome(by), "plan": plan,
                    "votes": by.get("vote", []), "vetoes": by.get("veto", []), "verdict": verdict, "checks": checks,
                    "orders": by.get("order", []), "fills": by.get("fill", []),
                    "exit_changes": by.get("exit_change", []), "close": (by.get("close") or [None])[-1],
                    "counterfactual": (by.get("counterfactual") or [None])[-1], "agent_opinions": opinions,
                    "timeline": timeline, "git_commit": plan.get("_git_commit"), "config_hash": plan.get("_config_hash")}
        finally:
            led.close()

    # --- candles ------------------------------------------------------------------------

    def bars(self, symbol: str, tf: str = "D1", decision_id: str | None = None, limit: int = 300) -> dict[str, Any]:
        from tradex.timeframes import TIMEFRAMES
        if not SYMBOL_RE.match(symbol) or tf not in TIMEFRAMES:
            return self._empty("bad symbol or timeframe", bars=[], levels=[])
        levels, markers = [], []
        if decision_id:
            d = self.decision(decision_id)
            if d:
                p = d["plan"]
                levels = [{"label": "entry", "price": p["entry_price"]}, {"label": "stop", "price": p["stop"]}] + [
                    {"label": f"target {i + 1}", "price": t} for i, t in enumerate(p["targets"])]
                markers = [{"time": f["time"], "label": f"{side_word(f['side'])} {f['qty']:g} @ {f['price']:g}"} for f in d["fills"]]
                if d["close"]:
                    markers.append({"time": d["close"]["time"], "label": f"Closed @ {d['close']['exit_price']:g}"})
        root = self.src.bars_dir
        found = sorted(root.rglob(f"{symbol}_{tf}.csv")) if root.exists() else []
        if not found:
            return self._empty(f"No cached {tf} bars for {symbol}. Run `tradex fetch` (Ray's Mac) to see candles here.",
                               bars=[], levels=levels, markers=markers)
        import pandas as pd
        from tradex.data.bars import normalize_bars
        df = normalize_bars(pd.read_csv(found[0], index_col=0)).tail(max(10, min(limit, 2000)))
        bars = [{"time": int(ts.timestamp()), "open": float(r.open), "high": float(r.high), "low": float(r.low),
                 "close": float(r.close)} for ts, r in df.iterrows()]
        for m in markers:
            t = parse_ts(m["time"])
            m["unix"] = int(t.timestamp()) if t else None
        return {"empty": not bars, "reason": "", "symbol": symbol, "tf": tf, "source_file": found[0].name, "bars": bars,
                "levels": levels, "markers": markers}

    # --- Performance --------------------------------------------------------------------

    def performance(self, book: str = "ensemble") -> dict[str, Any]:
        try:
            led = self.open()
        except LedgerUnavailable as exc:
            return self._empty(str(exc))
        try:
            closes = self.payloads(led, "SELECT seq, payload FROM events WHERE kind='close' AND book=? ORDER BY seq", (book,))
            snaps = self.payloads(led, "SELECT seq, payload FROM events WHERE kind='snapshot' AND book=? ORDER BY seq", (book,))
            fills = self.payloads(led, "SELECT seq, payload FROM events WHERE kind='fill' AND book=? ORDER BY seq", (book,))
            cfs = self.payloads(led, "SELECT seq, payload FROM events WHERE kind='counterfactual' ORDER BY seq")
            if not closes and not snaps:
                return self._empty("No closed trades or account snapshots yet.", trades=0, equity=[], filters=self._filters(cfs))
            plans = self._plans_by_id(led, [c["decision_id"] for c in closes])
            stats = self.trade_stats(closes)
            curve = self._daily_equity(snaps)
            peak, dd = 0.0, 0.0
            for p in curve:
                peak = max(peak, p["equity"])
                dd = min(dd, (p["equity"] / peak - 1) if peak else 0.0)
            by_class: dict[str, list[dict[str, Any]]] = {}
            for c in closes:
                ac = plans.get(c["decision_id"], {}).get("asset_class", "unknown")
                by_class.setdefault(ac, []).append(c)
            bins = [-3 + 0.5 * i for i in range(13)]                     # -3R .. +3R in half-R steps
            hist = [0] * 14
            for c in closes:
                r = c["r_multiple"]
                hist[sum(1 for b in bins if r >= b)] += 1
            cost = sum(f["fees_usd"] + f["spread_slippage_usd"] for f in fills)
            return {"empty": not closes and len(curve) < 2, "reason": "",
                    "stats": stats, "enough": stats["trades"] >= MIN_TRADES, "min_trades": MIN_TRADES,
                    "equity": curve, "max_drawdown_pct": dd * 100 if curve else None,
                    "costs_usd": round(cost, 2),
                    "by_asset_class": [{"asset_class": k, **self.trade_stats(v)} for k, v in sorted(by_class.items())],
                    "r_hist": {"edges": bins, "counts": hist},
                    "recent": [{"decision_id": c["decision_id"], "time": c["time"], "symbol_label": pretty_symbol(c["symbol"]),
                                "r": c["r_multiple"], "pnl": c["net_pnl_usd"], "reason": c["reason"]} for c in closes[-15:][::-1]],
                    "filters": self._filters(cfs)}
        finally:
            led.close()

    def _plans_by_id(self, led: Ledger, ids: list[str]) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        ids = list(dict.fromkeys(i for i in ids if i))
        for i in range(0, len(ids), 400):
            chunk = ids[i:i + 400]
            for r in led.db.execute(f"SELECT payload FROM events WHERE kind='plan' AND decision_id IN ({','.join('?' * len(chunk))})", chunk):
                d = json.loads(r["payload"])
                out[d["decision_id"]] = d
        return out

    @staticmethod
    def trade_stats(closes: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(closes)
        if not n:
            return {"trades": 0, "wins": 0, "win_rate": None, "avg_r": None, "total_r": 0.0, "net_pnl_usd": 0.0,
                    "profit_factor": None, "best_r": None, "worst_r": None}
        rs = [c["r_multiple"] for c in closes]
        pnl = [c["net_pnl_usd"] for c in closes]
        gw, gl = sum(p for p in pnl if p > 0), -sum(p for p in pnl if p <= 0)
        return {"trades": n, "wins": sum(1 for p in pnl if p > 0), "win_rate": sum(1 for p in pnl if p > 0) / n,
                "avg_r": sum(rs) / n, "total_r": round(sum(rs), 3), "net_pnl_usd": round(sum(pnl), 2),
                "profit_factor": (gw / gl) if gl > 0 else None, "best_r": max(rs), "worst_r": min(rs)}

    def _daily_equity(self, snaps: list[dict[str, Any]]) -> list[dict[str, Any]]:
        days: dict[str, dict[str, Any]] = {}
        for s in snaps:
            t = parse_ts(s["time"])
            if t:
                days[t.astimezone(self.tz).strftime("%Y-%m-%d")] = s
        return [{"day": k, "equity": v["equity_usd"]} for k, v in sorted(days.items())]

    def _filters(self, cfs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        from tradex.core.counterfactual import filter_report
        df = filter_report(cfs)
        return df.to_dict("records")

    # --- Strategies: forward vs backtest ------------------------------------------------

    def strategies(self) -> dict[str, Any]:
        from tradex.strategy.spec import load_dir
        specs: dict[str, dict[str, Any]] = {}
        try:
            for s in load_dir(self.src.strategies_dir):
                specs[s.id] = {"id": s.id, "family": s.family, "asset_class": s.asset_class, "status": s.status,
                               "version": s.version, "tf": s.signal_tf, "valid": not s.validate()}
        except Exception:  # noqa: BLE001 - a bad folder is an empty list, not a broken page
            specs = {}
        reports = self._reports()
        virtual: dict[str, list[dict[str, Any]]] = {}
        ens: dict[str, list[dict[str, Any]]] = {}
        try:
            led = self.open()
        except LedgerUnavailable:
            led = None
        if led is not None:
            try:
                for c in self.payloads(led, "SELECT seq, payload FROM events WHERE kind='close' ORDER BY seq"):
                    b = c.get("book", "")
                    if b.startswith("virtual:"):
                        virtual.setdefault(b[8:], []).append(c)
                plans = self._plans_by_id(led, [c["decision_id"] for c in self.payloads(
                    led, "SELECT seq, payload FROM events WHERE kind='close' AND book='ensemble' ORDER BY seq")])
                for c in self.payloads(led, "SELECT seq, payload FROM events WHERE kind='close' AND book='ensemble' ORDER BY seq"):
                    for sid in plans.get(c["decision_id"], {}).get("strategies", []):
                        ens.setdefault(sid, []).append(c)
            finally:
                led.close()
        ids = sorted(set(specs) | set(reports) | set(virtual) | set(ens))
        out = []
        for sid in ids:
            spec = specs.get(sid, {"id": sid, "family": None, "asset_class": None, "status": "unknown", "version": None,
                                   "tf": None, "valid": None})
            fwd = self.trade_stats(virtual.get(sid, []))
            ensx = self.trade_stats(ens.get(sid, []))
            back = reports.get(sid)
            delta = (fwd["avg_r"] - back["expectancy_r"]) if (fwd["avg_r"] is not None and back and back.get("expectancy_r") is not None) else None
            out.append(spec | {"forward": fwd, "in_ensemble": ensx, "backtest": back, "delta_avg_r": delta,
                               "too_early": fwd["trades"] < MIN_TRADES})
        return {"empty": not out, "reason": "No strategy files, reports or forward trades found." if not out else "",
                "strategies": out, "min_trades": MIN_TRADES}

    def _reports(self) -> dict[str, dict[str, Any]]:
        out = {}
        root = self.src.reports_dir
        if not root.exists():
            return out
        for vj in sorted(root.glob("*/validation.json")):
            try:
                v = json.loads(vj.read_text())
                oos = v.get("oos", {})
                out[v["strategy_id"]] = {
                    "gate": v.get("status"), "reasons": v.get("reasons", []), "trades": oos.get("trades"),
                    "win_rate": oos.get("win_rate"), "profit_factor": oos.get("profit_factor"),
                    "expectancy_r": oos.get("expectancy_r"), "dsr": oos.get("dsr"), "sharpe": oos.get("sharpe"),
                    "max_drawdown": oos.get("max_drawdown"), "data_source": v.get("data_source"),
                    "as_of": datetime.fromtimestamp(vj.stat().st_mtime, timezone.utc).isoformat()}
            except (OSError, ValueError, KeyError):
                continue
        return out

    # --- Accounts & system --------------------------------------------------------------

    def chain_status(self, led: Ledger, wait: bool = True) -> dict[str, Any]:
        """Hash-chain check, cached per last row so a busy page does not re-hash a big ledger."""
        last = self.q(led, "SELECT COALESCE(MAX(seq),0) n FROM events")[0]["n"]
        with self._chain_lock:
            c = self._chain
            if c.get("seq") == last and _time.monotonic() - c.get("at", 0) < 600:
                return c
        if not wait:
            return self._chain or {"ok": None, "seq": last}
        ok, bad = led.verify()
        c = {"ok": ok, "bad_seq": bad, "seq": last, "at": _time.monotonic(),
             "checked_at": self.now().isoformat()}
        with self._chain_lock:
            self._chain = c
        return c

    def accounts(self) -> list[dict[str, Any]]:
        try:
            doc = yaml.safe_load(self.src.accounts_path.read_text()) or {}
        except (OSError, yaml.YAMLError):
            return []
        return [{"name": a.get("name"), "venue": a.get("venue"), "mode": a.get("mode"),
                 "environment": a.get("environment"), "asset_classes": a.get("asset_classes", [])}
                for a in doc.get("accounts", [])]            # account IDs are never read or shown

    def _stamp_age(self, name: str, warn_after_s: float) -> dict[str, Any]:
        p = self.src.state_dir / name
        try:
            t = parse_ts(p.read_text().strip()) or datetime.fromtimestamp(p.stat().st_mtime, timezone.utc)
        except OSError:
            return {"name": name, "last": None, "age_s": None, "status": "unknown"}
        age = (self.now() - t).total_seconds()
        return {"name": name, "last": t.isoformat(), "age_s": age, "status": "ok" if age <= warn_after_s else "stale"}

    def system(self) -> dict[str, Any]:
        stamps = [dict(self._stamp_age("backup_ok", 36 * 3600), label="Nightly backup", warn_after_s=36 * 3600),
                  dict(self._stamp_age("healthcheck_ok", 15 * 60), label="Dead-man ping", warn_after_s=15 * 60)]
        base = {"accounts": self.accounts(), "ops": stamps, "agents_mode": self._agents_mode()}
        try:
            led = self.open()
        except LedgerUnavailable as exc:
            return base | self._empty(str(exc), venues=[], health=[], gateway={}, commands=[], jobs=[])
        try:
            now = self.now()
            last = self.q(led, "SELECT seq, time FROM events ORDER BY seq DESC LIMIT 1")
            size = self.src.ledger.stat().st_size
            venues = []
            for v in ("oanda", "moomoo"):
                fills = [d for d in self.payloads(led, "SELECT seq, payload FROM events WHERE kind='fill' ORDER BY seq DESC LIMIT 200")
                         if venue_of(d["symbol"]) == v]
                closes = [d for d in self.payloads(led, "SELECT seq, payload FROM events WHERE kind='close' AND book='ensemble' ORDER BY seq")
                          if venue_of(d["symbol"]) == v]
                venues.append({"venue": v, "last_fill": fills[0]["time"] if fills else None, "closed_trades": len(closes)})
            since = (now - timedelta(hours=24)).isoformat()
            calls = self._try_table(led, "SELECT provider, model, ok, latency_ms, tokens_in, tokens_out, time FROM agent_calls WHERE time>=?", (since,))
            lat = sorted(c["latency_ms"] for c in calls if c["latency_ms"] is not None)
            who: dict[str, int] = {}
            for c in calls:
                if c["ok"]:
                    k = f"{c['provider'] or '?'} / {c['model'] or '?'}"
                    who[k] = who.get(k, 0) + 1
            last_call = self._try_table(led, "SELECT time FROM agent_calls ORDER BY id DESC LIMIT 1")
            gateway = {"calls_24h": len(calls), "failed_24h": sum(1 for c in calls if not c["ok"]),
                       "median_latency_ms": lat[len(lat) // 2] if lat else None,
                       "tokens_in": sum(c["tokens_in"] or 0 for c in calls), "tokens_out": sum(c["tokens_out"] or 0 for c in calls),
                       "answered_by": [{"who": k, "calls": n} for k, n in sorted(who.items(), key=lambda kv: -kv[1])],
                       "last_call": last_call[0]["time"] if last_call else None}
            jobs = self._try_table(led, "SELECT agent, status, COUNT(*) n FROM jobs GROUP BY agent, status ORDER BY agent")
            cmds = self._try_table(led, "SELECT id, time, source, command, applied_at, result FROM commands ORDER BY id DESC LIMIT 10")
            tg = self._try_table(led, "SELECT MAX(sent_at) t FROM telegram_sent WHERE sent_at!='baseline'")
            return base | {
                "empty": not last, "reason": "",
                "ledger": {"file": self.src.ledger.name, "size_bytes": size, "last_seq": last[0]["seq"] if last else 0,
                           "last_event_time": last[0]["time"] if last else None,
                           "chain": self.chain_status(led)},
                "venues": venues, "health": self._health_latest(led),
                "recent_faults": [d for d in self.payloads(led, "SELECT seq, payload FROM events WHERE kind='health' ORDER BY seq DESC LIMIT 100")
                                  if not d.get("ok")][:10],
                "gateway": gateway, "jobs": jobs, "commands": cmds,
                "telegram_last_alert": tg[0]["t"] if tg and tg[0]["t"] else None,
                "config": self._config(led)}
        finally:
            led.close()

    def _config(self, led: Ledger) -> dict[str, Any]:
        r = self.q(led, "SELECT time, git_commit, config_hash FROM events ORDER BY seq DESC LIMIT 1")
        return r[0] if r else {}

    # --- Readiness ----------------------------------------------------------------------

    def readiness(self) -> dict[str, Any]:
        from tradex import readiness as rd
        try:
            criteria = rd.load_criteria(self.src.readiness_path)
        except (OSError, ValueError, yaml.YAMLError, KeyError) as exc:
            return self._empty(f"cannot read the readiness file: {exc}", criteria=[])
        evidence = self._evidence()
        outcomes = rd.evaluate(criteria, evidence)
        desc = {c.id: c.description for c in criteria}
        return {"empty": False, "ready": rd.ready(outcomes), "passing": sum(o.status == "pass" for o in outcomes),
                "total": len(outcomes),
                "criteria": [{"id": o.id, "description": desc[o.id], "status": o.status, "value": o.value,
                              "threshold": o.threshold, "detail": o.detail, "ignored": o.ignored} for o in outcomes],
                "note": "Evidence counts only from ledger runs labelled paper or live. Replays and unlabelled runs are ignored."}

    def _evidence(self) -> dict[str, list[dict[str, Any]]]:
        try:
            led = self.open()
        except LedgerUnavailable:
            return {}
        try:
            kind = self._meta(led)["source"]
            if kind == "empty":
                return {}
            mode = kind if kind in ("paper", "live") else "replay"
            real = mode in ("paper", "live")

            def item(v: float, **kw: Any) -> dict[str, Any]:
                return {"value": float(v), "mode": mode, "real_data": real, **kw}

            ev: dict[str, list[dict[str, Any]]] = {}
            # health rows: consecutive trailing clean days of reconcile/parity checks
            hs = [d for d in self.payloads(led, "SELECT seq, payload FROM events WHERE kind='health' ORDER BY seq")
                  if d.get("check") in ("reconcile", "parity")]
            if hs:
                days: dict[str, bool] = {}
                for d in hs:
                    t = parse_ts(d["time"])
                    if t:
                        k = t.astimezone(self.tz).strftime("%Y-%m-%d")
                        days[k] = days.get(k, True) and bool(d.get("ok"))
                n, cur = 0, parse_ts(hs[-1]["time"]).astimezone(self.tz).date()          # type: ignore[union-attr]
                while days.get(cur.strftime("%Y-%m-%d")):
                    n, cur = n + 1, cur - timedelta(days=1)
                ev["clean_reconcile_parity"] = [item(n)]
            closes = self.payloads(led, "SELECT seq, payload FROM events WHERE kind='close' AND book='ensemble' ORDER BY seq")
            snaps = self.payloads(led, "SELECT seq, payload FROM events WHERE kind='snapshot' AND book='ensemble' ORDER BY seq")
            if closes or snaps:
                per = {"oanda": 0, "moomoo": 0}
                for c in closes:
                    per[venue_of(c["symbol"])] += 1
                ev["paper_trades_per_venue"] = [item(min(per.values()), per_venue=per)]
            if snaps:
                ev["stops_on_every_position"] = [item(sum(1 for p in snaps[-1].get("positions", []) if p.get("stop") is None))]
                tiers = [s.get("limits", {}).get("tier") for s in snaps]
                ev["governor_tiers_tested"] = [item(sum(1 for a, b in zip(tiers, tiers[1:]) if a != b))]
            passing = [r for r in self._reports().values() if r.get("data_source") == "real"]
            if passing:
                ev["strategies_passing_real_data"] = [item(sum(1 for r in passing if r.get("gate") == "validated"))]
            appr = self._try_table(led, "SELECT COUNT(*) n FROM commands WHERE command='go_live_approved' AND source='dashboard'")
            if appr and kind != "empty":
                ev["ray_go_ahead"] = [item(appr[0]["n"])]
            # not derivable from the ledger today (no modelled cost or bootstrap band is stored): left unknown on purpose
            return ev
        finally:
            led.close()
