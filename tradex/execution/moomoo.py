"""moomoo OpenD SIMULATE venue adapter for US stocks (protected path).

SIMULATE only: ``trd_env`` is forced to SIMULATE, REAL is refused at construction and
re-checked on every call, and before the first call the account ID is confirmed to be a
SIMULATE account with US rights in OpenD's own account list (so a mistyped secret that
points at the REAL account is refused, not traded). The account comes from the
``moomoo_sim_account_id`` secret when set, otherwise ``pick_sim_account`` selects the
single SIMULATE account with US rights (zero or several: refused).

SIMULATE accepts only market and limit orders for US stocks (no stop, stop-limit or
trailing; regular session only), so stops and targets are core-managed: the adapter keeps
the resting levels per position (``stop_levels``), ``check_stops(marks)`` says which have
been crossed, and the stop guardian (``tradex.execution.guardian``) sends the market exit
through ``place`` (the order guard), at most once per position.

Idempotency: the client order ID goes in the order ``remark``; before every submit, and
after a timeout, today's orders are searched for it and a found order is never sent again.
Fills are derived from the cumulative ``dealt_qty`` / ``dealt_avg_price`` of our orders
(``deal_list_query`` is not available on SIMULATE). Decisions, levels and orders persist
to ``state_path`` (JSON, under the gitignored data/state/) so a restart keeps its stops.
The trade context is injected (``ctx``) so tests run on a fake.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from tradex.core.interfaces import AccountInfo, BrokerFill, BrokerPosition, OrderRequest, OrderStatus
from tradex.data.opend import HOST, NY, PORT, us_code
from tradex.execution.accounts import AgentAccount
from tradex.execution.errors import (ClientIdCollision, OrderRejected, VenueAuthError, VenueError, VenueUnavailable,
                                     WrongEnvironment)
from tradex.execution.guard import OrderGuard, VenueAdapter

SIMULATE = "SIMULATE"
RET_OK = 0                                   # the SDK's documented success code
QTY_TOL = 1e-9
PENDING = {"UNSUBMITTED", "WAITING_SUBMIT", "SUBMITTING", "SUBMITTED", "FILLED_PART", "CANCELLING_PART",
           "CANCELLING_ALL", "NONE"}
CANCELLED = {"CANCELLED_ALL", "CANCELLED_PART", "DELETED"}
REJECTED = {"SUBMIT_FAILED", "FAILED", "DISABLED", "TIMEOUT", "FILL_CANCELLED"}


def pick_sim_account(ctx) -> str:
    """The single SIMULATE account with US trading rights; refuses zero or several."""
    ret, df = ctx.get_acc_list()
    if ret != RET_OK:
        raise VenueUnavailable("OpenD get_acc_list failed")
    rows = [r for _, r in df.iterrows()
            if str(r["trd_env"]).upper() == SIMULATE and "US" in list(r.get("trdmarket_auth") or [])
            and str(r.get("acc_status", "ACTIVE")).upper() != "DISABLED"]
    if len(rows) != 1:
        raise WrongEnvironment(f"expected exactly one active US SIMULATE account in OpenD, found {len(rows)}")
    return str(rows[0]["acc_id"])


def open_trade_context(host: str = HOST, port: int = PORT):
    from moomoo import OpenSecTradeContext, TrdMarket   # only on the machine that runs OpenD
    return OpenSecTradeContext(filter_trdmarket=TrdMarket.US, host=host, port=port)


@dataclass
class StopTrigger:
    decision_id: str
    symbol: str
    direction: int
    qty: float
    reason: str                        # stop | target
    level: float
    mark: float
    client_order_id: str
    book: str = "ensemble"


def _ny_to_utc(s: Any) -> pd.Timestamp:
    t = pd.Timestamp(s)
    return (t.tz_localize(NY) if t.tzinfo is None else t).tz_convert("UTC")


class MoomooSimAdapter(VenueAdapter):
    venue = "moomoo"

    def __init__(self, account_id: str, guard: OrderGuard, *, ctx=None, account: AgentAccount | None = None,
                 trd_env: str = SIMULATE, state_path: str | Path | None = None, host: str = HOST, port: int = PORT,
                 clock: Callable[[], pd.Timestamp] = lambda: pd.Timestamp.now(tz="UTC")):
        if str(trd_env).upper() != SIMULATE:
            raise WrongEnvironment(f"moomoo trd_env {trd_env!r} refused: SIMULATE only")
        if account is not None and (account.venue != "moomoo" or account.mode != "paper"
                                    or account.environment != SIMULATE):
            raise WrongEnvironment(f"account {account.name!r} is not a moomoo paper/SIMULATE account")
        if not account_id:
            raise WrongEnvironment("no moomoo SIMULATE account ID (set it or let auto-select pick one)")
        super().__init__(str(account_id), guard)
        self.trd_env = SIMULATE
        self._ctx = ctx
        self.host, self.port, self.clock = host, port, clock
        self.state_path = Path(state_path) if state_path else None
        self._verified = False
        self._armed: dict[str, str] = {}                   # guardian exit coid -> stop | target
        self.decisions: dict[str, dict] = {}               # decision ID -> symbol, direction, qty, entry, stop, target
        self.orders: dict[str, dict] = {}                  # client order ID -> our record of the order
        self._fills: list[BrokerFill] = []
        if self.state_path and self.state_path.exists():
            st = json.loads(self.state_path.read_text())
            self.decisions, self.orders = st.get("decisions", {}), st.get("orders", {})

    # --- context and environment ------------------------------------------------------------

    @property
    def ctx(self):
        if self._ctx is None:
            self._ctx = open_trade_context(self.host, self.port)
        return self._ctx

    def close(self) -> None:
        if self._ctx is not None:
            self._ctx.close()
            self._ctx = None

    def _scrub(self, msg: Any) -> str:
        return str(msg).replace(self.account_id, "<acc>")

    def _verify_account(self) -> None:
        if self._verified:
            return
        ret, df = self.ctx.get_acc_list()
        if ret != RET_OK:
            raise VenueUnavailable(f"OpenD get_acc_list failed: {self._scrub(df)}")
        row = [r for _, r in df.iterrows() if str(r["acc_id"]) == self.account_id]
        if not row or str(row[0]["trd_env"]).upper() != SIMULATE or "US" not in list(row[0].get("trdmarket_auth") or []):
            raise WrongEnvironment("the configured moomoo account is not a US SIMULATE account in OpenD; refused")
        self._verified = True

    def _call(self, name: str, *a, trading: bool = False, **kw):
        """Every OpenD trade call goes through here: SIMULATE and our account, always."""
        if self.trd_env != SIMULATE or str(kw.get("trd_env", SIMULATE)).upper() != SIMULATE:
            raise WrongEnvironment("moomoo REAL environment refused")
        self._verify_account()
        kw["trd_env"], kw["acc_id"] = SIMULATE, int(self.account_id)
        try:
            ret, data = getattr(self.ctx, name)(*a, **kw)
        except (OSError, TimeoutError) as exc:
            raise VenueUnavailable(f"OpenD {name}: {type(exc).__name__}") from None
        if ret != RET_OK:
            msg = self._scrub(data)
            low = msg.lower()
            if "unlock" in low or "login" in low:
                raise VenueAuthError(f"OpenD {name}: {msg}")
            if any(w in low for w in ("timeout", "time out", "disconnect", "network", "busy", "frequent")):
                raise VenueUnavailable(f"OpenD {name}: {msg}")
            raise (OrderRejected if trading else VenueError)(f"OpenD {name}: {msg}")
        return data

    def _save(self) -> None:
        if self.state_path:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"decisions": self.decisions, "orders": self.orders}, indent=1, default=str))
            tmp.replace(self.state_path)

    # --- orders ------------------------------------------------------------------------------

    def _today(self, code: str = "") -> list[dict]:
        df = self._call("order_list_query", code=code, refresh_cache=True)
        return [] if df is None or len(df) == 0 else df.to_dict("records")

    def _by_remark(self, coid: str, code: str = "") -> dict | None:
        hits = [o for o in self._today(code) if o.get("remark") == coid]
        return hits[-1] if hits else None

    @staticmethod
    def _side(side: int) -> str:
        return "BUY" if side > 0 else "SELL"

    def _submit(self, req: OrderRequest) -> str:
        if req.asset_class != "stocks":
            raise OrderRejected(f"{req.client_order_id}: moomoo SIMULATE adapter trades US stocks only")
        if req.order_type not in ("market", "limit"):
            raise OrderRejected(f"{req.client_order_id}: SIMULATE takes market and limit orders only")
        if len(req.client_order_id.encode()) > 64:
            raise OrderRejected(f"{req.client_order_id}: client ID longer than the 64-byte remark")
        qty = int(req.qty + QTY_TOL)
        if qty < 1 or abs(qty - req.qty) > 1e-6:
            raise OrderRejected(f"{req.client_order_id}: whole shares only (qty {req.qty})")
        code = us_code(req.symbol)
        found = self._by_remark(req.client_order_id)
        if found is not None:
            return self._adopt(req, found)
        kind = "MARKET" if req.order_type == "market" else "NORMAL"
        price = 0.0 if kind == "MARKET" else float(req.limit_price or 0.0)
        if kind == "NORMAL" and price <= 0:
            raise OrderRejected(f"{req.client_order_id}: limit order without a price")
        for _ in range(2):
            try:
                df = self._call("place_order", price, qty, code, self._side(req.side), order_type=kind,
                                remark=req.client_order_id, time_in_force="DAY", fill_outside_rth=False,
                                trading=True)
                oid = str(df["order_id"].iloc[0]) if df is not None and len(df) else ""
                self._record(req, oid)
                return req.client_order_id
            except VenueUnavailable:
                found = self._by_remark(req.client_order_id)     # landed after all? never send twice
                if found is not None:
                    return self._adopt(req, found)
        raise VenueUnavailable(f"{req.client_order_id}: OpenD did not take the order after 2 attempts")

    def _adopt(self, req: OrderRequest, found: dict) -> str:
        if (str(found.get("code")) != us_code(req.symbol) or abs(float(found.get("qty", 0)) - req.qty) > 1e-6
                or str(found.get("trd_side", "")).upper() not in ({"BUY", "BUY_BACK"} if req.side > 0 else {"SELL", "SELL_SHORT"})):
            raise ClientIdCollision(f"{req.client_order_id}: remark already used for another order today")
        if req.client_order_id not in self.orders:
            self._record(req, str(found.get("order_id", "")))
        return req.client_order_id

    def _record(self, req: OrderRequest, order_id: str) -> None:
        reason = "entry" if req.purpose != "exit" else self._armed.pop(req.client_order_id, "exit")
        self.orders[req.client_order_id] = {"order_id": order_id, "decision_id": req.decision_id, "symbol": req.symbol,
                                            "side": req.side, "qty": req.qty, "purpose": req.purpose, "reason": reason,
                                            "book": req.book, "dealt": 0.0, "avg": 0.0, "status": "SUBMITTING"}
        if req.purpose != "exit":
            self.decisions.setdefault(req.decision_id, {
                "symbol": req.symbol, "direction": req.side, "qty": 0.0, "entry_price": 0.0, "entry_time": None,
                "stop": req.stop_loss, "target": req.take_profit, "book": req.book, "fired": None, "fire_n": 0})
        self._save()

    def _order_row(self, coid: str) -> dict | None:
        rec = self.orders.get(coid)
        if rec is None:
            return None
        if rec.get("order_id"):
            df = self._call("order_list_query", order_id=rec["order_id"], refresh_cache=True)
            if df is not None and len(df):
                return df.to_dict("records")[0]
        return self._by_remark(coid, us_code(rec["symbol"]))

    def cancel(self, client_order_id: str) -> bool:
        row = self._order_row(client_order_id)
        if row is None or str(row.get("order_status")) not in PENDING - {"CANCELLING_ALL"}:
            return False
        self._call("modify_order", "CANCEL", str(row["order_id"]), 0, 0, trading=True)
        return True

    def amend_limit(self, client_order_id: str, price: float, qty: float | None = None) -> None:
        """Change the price (and optionally size, never above the original) of a resting limit order."""
        rec, row = self.orders.get(client_order_id), self._order_row(client_order_id)
        if rec is None or row is None or str(row.get("order_status")) not in PENDING:
            raise VenueError(f"{client_order_id}: no resting order to amend")
        q = rec["qty"] if qty is None else min(float(qty), rec["qty"])
        self._call("modify_order", "NORMAL", str(row["order_id"]), q, float(price), trading=True)

    def amend_stop(self, decision_id: str, stop: float) -> None:
        """Stops are core-managed on SIMULATE: move the level the guardian watches."""
        d = self.decisions.get(decision_id)
        if d is None:
            raise VenueError(f"{decision_id}: no open agent position to amend")
        d["stop"] = float(stop)
        self._save()

    def order_status(self, client_order_id: str) -> OrderStatus:
        row = self._order_row(client_order_id)
        if row is None:
            return OrderStatus(client_order_id, "unknown")
        st = str(row.get("order_status"))
        dealt, avg = float(row.get("dealt_qty") or 0.0), float(row.get("dealt_avg_price") or 0.0)
        status = ("filled" if st == "FILLED_ALL" else "pending" if st in PENDING else
                  "cancelled" if st in CANCELLED else "rejected" if st in REJECTED else "unknown")
        return OrderStatus(client_order_id, status, dealt, avg or None)

    # --- fills ---------------------------------------------------------------------------------

    def _sync(self) -> None:
        live = [c for c, r in self.orders.items() if r["status"] not in CANCELLED | REJECTED | {"FILLED_ALL"}]
        if not live:
            return
        rows = {str(o.get("remark")): o for o in self._today()}
        changed = False
        for coid in live:
            row = rows.get(coid)
            if row is None:
                continue
            rec = self.orders[coid]
            rec["status"] = str(row.get("order_status"))
            dealt, avg = float(row.get("dealt_qty") or 0.0), float(row.get("dealt_avg_price") or 0.0)
            if dealt > rec["dealt"] + QTY_TOL:
                inc = dealt - rec["dealt"]
                px = (avg * dealt - rec["avg"] * rec["dealt"]) / inc
                t = _ny_to_utc(row.get("updated_time") or row.get("create_time"))
                self._fills.append(self._apply(coid, rec, inc, px, t, f"moomoo-{rec['order_id']}-{dealt:g}"))
                rec["dealt"], rec["avg"] = dealt, avg
            if rec["purpose"] != "exit" and rec["status"] in CANCELLED | REJECTED:
                d = self.decisions.get(rec["decision_id"])
                if d is not None and d["qty"] <= QTY_TOL:        # an entry that never filled leaves nothing behind
                    del self.decisions[rec["decision_id"]]
            changed = True
        if changed:
            self._save()

    def _apply(self, coid: str, rec: dict, q: float, px: float, t: pd.Timestamp, fid: str) -> BrokerFill:
        did = rec["decision_id"]
        d = self.decisions.get(did)
        if rec["purpose"] != "exit":
            if d is not None:
                d["entry_price"] = (d["entry_price"] * d["qty"] + px * q) / (d["qty"] + q)
                d["qty"] += q
                d["entry_time"] = d["entry_time"] or t.isoformat()
            return BrokerFill(coid, did, t, rec["symbol"], rec["side"], q, px, 0.0, 0.0, "entry", rec["book"], fid)
        net, closed = None, None
        if d is not None:
            net = d["direction"] * (px - d["entry_price"]) * q
            d["qty"] = max(0.0, d["qty"] - q)
            closed = d["qty"] <= QTY_TOL
            if closed:
                del self.decisions[did]
        return BrokerFill(coid, did, t, rec["symbol"], rec["side"], q, px, 0.0, 0.0, rec["reason"], rec["book"], fid,
                          net_pnl_usd=net, position_closed=closed)

    def fills(self, since: pd.Timestamp | None = None) -> list[BrokerFill]:
        self._sync()
        return [f for f in self._fills if since is None or f.time >= since]

    # --- positions and account ------------------------------------------------------------------

    def positions(self, account: str | None = "agent") -> list[BrokerPosition]:
        self._sync()
        df = self._call("position_list_query", refresh_cache=True)
        venue: dict[str, float] = {}
        cost: dict[str, float] = {}
        for r in ([] if df is None else df.to_dict("records")):
            sym = str(r["code"]).removeprefix("US.")
            q = float(r.get("qty") or 0.0)
            if str(r.get("position_side", "LONG")).upper() == "SHORT" and q > 0:
                q = -q
            venue[sym] = venue.get(sym, 0.0) + q
            cost[sym] = float(r.get("cost_price") or 0.0)
        out = []
        mine: dict[str, float] = {}
        for did, d in self.decisions.items():
            if d["qty"] <= QTY_TOL:
                continue
            mine[d["symbol"]] = mine.get(d["symbol"], 0.0) + d["direction"] * d["qty"]
            if account in (None, "agent"):
                out.append(BrokerPosition(did, d["symbol"], "stocks", d["direction"], d["qty"], d["entry_price"],
                                          pd.Timestamp(d["entry_time"]) if d["entry_time"] else self.clock(),
                                          d.get("stop"), d.get("target"), book=d.get("book", "ensemble"),
                                          meta={"venue": "moomoo", "stops": "core-managed"}))
        for sym, q in mine.items():
            held = venue.get(sym, 0.0)
            if q * held <= 0 or abs(held) + QTY_TOL < abs(q):
                raise VenueError(f"{sym}: moomoo holds {held:g} but the agent book expects {q:g}; reconcile first")
        if account in (None, "ray"):
            for sym, held in venue.items():
                rest = held - mine.get(sym, 0.0)
                if abs(rest) > QTY_TOL:                    # not opened by tradex: reported, never traded
                    out.append(BrokerPosition(f"moomoo-{sym}", sym, "stocks", 1 if rest > 0 else -1, abs(rest),
                                              cost.get(sym, 0.0), self.clock(), None, None, account="ray",
                                              meta={"venue": "moomoo"}))
        return out

    def account(self) -> AccountInfo:
        df = self._call("accinfo_query", refresh_cache=True, currency="USD")
        r = df.to_dict("records")[0]
        num = lambda k: float(r.get(k) or 0.0) if str(r.get(k)) not in ("N/A", "nan") else 0.0  # noqa: E731
        equity, cash = num("total_assets"), num("cash")
        margin = num("initial_margin") or num("long_mv") + abs(num("short_mv"))
        # free margin, conservatively: never more than the cash a cash account would have
        free = max(0.0, min(num("power"), cash)) if num("power") else max(0.0, cash)
        ccy = str(r.get("currency") or "")
        ccy = ccy if len(ccy) == 3 and ccy.isalpha() else "USD"   # SIMULATE reports N/A; values were asked for in USD
        return AccountInfo(self.account_id, self.venue, ccy, equity, cash, margin, free)

    # --- core-managed stops ----------------------------------------------------------------------

    def stop_levels(self) -> dict[str, dict]:
        """Resting stop/target per open agent position, as the guardian watches them."""
        return {did: {"symbol": d["symbol"], "direction": d["direction"], "qty": d["qty"], "stop": d.get("stop"),
                      "target": d.get("target"), "fired": d.get("fired")}
                for did, d in self.decisions.items() if d["qty"] > QTY_TOL}

    def check_stops(self, marks: dict[str, float]) -> list[StopTrigger]:
        """Positions whose mark crossed the stop or target and whose exit has not been sent.
        A fired exit that the venue rejected or cancelled re-arms (with a new client ID)."""
        out = []
        for did, d in self.decisions.items():
            if d["qty"] <= QTY_TOL or d["symbol"] not in marks:
                continue
            if d.get("fired"):
                if self.order_status(d["fired"]).status not in ("rejected", "cancelled", "expired"):
                    continue
                d["fired"] = None
            px, k = float(marks[d["symbol"]]), d["direction"]
            hit = None
            if d.get("stop") is not None and k * (px - d["stop"]) <= 0:
                hit = ("stop", d["stop"])
            elif d.get("target") is not None and k * (px - d["target"]) >= 0:
                hit = ("target", d["target"])
            if hit:
                n = d.get("fire_n", 0)
                coid = f"{did}-{hit[0]}" + (f"-{n}" if n else "")
                self._armed[coid] = hit[0]
                out.append(StopTrigger(did, d["symbol"], k, d["qty"], hit[0], hit[1], px, coid, d.get("book", "ensemble")))
        return out

    def mark_fired(self, decision_id: str, client_order_id: str) -> None:
        d = self.decisions.get(decision_id)
        if d is not None:
            d["fired"], d["fire_n"] = client_order_id, d.get("fire_n", 0) + 1
            self._save()


def from_accounts(accounts: list[AgentAccount], guard: OrderGuard, **kw) -> MoomooSimAdapter:
    acc = [a for a in accounts if a.venue == "moomoo"]
    if len(acc) != 1:
        raise WrongEnvironment(f"expected one moomoo account in config/accounts.yaml, found {len(acc)}")
    return MoomooSimAdapter(acc[0].account_id, guard, account=acc[0], **kw)


def auto_resolver(ctx) -> Callable[[AgentAccount], str | None]:
    """For ``load_accounts(auto={"moomoo": auto_resolver(ctx)})``."""
    return lambda acc: pick_sim_account(ctx) if acc.environment == SIMULATE else None
