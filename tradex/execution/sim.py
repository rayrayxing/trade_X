"""Simulated broker: the Broker interface over recorded bars.

Used by the replay harness, the virtual book of every strategy, and end-to-end tests.
Fill rules match the backtester so replay and backtest agree:

- Orders placed at a bar's close fill at the next bar's open (market) or when the next
  bar trades through the limit (limit, one-bar life by default).
- Stop and take-profit attach on fill, like Oanda's stopLossOnFill / takeProfitOnFill.
  Inside a bar the stop is checked first; a gap through the stop fills at the open.
- Every fill pays half the spread plus slippage and the order fees; open positions pay
  financing, borrow and margin interest from the same cost models.
- Client order IDs are idempotent: placing the same ID twice returns the first order.
- Positions tagged ``account="ray"`` are Ray's own holdings: marked to market and counted
  in exposure, never closed or amended.
- It satisfies the full Broker protocol (fills, order status, account), so the core treats
  it exactly like a venue adapter. Margin uses the same conservative rates as the core's
  headroom check (forex 5% = 20:1, stocks 100% = cash account).
- A missing FX rate never fills an order at a guessed rate: the order waits.

This file is under ``tradex/execution``, a protected path agents cannot change.
"""
from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from tradex.core.interfaces import AccountInfo, BrokerFill, BrokerPosition, OrderRequest, OrderStatus
from tradex.costs.models import CostModel, model_for, split_pair, usd_per_unit


@dataclass
class _Open:
    pos: BrokerPosition
    base_to_usd: float
    entry_fees: float
    spread_slip: float
    risk_usd: float
    hold: dict = field(default_factory=lambda: defaultdict(float))
    last_accrual: pd.Timestamp | None = None
    realised: float = 0.0             # net P&L already booked from partial exits


@dataclass
class ClosedLot:
    decision_id: str
    symbol: str
    time: pd.Timestamp
    qty: float
    exit_price: float
    net_pnl_usd: float
    r_multiple: float
    reason: str
    book: str
    fully_closed: bool


MARGIN_RATES = {"forex": 0.05, "stocks": 1.0}


class SimBroker:
    simulated = True                         # the core feeds simulated brokers its bars; venues fill on their own

    def __init__(self, initial_cash: float = 10_000.0, costs: dict[str, CostModel] | None = None,
                 fx_rates: dict[str, pd.Series] | None = None, bar: pd.Timedelta = pd.Timedelta(hours=1),
                 limit_bars: int = 1, book: str = "ensemble", account_id: str = "sim",
                 margin_rates: dict[str, float] | None = None,
                 rate_fn: Callable[[str, pd.Timestamp], float] | None = None):
        self._cash = float(initial_cash)
        self.costs = costs or {"stocks": model_for("stocks"), "forex": model_for("forex")}
        self.fx_rates = fx_rates
        self.rate_fn = rate_fn              # USD per unit of a currency; raises LookupError when unknown
        self.account_id = account_id
        self.venue = "sim"
        self.margin_rates = margin_rates or dict(MARGIN_RATES)
        self._fills: list[BrokerFill] = []
        self._status: dict[str, OrderStatus] = {}
        self._fill_seq = itertools.count(1)
        self.bar = bar
        self.limit_bars = limit_bars
        self.book = book
        self.orders: dict[str, OrderRequest] = {}
        self._pending: dict[str, list[tuple[OrderRequest, int]]] = defaultdict(list)
        self._open: dict[str, _Open] = {}
        self._marks: dict[str, float] = {}
        self.closed: list[ClosedLot] = []

    # --- Broker interface ------------------------------------------------------------

    def place(self, req: OrderRequest) -> str:
        if req.client_order_id in self.orders:
            return req.client_order_id
        if req.account != "agent":
            raise PermissionError("the agent never places orders on Ray's own account")
        self.orders[req.client_order_id] = req
        self._status[req.client_order_id] = OrderStatus(req.client_order_id, "pending")
        self._pending[req.symbol].append((req, 0))
        return req.client_order_id

    def cancel(self, client_order_id: str) -> bool:
        req = self.orders.get(client_order_id)
        if not req:
            return False
        before = len(self._pending[req.symbol])
        self._pending[req.symbol] = [(r, n) for r, n in self._pending[req.symbol] if r.client_order_id != client_order_id]
        done = len(self._pending[req.symbol]) < before
        if done:
            self._status[client_order_id].status = "cancelled"
        return done

    def amend_stop(self, decision_id: str, stop: float) -> None:
        op = self._open.get(decision_id)
        if op and op.pos.account == "agent":
            op.pos.stop = stop

    def positions(self, account: str | None = "agent") -> list[BrokerPosition]:
        return [o.pos for o in self._open.values() if account is None or o.pos.account == account]

    def cash(self) -> float:
        return self._cash

    def fills(self, since: pd.Timestamp | None = None) -> list[BrokerFill]:
        if since is None:
            return list(self._fills)
        i = len(self._fills)
        while i > 0 and self._fills[i - 1].time >= since:
            i -= 1
        return self._fills[i:]

    def order_status(self, client_order_id: str) -> OrderStatus:
        return self._status.get(client_order_id) or OrderStatus(client_order_id, "unknown")

    def account(self) -> AccountInfo:
        eq = self.equity()
        margin = sum(o.pos.qty * self._marks.get(o.pos.symbol, o.pos.entry_price) * o.base_to_usd
                     * self.margin_rates.get(o.pos.asset_class, 1.0)
                     for o in self._open.values() if o.pos.account == "agent")
        return AccountInfo(self.account_id, self.venue, "USD", eq, self._cash, margin, max(0.0, eq - margin))

    def equity(self) -> float:
        u = 0.0
        for o in self._open.values():
            if o.pos.account != "agent":
                continue
            px = self._marks.get(o.pos.symbol, o.pos.entry_price)
            u += o.pos.direction * (px - o.pos.entry_price) * o.pos.qty * o.base_to_usd
        return self._cash + u

    # --- extras the core uses ----------------------------------------------------------

    def add_external_position(self, pos: BrokerPosition) -> None:
        """Register one of Ray's own holdings (read-only)."""
        pos.account = "ray"
        self._open[pos.decision_id] = _Open(pos=pos, base_to_usd=1.0, entry_fees=0.0, spread_slip=0.0, risk_usd=0.0)

    def open_risk_usd(self) -> float:
        out = 0.0
        for o in self._open.values():
            p = o.pos
            if p.account == "agent" and p.stop is not None:
                px = self._marks.get(p.symbol, p.entry_price)
                out += max(0.0, p.direction * (px - p.stop)) * p.qty * o.base_to_usd
        return out

    def gross_notional_usd(self) -> float:
        return sum(o.pos.qty * self._marks.get(o.pos.symbol, o.pos.entry_price) * o.base_to_usd
                   for o in self._open.values() if o.pos.account == "agent")

    def mark(self, symbol: str) -> float | None:
        return self._marks.get(symbol)

    def initial_risk_usd(self, decision_id: str) -> float:
        o = self._open.get(decision_id)
        return o.risk_usd if o else 0.0

    def has_pending(self, symbol: str) -> bool:
        return bool(self._pending.get(symbol))

    def base_to_usd(self, symbol: str, asset_class: str, ts: pd.Timestamp) -> float:
        if asset_class == "forex":
            quote = split_pair(symbol)[1]
            if self.rate_fn is not None:
                return self.rate_fn(quote, ts)
            return usd_per_unit(quote, ts, self.fx_rates)
        return 1.0

    # --- the bar loop --------------------------------------------------------------------

    def on_bar(self, symbol: str, ts: pd.Timestamp, o: float, h: float, l: float, c: float) -> list[BrokerFill]:
        """Process one closed bar for ``symbol``: pending orders at the open, then stops and
        targets inside the bar, then carrying costs to the bar's close."""
        fills: list[BrokerFill] = []
        keep: list[tuple[OrderRequest, int]] = []
        for req, age in self._pending.pop(symbol, []):
            try:
                f = self._try_fill(req, ts, o, h, l)
            except LookupError:             # no FX rate: wait rather than fill at a guessed rate
                keep.append((req, age))
                continue
            st = self._status[req.client_order_id]
            if f is not None:
                fills.append(f)
                st.status, st.filled_qty, st.avg_price = "filled", f.qty, f.price
            elif req.order_type == "limit" and age + 1 < self.limit_bars:
                keep.append((req, age + 1))
            else:
                st.status = "expired" if req.order_type == "limit" else "rejected"
        if keep:
            self._pending[symbol] = keep

        for did, op in list(self._open.items()):
            p = op.pos
            if p.symbol != symbol or p.account != "agent":
                continue
            d = p.direction
            entered_now = p.entry_time == ts
            hit = None
            if p.stop is not None and not entered_now and d * (o - p.stop) <= 0:
                hit = ("stop_gap", o)
            elif p.take_profit is not None and not entered_now and d * (o - p.take_profit) >= 0:
                hit = ("target_gap", o)
            else:
                if p.stop is not None and ((l <= p.stop) if d > 0 else (h >= p.stop)):
                    hit = ("stop", p.stop)
                elif p.take_profit is not None and ((h >= p.take_profit) if d > 0 else (l <= p.take_profit)):
                    hit = ("target", p.take_profit)
            if hit:
                reason, px = hit
                if reason.startswith("stop") and abs(p.stop - p.entry_price) < 1e-12:
                    reason = reason.replace("stop", "breakeven")
                fills.append(self._close(did, ts, px, p.qty, reason, f"{did}-{reason}"))

        bar_end = ts + self.bar
        for op in self._open.values():
            p = op.pos
            if p.symbol != symbol or p.account != "agent":
                continue
            cm = self.costs[p.asset_class]
            hc = cm.holding_cost(p.symbol, p.direction, p.qty, c, op.last_accrual or ts, bar_end, op.base_to_usd, 0.0)
            for k, v in hc.items():
                op.hold[k] += v
                self._cash -= v
            op.last_accrual = bar_end
        self._marks[symbol] = c
        for f in fills:
            f.fill_id = f"sim-{next(self._fill_seq)}"
        self._fills.extend(fills)
        return fills

    def _try_fill(self, req: OrderRequest, ts, o, h, l) -> BrokerFill | None:
        if req.purpose == "exit":
            op = self._open.get(req.decision_id)
            if op is None:
                return None
            return self._close(req.decision_id, ts, o, min(req.qty, op.pos.qty), "exit", req.client_order_id)
        if req.order_type == "limit":
            lp = req.limit_price
            if req.side > 0 and l > lp or req.side < 0 and h < lp:
                return None
            mid = min(o, lp) if req.side > 0 else max(o, lp)
        else:
            mid = o
        cm = self.costs[req.asset_class]
        fill, unit = cm.fill(req.symbol, req.side, mid, ts)
        if req.order_type == "limit":
            fill, unit = mid, {"spread": 0.0, "slippage": 0.0}  # resting limit: no crossing cost
        fees = sum(cm.order_fees(req.symbol, req.side, req.qty, fill, ts).values())
        bu = self.base_to_usd(req.symbol, req.asset_class, ts)
        self._cash -= fees
        pos = BrokerPosition(decision_id=req.decision_id, symbol=req.symbol, asset_class=req.asset_class,
                             direction=req.side, qty=req.qty, entry_price=fill, entry_time=ts,
                             stop=req.stop_loss, take_profit=req.take_profit, book=req.book)
        risk = abs(fill - req.stop_loss) * req.qty * bu if req.stop_loss is not None else 0.0
        self._open[req.decision_id] = _Open(pos=pos, base_to_usd=bu, entry_fees=fees,
                                            spread_slip=(unit["spread"] + unit["slippage"]) * req.qty * bu,
                                            risk_usd=risk, last_accrual=ts)
        return BrokerFill(req.client_order_id, req.decision_id, ts, req.symbol, req.side, req.qty, fill, fees,
                          (unit["spread"] + unit["slippage"]) * req.qty * bu, "entry", req.book)

    def _close(self, did: str, ts, mid: float, qty: float, reason: str, coid: str) -> BrokerFill:
        op = self._open[did]
        p = op.pos
        cm = self.costs[p.asset_class]
        fill, unit = cm.fill(p.symbol, -p.direction, mid, ts)
        fees = sum(cm.order_fees(p.symbol, -p.direction, qty, fill, ts).values())
        share = qty / p.qty if p.qty else 1.0
        gross = p.direction * (fill - p.entry_price) * qty * op.base_to_usd
        hold = sum(op.hold.values()) * share
        net = gross - op.entry_fees * share - fees - hold
        self._cash += gross - fees
        full = share >= 0.999
        risk = op.risk_usd * share
        self.closed.append(ClosedLot(did, p.symbol, ts, qty, fill, net, net / risk if risk else 0.0, reason,
                                     p.book, full))
        if full:
            del self._open[did]
        else:
            p.qty -= qty
            op.entry_fees *= 1 - share
            op.risk_usd *= 1 - share
            for k in op.hold:
                op.hold[k] *= 1 - share
        return BrokerFill(coid, did, ts, p.symbol, -p.direction, qty, fill, fees,
                          (unit["spread"] + unit["slippage"]) * qty * op.base_to_usd, reason, p.book,
                          net_pnl_usd=net, position_closed=full)
