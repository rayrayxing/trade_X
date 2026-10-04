"""The four interfaces the core runs on: Clock, MarketData, Broker, Ledger.

The same core code runs in backtest, nightly replay, paper and live; only the
implementations behind these interfaces change. That is what makes the nightly
parity check possible: replaying a live day through the replay implementations must
produce the same decision rows.

This module holds the protocols plus the replay implementations of Clock and
MarketData. Broker implementations live in ``tradex/execution`` (a protected path);
the Ledger is ``tradex.core.ledger.Ledger``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Protocol

import pandas as pd

from tradex.timeframes import duration


class Clock(Protocol):
    def now(self) -> pd.Timestamp: ...


class MarketData(Protocol):
    def bars(self, symbol: str, end: pd.Timestamp | None = None, tf: str | None = None) -> pd.DataFrame:
        """Closed bars for ``symbol`` on timeframe ``tf`` (default: the store's base timeframe)
        whose close is at or before ``end`` (default: now)."""

    def last_price(self, symbol: str) -> float: ...


@dataclass
class OrderRequest:
    client_order_id: str
    decision_id: str
    symbol: str
    asset_class: str
    side: int                          # +1 buy, -1 sell
    qty: float
    order_type: str = "market"         # market | limit
    limit_price: float | None = None
    stop_loss: float | None = None     # attached on fill, like Oanda stopLossOnFill
    take_profit: float | None = None   # attached on fill, like Oanda takeProfitOnFill
    purpose: str = "entry"             # entry | exit | hedge
    book: str = "ensemble"
    account: str = "agent"
    verdict_id: str = ""               # the risk-gate verdict that sized this order; the order guard checks it


@dataclass
class BrokerPosition:
    decision_id: str
    symbol: str
    asset_class: str
    direction: int
    qty: float
    entry_price: float
    entry_time: pd.Timestamp
    stop: float | None
    take_profit: float | None
    account: str = "agent"             # "agent" or "ray": Ray's own holdings are never traded
    book: str = "ensemble"
    meta: dict = field(default_factory=dict)


@dataclass
class BrokerFill:
    client_order_id: str
    decision_id: str
    time: pd.Timestamp
    symbol: str
    side: int
    qty: float
    price: float
    fees_usd: float
    spread_slippage_usd: float
    reason: str                        # entry | stop | stop_gap | target | target_gap | exit
    book: str = "ensemble"
    fill_id: str = ""                  # unique per venue; the core de-duplicates on it when polling fills()
    net_pnl_usd: float | None = None   # closing fills: realised P&L net of fees and carry, as the venue books it
    position_closed: bool | None = None  # closing fills: True when nothing of the position is left


@dataclass
class OrderStatus:
    client_order_id: str
    status: str                        # pending | filled | cancelled | expired | rejected | unknown
    filled_qty: float = 0.0
    avg_price: float | None = None

    @property
    def open(self) -> bool:
        return self.status == "pending"


@dataclass
class AccountInfo:
    """One venue account as the venue reports it, in the account's own currency."""
    account_id: str
    venue: str
    currency: str
    equity: float                      # balance plus unrealised P&L (Oanda NAV, moomoo total assets)
    cash: float
    margin_used: float
    buying_power: float                # free margin: what new positions may still tie up


class Broker(Protocol):
    """What the core needs from any venue. Marks come from MarketData and open risk from
    positions, so nothing here is specific to the simulator."""

    def place(self, req: OrderRequest) -> str:
        """Submit an order. Idempotent on client_order_id: resubmitting returns the same ID."""

    def cancel(self, client_order_id: str) -> bool: ...

    def amend_stop(self, decision_id: str, stop: float) -> None: ...

    def positions(self, account: str | None = "agent") -> list[BrokerPosition]: ...

    def fills(self, since: pd.Timestamp | None = None) -> list[BrokerFill]:
        """Fills with ``time >= since`` (all when None), oldest first."""

    def order_status(self, client_order_id: str) -> OrderStatus: ...

    def account(self) -> AccountInfo: ...


@dataclass
class ReplayClock:
    """A clock the replay loop moves bar by bar."""
    t: pd.Timestamp = field(default_factory=lambda: pd.Timestamp("1970-01-01", tz="UTC"))

    def now(self) -> pd.Timestamp:
        return self.t

    def set(self, t: pd.Timestamp) -> None:
        self.t = t


class WallClock:
    def now(self) -> pd.Timestamp:
        return pd.Timestamp.now(tz="UTC")


class ReplayData:
    """MarketData over recorded bars. Bars are indexed by bar open time; a bar is closed
    (visible) once ``open time + bar length <= now``."""

    def __init__(self, frames: dict[str, pd.DataFrame], bar: pd.Timedelta, clock: Clock):
        self.frames = frames
        self.bar = bar
        self.clock = clock

    def bars(self, symbol: str, end: pd.Timestamp | None = None, tf: str | None = None) -> pd.DataFrame:
        if tf is not None and duration(tf) != self.bar:
            raise KeyError(f"ReplayData holds {self.bar} bars only, not {tf}")
        end = end if end is not None else self.clock.now()
        df = self.frames[symbol]
        return df.iloc[:df.index.searchsorted(end - self.bar, side="right")]

    def last_price(self, symbol: str) -> float:
        b = self.bars(symbol)
        return float(b["close"].iloc[-1]) if len(b) else float("nan")

    def timeline(self, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None
                 ) -> Iterator[tuple[pd.Timestamp, list[tuple[str, int]]]]:
        """Bar open times in order, with (symbol, row index) for each symbol that has that bar."""
        events: dict[pd.Timestamp, list[tuple[str, int]]] = {}
        for sym, df in self.frames.items():
            for i, ts in enumerate(df.index):
                if (start is None or ts >= start) and (end is None or ts < end):
                    events.setdefault(ts, []).append((sym, i))
        for ts in sorted(events):
            yield ts, events[ts]
