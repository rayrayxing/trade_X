"""Turns ticks into closed bars (M1..H1) on mid price, one builder per instrument.

Guarantees: a bar closes exactly once and is never reopened; a tick older than the
current bar is logged and dropped (never merged into a closed bar); a missing stretch
is recorded as a Gap and flagged on the next bar, never filled. Bars follow the engine
convention: stamped with open time, covering [ts, ts + duration).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from tradex.timeframes import duration

log = logging.getLogger("tradex.data.bars")


@dataclass(frozen=True)
class Bar:
    instrument: str
    tf: str
    ts: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float                 # tick count
    bid_close: float
    ask_close: float
    after_gap: bool = False       # previous bar(s) missing: indicators over this boundary are suspect


@dataclass(frozen=True)
class Gap:
    instrument: str
    tf: str
    start: pd.Timestamp           # first missing bar open
    end: pd.Timestamp             # open of the bar that resumed


@dataclass
class _Open:
    ts: pd.Timestamp
    o: float
    h: float
    l: float
    c: float
    n: int
    bid: float
    ask: float


@dataclass
class BarBuilder:
    instrument: str
    tf: str = "M1"
    late_ticks: int = 0
    gaps: list[Gap] = field(default_factory=list)
    _cur: _Open | None = None
    _last_closed: pd.Timestamp | None = None
    _gap_pending: bool = False

    def __post_init__(self):
        self.dur = duration(self.tf)
        if self.dur > pd.Timedelta(hours=1):
            raise ValueError(f"{self.tf}: only M1..H1 bars are built from ticks (H4/D1 align to 17:00 New York)")

    def _bucket(self, t: pd.Timestamp) -> pd.Timestamp:
        return t.floor(self.dur)

    def _close(self) -> Bar:
        c = self._cur
        bar = Bar(self.instrument, self.tf, c.ts, c.o, c.h, c.l, c.c, float(c.n), c.bid, c.ask, self._gap_pending)
        self._last_closed, self._cur, self._gap_pending = c.ts, None, False
        return bar

    def on_tick(self, t: pd.Timestamp, bid: float, ask: float) -> list[Bar]:
        """Feed one tick; returns the bar(s) this tick closed (at most one)."""
        t = t.tz_convert("UTC") if t.tzinfo else t.tz_localize("UTC")
        b, mid, out = self._bucket(t), (bid + ask) / 2, []
        if self._cur is not None and b < self._cur.ts or (self._cur is None and self._last_closed is not None and b <= self._last_closed):
            self.late_ticks += 1
            log.warning("late tick dropped", extra={"instrument": self.instrument, "tf": self.tf,
                                                    "tick_time": t.isoformat(), "bar_open": str(self._cur.ts if self._cur else self._last_closed)})
            return out
        if self._cur is not None and b > self._cur.ts:
            out.append(self._close())
        if self._cur is None:
            if self._last_closed is not None and b > self._last_closed + self.dur:
                self.gaps.append(Gap(self.instrument, self.tf, self._last_closed + self.dur, b))
                self._gap_pending = True
                log.warning("bar gap", extra={"instrument": self.instrument, "tf": self.tf,
                                              "from": str(self._last_closed + self.dur), "to": str(b)})
            self._cur = _Open(b, mid, mid, mid, mid, 0, bid, ask)
        c = self._cur
        c.h, c.l, c.c, c.n, c.bid, c.ask = max(c.h, mid), min(c.l, mid), mid, c.n + 1, bid, ask
        return out

    def flush(self, now: pd.Timestamp) -> list[Bar]:
        """Close the open bar if the clock has passed its end, without waiting for the next tick."""
        if self._cur is not None and now >= self._cur.ts + self.dur:
            return [self._close()]
        return []


def to_frame(bars: list[Bar]) -> pd.DataFrame:
    """OHLCV frame (engine convention) plus bid/ask close and the after_gap flag."""
    df = pd.DataFrame([{"ts": b.ts, "open": b.open, "high": b.high, "low": b.low, "close": b.close,
                        "volume": b.volume, "bid_close": b.bid_close, "ask_close": b.ask_close,
                        "after_gap": b.after_gap} for b in bars])
    return df.set_index("ts") if len(df) else df
