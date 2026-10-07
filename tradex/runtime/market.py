"""BarStore: the MarketData the core reads in every mode.

Replay loads whole frames; the live feed appends each bar once it has closed. Bars are
stamped with their open time and visible once ``open + duration <= now``. Higher
timeframes are resampled from the base timeframe, so an H4 bar exists only when its
last H1 bar has closed, the same rule as ``align_higher``.
"""
from __future__ import annotations

import pandas as pd

from tradex.core.interfaces import Clock
from tradex.timeframes import OHLCV, duration, resample


class BarStore:
    def __init__(self, base_tf: str, frames: dict[str, pd.DataFrame] | None = None, clock: Clock | None = None):
        self.base_tf = base_tf
        self.bar = duration(base_tf)
        self.clock = clock
        self.frames: dict[str, pd.DataFrame] = {s: df.sort_index() for s, df in (frames or {}).items()}
        self._higher: dict[tuple[str, str], tuple[int, pd.DataFrame]] = {}
        self.research = None              # live research columns (tradex.runtime.columns.LiveColumns), if any

    def symbols(self) -> list[str]:
        return list(self.frames)

    def append(self, symbol: str, bars: pd.DataFrame) -> None:
        """Add closed base-timeframe bars (index = open time) for ``symbol``."""
        old = self.frames.get(symbol)
        new = bars[OHLCV] if old is None else pd.concat([old, bars[OHLCV]])
        self.frames[symbol] = new[~new.index.duplicated(keep="last")].sort_index()

    def frame(self, symbol: str, tf: str | None = None) -> pd.DataFrame:
        base = self.frames[symbol]
        if tf is None or tf == self.base_tf:
            return base
        if duration(tf) < self.bar:
            raise KeyError(f"{tf} is finer than the base timeframe {self.base_tf}")
        hit = self._higher.get((symbol, tf))
        if hit is None or hit[0] != len(base):
            hit = (len(base), resample(base, tf))
            self._higher[(symbol, tf)] = hit
        return hit[1]

    def bars(self, symbol: str, end: pd.Timestamp | None = None, tf: str | None = None) -> pd.DataFrame:
        if symbol not in self.frames:
            return pd.DataFrame(columns=OHLCV)
        end = end if end is not None else self.clock.now()
        df = self.frame(symbol, tf)
        d = duration(tf) if tf else self.bar
        return df.iloc[:df.index.searchsorted(end - d, side="right")]

    def bar_at(self, symbol: str, tf: str, open_t: pd.Timestamp) -> pd.Series | None:
        """The bar of ``tf`` that opened at ``open_t``, if the store has it."""
        if symbol not in self.frames:
            return None
        df = self.frame(symbol, tf)
        i = df.index.searchsorted(open_t)
        return df.iloc[i] if i < len(df) and df.index[i] == open_t else None

    def last_price(self, symbol: str) -> float:
        b = self.bars(symbol)
        return float(b["close"].iloc[-1]) if len(b) else float("nan")

    def last_time(self, symbol: str) -> pd.Timestamp | None:
        """Close time of the latest visible base bar (for staleness checks)."""
        b = self.bars(symbol)
        return b.index[-1] + self.bar if len(b) else None
