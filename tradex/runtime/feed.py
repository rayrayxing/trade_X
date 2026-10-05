"""The live forex feed: Oanda price stream -> quote book and bar builders -> bar store.

A reader thread (``start``) pulls ticks from the stream: each tick updates the quote book
(rates and spreads for the core) and is queued. ``before_close`` is the LiveRunner hook and
runs on the core's thread: it drains the queue into one bar builder per traded symbol,
closes every bar that ended by the close time and appends it to the bar store. Only the
core's thread touches the builders and the store.

A stream that has been silent (no tick, no heartbeat) for ``stale_after`` while forex
trades is a fault: one Health row with ok=False when it goes stale (loud on Telegram) and
one with ok=True when it recovers, not one per bar.
"""
from __future__ import annotations

import queue
import threading
from typing import Callable, Iterable

import pandas as pd

from tradex.data.barbuilder import BarBuilder, to_frame
from tradex.data.oanda import QuoteBook, Tick

NY = "America/New_York"
HealthFn = Callable[[str, bool, str, pd.Timestamp], None]


def forex_open(ts: pd.Timestamp) -> bool:
    """Forex trades from Sunday 17:00 to Friday 17:00 New York time."""
    ny = ts.tz_convert(NY)
    wd = ny.weekday()
    if wd == 5:
        return False
    if wd == 6:
        return ny.hour >= 17
    if wd == 4:
        return ny.hour < 17
    return True


class StreamFeed:
    def __init__(self, stream, quotes: QuoteBook, store, symbols: Iterable[str], tf: str,
                 health: HealthFn | None = None, stale_after: pd.Timedelta = pd.Timedelta(seconds=120),
                 clock: Callable[[], pd.Timestamp] = lambda: pd.Timestamp.now(tz="UTC"),
                 market_open: Callable[[pd.Timestamp], bool] = forex_open):
        self.stream, self.quotes, self.store = stream, quotes, store
        self.builders = {s: BarBuilder(s, tf) for s in symbols}
        self.health, self.stale_after, self.clock, self.market_open = health, stale_after, clock, market_open
        self._q: queue.Queue[Tick] = queue.Queue()
        self._last_seen: pd.Timestamp | None = clock()     # silence counts from start-up
        self._beats = getattr(stream, "heartbeats", 0)
        self.stale = False
        self.thread: threading.Thread | None = None

    def on_tick(self, tick: Tick) -> None:
        """Reader side: safe to call from the stream thread."""
        self.quotes.update(tick)
        self._last_seen = self.clock()
        self._q.put(tick)

    def start(self) -> threading.Thread:
        def run() -> None:
            for t in self.stream.ticks():
                self.on_tick(t)
        self.thread = threading.Thread(target=run, name="oanda-stream", daemon=True)
        self.thread.start()
        return self.thread

    def before_close(self, tf: str, ts: pd.Timestamp) -> None:
        bars: dict[str, list] = {}
        while True:
            try:
                t = self._q.get_nowait()
            except queue.Empty:
                break
            b = self.builders.get(t.instrument)
            if b is not None:
                bars.setdefault(t.instrument, []).extend(b.on_tick(t.time, t.bid, t.ask))
        for sym, b in self.builders.items():
            done = bars.get(sym, []) + b.flush(ts)
            if done:
                self.store.append(sym, to_frame(done))
        self._check_stale(ts)

    def _check_stale(self, ts: pd.Timestamp) -> None:
        beats = getattr(self.stream, "heartbeats", 0)
        if beats != self._beats:
            self._beats, self._last_seen = beats, self.clock()
        now = self.clock()
        silent = self._last_seen is None or now - self._last_seen > self.stale_after
        stale = silent and self.market_open(now)
        if stale != self.stale and self.health is not None:
            since = self._last_seen.isoformat() if self._last_seen is not None else "start"
            self.health("feed", not stale, f"Oanda stream silent since {since}" if stale else "Oanda stream back", ts)
        self.stale = stale
