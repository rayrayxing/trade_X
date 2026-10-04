"""US stock bars from moomoo OpenD (read-only quote context).

Only ``OpenQuoteContext`` is ever opened here; this module must never open a trade
context (a test checks the source). OpenD runs on Ray's Mac at 127.0.0.1:11111.

Limits that shape the code (moomoo OpenAPI docs): historical candlesticks count against
a quota of distinct symbols per 30 days (300 on this account), and history requests are
limited to 60 per 30 seconds. So the fetcher checks the quota before touching a new
symbol, refuses to go past a caller-set budget, paces requests, and caches every result
to CSV (data/cache/opend/, gitignored) so a symbol is downloaded once.

Prices are forward-adjusted (qfq). OpenD stamps intraday bars with their END time in New
York; tradex stamps bars with their OPEN time in UTC, so a 60-minute bar is shifted back
one hour (the 15:30-16:00 half bar becomes 15:00, which keeps "known at" = 16:00 right).
Daily bars are stamped 00:00 New York, the same convention as the Alpaca daily feed.
"""
from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Callable

import pandas as pd

from tradex.data.bars import normalize_bars
from tradex.data.providers import CsvProvider

NY = "America/New_York"
HOST, PORT = "127.0.0.1", 11111
DEFAULT_CACHE = Path(__file__).resolve().parents[2] / "data" / "cache" / "opend"
KTYPES = {"D1": "K_DAY", "H1": "K_60M", "M30": "K_30M", "M15": "K_15M", "M5": "K_5M", "M1": "K_1M", "W1": "K_WEEK"}
BAR_END_LABELLED = {"H1": pd.Timedelta(hours=1), "M30": pd.Timedelta(minutes=30), "M15": pd.Timedelta(minutes=15),
                    "M5": pd.Timedelta(minutes=5), "M1": pd.Timedelta(minutes=1)}


class QuotaExceeded(RuntimeError):
    pass


class OpenDError(RuntimeError):
    pass


def us_code(symbol: str) -> str:
    return symbol if symbol.startswith("US.") else f"US.{symbol}"


def to_bars(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """OpenD kline frame -> tradex bars (open-time UTC index, float OHLCV)."""
    if df is None or df.empty:
        return normalize_bars(pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"]))
    t = pd.to_datetime(df["time_key"]).dt.tz_localize(NY, ambiguous="infer", nonexistent="shift_forward")
    if tf in BAR_END_LABELLED:
        t = t - BAR_END_LABELLED[tf]
    out = pd.DataFrame({"timestamp": t.dt.tz_convert("UTC"), **{c: df[c].astype(float) for c in
                                                                ("open", "high", "low", "close", "volume")}})
    return normalize_bars(out)


class RateLimiter:
    """At most ``n`` calls in any ``window_s`` seconds (OpenD: 60 history requests / 30 s)."""

    def __init__(self, n: int = 50, window_s: float = 30.0, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self.n, self.window_s, self.clock, self.sleep = n, window_s, clock, sleep
        self.calls: deque[float] = deque()

    def wait(self) -> None:
        now = self.clock()
        while self.calls and now - self.calls[0] >= self.window_s:
            self.calls.popleft()
        if len(self.calls) >= self.n:
            self.sleep(self.window_s - (now - self.calls[0]) + 0.05)
            now = self.clock()
            while self.calls and now - self.calls[0] >= self.window_s:
                self.calls.popleft()
        self.calls.append(now)


class OpenDFetcher:
    """Pages ``request_history_kline`` for one symbol at a time; CSV-cached.

    ``ctx`` is injectable for tests; by default a quote context is opened lazily (the
    moomoo package is only needed on the machine that runs OpenD).
    """

    def __init__(self, cache_dir: str | Path = DEFAULT_CACHE, ctx=None, max_new_symbols: int = 60,
                 limiter: RateLimiter | None = None, host: str = HOST, port: int = PORT):
        self.cache = CsvProvider(cache_dir)
        self._ctx = ctx
        self._own_ctx = ctx is None
        self.max_new_symbols = max_new_symbols
        self.limiter = limiter or RateLimiter()
        self.host, self.port = host, port
        self.new_symbols: list[str] = []

    @property
    def ctx(self):
        if self._ctx is None:
            from moomoo import OpenQuoteContext  # quote context only, never a trade context
            self._ctx = OpenQuoteContext(host=self.host, port=self.port)
        return self._ctx

    def close(self) -> None:
        if self._own_ctx and self._ctx is not None:
            self._ctx.close()
            self._ctx = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def quota(self) -> tuple[int, int, set[str]]:
        """(used, remaining, codes already downloaded in the last 30 days)."""
        ret, data = self.ctx.get_history_kl_quota(get_detail=True)
        if ret != 0:
            raise OpenDError(f"get_history_kl_quota failed: {data}")
        used, remaining, detail = data
        return int(used), int(remaining), {d["code"] for d in (detail or [])}

    def _check_quota(self, code: str) -> None:
        used, remaining, seen = self.quota()
        if code in seen:
            return
        if len(self.new_symbols) >= self.max_new_symbols:
            raise QuotaExceeded(f"{code}: run budget of {self.max_new_symbols} new symbols reached")
        if remaining <= 0:
            raise QuotaExceeded(f"{code}: OpenD history quota exhausted ({used} used in the last 30 days)")
        self.new_symbols.append(code)

    def fetch(self, symbol: str, tf: str, start: str = "1990-01-01", end: str | None = None,
              refresh: bool = False) -> pd.DataFrame:
        p = self.cache.path(symbol, tf)
        if p.exists() and not refresh:
            return self.cache.get_bars(symbol, tf)
        code = us_code(symbol)
        self._check_quota(code)
        end = end or pd.Timestamp.now(tz=NY).strftime("%Y-%m-%d")
        frames, key = [], None
        while True:
            self.limiter.wait()
            ret, df, key = self.ctx.request_history_kline(code, start=start, end=end, ktype=KTYPES[tf],
                                                          autype="qfq", max_count=1000, page_req_key=key)
            if ret != 0:
                raise OpenDError(f"{code} {tf}: {df}")
            frames.append(df)
            if key is None:
                break
        bars = to_bars(pd.concat(frames, ignore_index=True) if frames else None, tf)
        self.cache.save(symbol, tf, bars)
        return bars


def load_cached(symbol: str, tf: str, cache_dir: str | Path = DEFAULT_CACHE) -> pd.DataFrame | None:
    p = CsvProvider(cache_dir)
    return p.get_bars(symbol, tf) if p.path(symbol, tf).exists() else None
