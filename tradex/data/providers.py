"""Historical bar providers.

Keys come from the Keychain via ``tradex.secrets`` (set with ``trade-x setup``) and are
never stored in the repo. Nothing here places orders: these are read-only market-data
endpoints.

  Alpaca:  alpaca_key_id, alpaca_secret   (free IEX feed)
  Massive: massive_key                    (free plan: EOD, 5 calls/min)
  Oanda:   oanda_token                    (practice host only; mid candles)
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Protocol

import pandas as pd

from tradex import secrets
from tradex.data.bars import normalize_bars


class BarProvider(Protocol):
    def get_bars(self, symbol: str, tf: str, start: str, end: str) -> pd.DataFrame: ...


class CsvProvider:
    """Reads ``{root}/{symbol}_{tf}.csv`` with a timestamp column plus OHLCV."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def path(self, symbol: str, tf: str) -> Path:
        return self.root / f"{symbol}_{tf}.csv"

    def get_bars(self, symbol: str, tf: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        df = pd.read_csv(self.path(symbol, tf), index_col=0)
        df = normalize_bars(df)
        if start:
            df = df[df.index >= pd.Timestamp(start, tz="UTC")]
        if end:
            df = df[df.index < pd.Timestamp(end, tz="UTC")]
        return df

    def save(self, symbol: str, tf: str, df: pd.DataFrame) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        p = self.path(symbol, tf)
        df.to_csv(p, index_label="timestamp")
        return p


def _get_json(url: str, headers: dict[str, str] | None = None, retries: int = 4) -> dict:
    delay = 2.0
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < retries:
                time.sleep(delay)
                delay *= 2
                continue
            raise
    raise RuntimeError("unreachable")


_ALPACA_TF = {"M1": "1Min", "M5": "5Min", "M15": "15Min", "M30": "30Min", "H1": "1Hour", "H4": "4Hour", "D1": "1Day"}


class AlpacaProvider:
    """Alpaca market data v2 stock bars. Free plan = IEX feed only."""

    base = "https://data.alpaca.markets/v2/stocks"

    def __init__(self, feed: str = "iex", adjustment: str = "all"):
        self.feed = feed
        self.adjustment = adjustment

    def get_bars(self, symbol: str, tf: str, start: str, end: str) -> pd.DataFrame:
        headers = {"APCA-API-KEY-ID": secrets.get("alpaca_key_id"), "APCA-API-SECRET-KEY": secrets.get("alpaca_secret")}
        params = {
            "timeframe": _ALPACA_TF[tf], "start": start, "end": end, "limit": 10000,
            "feed": self.feed, "adjustment": self.adjustment,
        }
        rows, token = [], None
        while True:
            if token:
                params["page_token"] = token
            data = _get_json(f"{self.base}/{symbol}/bars?{urllib.parse.urlencode(params)}", headers)
            rows += data.get("bars") or []
            token = data.get("next_page_token")
            if not token:
                break
            time.sleep(0.35)  # stay well under 200 calls/min
        df = pd.DataFrame(rows, columns=["t", "o", "h", "l", "c", "v"]) if not rows else pd.DataFrame(rows)
        return normalize_bars(df.rename(columns={"t": "timestamp"}).pipe(_rename_short))


_MASSIVE_TF = {"M1": (1, "minute"), "M5": (5, "minute"), "M15": (15, "minute"), "M30": (30, "minute"),
               "H1": (1, "hour"), "H4": (4, "hour"), "D1": (1, "day")}


class MassiveProvider:
    """Massive (formerly Polygon) aggregates. Free plan: 5 calls/min, about 2 years of history."""

    base = "https://api.massive.com/v2/aggs/ticker"

    def __init__(self, min_interval_s: float = 12.5):
        self.min_interval_s = min_interval_s
        self._last = 0.0

    def _throttle(self) -> None:
        wait = self.min_interval_s - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()

    def get_bars(self, symbol: str, tf: str, start: str, end: str) -> pd.DataFrame:
        mult, span = _MASSIVE_TF[tf]
        key = secrets.get("massive_key")
        url = (f"{self.base}/{urllib.parse.quote(symbol)}/range/{mult}/{span}/{start[:10]}/{end[:10]}"
               f"?adjusted=true&sort=asc&limit=50000&apiKey={key}")
        rows = []
        while url:
            self._throttle()
            data = _get_json(url)
            rows += data.get("results") or []
            nxt = data.get("next_url")
            url = f"{nxt}&apiKey={key}" if nxt else None
        df = pd.DataFrame(rows)
        if df.empty:
            df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        else:
            df["timestamp"] = pd.to_datetime(df["t"], unit="ms", utc=True)
            df = df.pipe(_rename_short)
        return normalize_bars(df)


_OANDA_TF = {"M1": "M1", "M5": "M5", "M15": "M15", "M30": "M30", "H1": "H1", "H4": "H4", "D1": "D", "W1": "W"}


class OandaProvider:
    """Oanda v20 mid candles from the practice host only. Daily candles align to 17:00 New York."""

    base = "https://api-fxpractice.oanda.com"

    def get_bars(self, symbol: str, tf: str, start: str, end: str) -> pd.DataFrame:
        headers = {"Authorization": f"Bearer {secrets.get('oanda_token')}"}
        rows = []
        cursor = pd.Timestamp(start, tz="UTC")
        stop = pd.Timestamp(end, tz="UTC")
        while cursor < stop:
            params = {"granularity": _OANDA_TF[tf], "price": "M", "from": cursor.isoformat(), "count": 5000}
            data = _get_json(f"{self.base}/v3/instruments/{symbol}/candles?{urllib.parse.urlencode(params)}", headers)
            candles = [c for c in data.get("candles", []) if c.get("complete")]
            if not candles:
                break
            for c in candles:
                m = c["mid"]
                rows.append({"timestamp": c["time"], "open": float(m["o"]), "high": float(m["h"]),
                             "low": float(m["l"]), "close": float(m["c"]), "volume": float(c["volume"])})
            nxt = pd.Timestamp(candles[-1]["time"])
            if nxt <= cursor:
                break
            cursor = nxt + pd.Timedelta(seconds=1)
        df = normalize_bars(pd.DataFrame(rows) if rows else pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"]))
        return df[df.index < stop]


def _rename_short(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})


class CachedProvider:
    """Wraps a remote provider with a CSV cache so backtests are reproducible offline."""

    def __init__(self, inner: BarProvider, cache_dir: str | Path, tag: str):
        self.inner = inner
        self.cache = CsvProvider(Path(cache_dir) / tag)

    def get_bars(self, symbol: str, tf: str, start: str, end: str) -> pd.DataFrame:
        p = self.cache.path(symbol, tf)
        if p.exists():
            df = self.cache.get_bars(symbol, tf)
            if len(df) and df.index[0] <= pd.Timestamp(start, tz="UTC") + pd.Timedelta(days=5) and \
                    df.index[-1] >= pd.Timestamp(end, tz="UTC") - pd.Timedelta(days=5):
                return df[(df.index >= pd.Timestamp(start, tz="UTC")) & (df.index < pd.Timestamp(end, tz="UTC"))]
        df = self.inner.get_bars(symbol, tf, start, end)
        self.cache.save(symbol, tf, df)
        return df
