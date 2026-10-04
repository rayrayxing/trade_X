"""Bulk FX history from Oanda v20 (practice host only): bid/ask candles, CSV-cached.

The only Oanda candle downloader in tradex (tradex.data.oanda.fetch_ba_candles wraps it).

For research: ten years of H1/H4/D bid and ask candles per pair, so backtests can pay the
spread Oanda actually quoted at each bar instead of a typical-spread guess. Only the
practice REST host is ever called. The token comes from tradex.secrets (Keychain,
``oanda_token``), imported lazily so tests and CI never need it; tests inject ``http``.

Columns saved: bid_/ask_ OHLC, mid OHLC as open..close for the engine, volume.
Documented API: GET /v3/instruments/{instrument}/candles with price=BA, granularity,
from, count (max 5000), dailyAlignment=17 and alignmentTimezone=America/New_York.
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from tradex.data.oanda import GRANULARITY, REST_HOST as PRACTICE_HOST, LiveHostRefused, check_host

HostRefused = LiveHostRefused
DEFAULT_CACHE = Path(__file__).resolve().parents[2] / "data" / "cache" / "oanda"
COLS = ["open", "high", "low", "close", "volume", "bid_open", "bid_high", "bid_low", "bid_close",
        "ask_open", "ask_high", "ask_low", "ask_close"]


def _http_get(url: str, headers: dict[str, str]) -> dict:
    if urllib.parse.urlparse(url).hostname != PRACTICE_HOST:
        raise HostRefused(f"only the Oanda practice REST host {PRACTICE_HOST} is allowed for candles")
    check_host(url)
    delay = 2.0
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 502, 503) and attempt < 4:
                time.sleep(delay)
                delay *= 2
                continue
            raise
    raise RuntimeError("unreachable")


def parse_candles(candles: Iterable[dict]) -> pd.DataFrame:
    """Complete bid/ask candles -> frame indexed by open time (UTC)."""
    rows = []
    for c in candles:
        if not c.get("complete", False):
            continue
        r = {"ts": pd.Timestamp(c["time"]), "volume": float(c["volume"])}
        for side in ("bid", "ask"):
            for k, n in (("o", "open"), ("h", "high"), ("l", "low"), ("c", "close")):
                r[f"{side}_{n}"] = float(c[side][k])
        for n in ("open", "high", "low", "close"):
            r[n] = (r[f"bid_{n}"] + r[f"ask_{n}"]) / 2
        rows.append(r)
    if not rows:
        return pd.DataFrame(columns=COLS, index=pd.DatetimeIndex([], tz="UTC", name="ts"), dtype=float)
    df = pd.DataFrame(rows).set_index("ts")
    df.index = pd.DatetimeIndex(df.index).tz_convert("UTC").astype("datetime64[ns, UTC]")
    return df[~df.index.duplicated(keep="last")].sort_index()[COLS]


class OandaHistory:
    def __init__(self, cache_dir: str | Path = DEFAULT_CACHE, token: str | None = None,
                 http: Callable[[str, dict], dict] = _http_get, page: int = 5000, pause_s: float = 0.1):
        self.cache_dir = Path(cache_dir)
        self._token = token
        self.http, self.page, self.pause_s = http, page, pause_s

    @property
    def token(self) -> str:
        if self._token is None:
            from tradex import secrets  # lane B's Keychain wrapper; raises MissingSecret when absent
            self._token = secrets.get("oanda_token")
        return self._token

    def path(self, pair: str, tf: str) -> Path:
        return self.cache_dir / f"{pair}_{tf}.csv"

    def load(self, pair: str, tf: str) -> pd.DataFrame | None:
        p = self.path(pair, tf)
        if not p.exists():
            return None
        df = pd.read_csv(p, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True).astype("datetime64[ns, UTC]")
        df.index.name = "ts"
        return df[COLS].astype(float)

    def download(self, pair: str, tf: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        headers = {"Authorization": f"Bearer {self.token}", "Accept-Datetime-Format": "RFC3339"}
        cursor, parts = start, []
        while cursor < end:
            q = {"price": "BA", "granularity": GRANULARITY[tf], "from": cursor.strftime("%Y-%m-%dT%H:%M:%SZ"),
                 "count": self.page, "dailyAlignment": 17, "alignmentTimezone": "America/New_York"}
            raw = self.http(f"https://{PRACTICE_HOST}/v3/instruments/{pair}/candles?{urllib.parse.urlencode(q)}",
                            headers).get("candles", [])
            if not raw:
                break
            parts.append(parse_candles(raw))
            last = pd.Timestamp(raw[-1]["time"]).tz_convert("UTC")
            if len(raw) < self.page or last <= cursor or not raw[-1].get("complete", False):  # reached now
                break
            cursor = last + pd.Timedelta(seconds=1)
            time.sleep(self.pause_s)
        df = pd.concat(parts) if parts else parse_candles([])
        df = df[~df.index.duplicated(keep="last")].sort_index()
        return df[df.index < end]

    def fetch(self, pair: str, tf: str, years: int = 10, end: str | pd.Timestamp | None = None) -> pd.DataFrame:
        """Cached history; extends the cache forward when it is stale."""
        end_ts = pd.Timestamp(end, tz="UTC") if isinstance(end, str) else (end or pd.Timestamp.now(tz="UTC"))
        cached = self.load(pair, tf)
        start = end_ts - pd.DateOffset(years=years)
        if cached is not None and len(cached) and cached.index[0] <= start + pd.Timedelta(days=7):
            new = self.download(pair, tf, cached.index[-1] + pd.Timedelta(seconds=1), end_ts)
            df = pd.concat([cached, new])
        else:
            df = self.download(pair, tf, start, end_ts)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(self.path(pair, tf), index_label="ts")
        return df


def spread_series(ba: pd.DataFrame, at: str = "open") -> pd.Series:
    """Quoted spread (price units) at each bar's open or close: the cost a market order pays there."""
    return (ba[f"ask_{at}"] - ba[f"bid_{at}"]).rename("spread")
