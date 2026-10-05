"""Oanda v20 practice market data: price stream, bid/ask candles, live quote book.

Practice hosts only (api-fxpractice / stream-fxpractice); any other host is refused, so
this module cannot reach a live account. Read-only: no order endpoints appear here.
HTTP is injected (``connect`` / ``http``) so tests run on recorded responses.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator

import pandas as pd

from tradex import secrets
from tradex.costs.models import RateMissing, pip_size, split_pair

log = logging.getLogger("tradex.data.oanda")

REST_HOST = "api-fxpractice.oanda.com"
STREAM_HOST = "stream-fxpractice.oanda.com"
_ALLOWED = {REST_HOST, STREAM_HOST}
GRANULARITY = {"M1": "M1", "M5": "M5", "M15": "M15", "M30": "M30", "H1": "H1", "H4": "H4", "D1": "D", "W1": "W"}
# Oanda quotes these as CCY_USD (True) or USD_CCY (False)
_USD_PAIR = {"EUR": "EUR_USD", "GBP": "GBP_USD", "AUD": "AUD_USD", "NZD": "NZD_USD",
             "JPY": "USD_JPY", "CHF": "USD_CHF", "CAD": "USD_CAD", "SGD": "USD_SGD"}


class LiveHostRefused(RuntimeError):
    pass


def check_host(url: str) -> str:
    host = urllib.parse.urlparse(url).hostname
    if host not in _ALLOWED:
        raise LiveHostRefused(f"refusing {host!r}: only Oanda practice hosts {sorted(_ALLOWED)} are allowed")
    return url


def _http_get(url: str, headers: dict[str, str]) -> dict:
    req = urllib.request.Request(check_host(url), headers=headers)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


# --- candles -----------------------------------------------------------------------------
# One implementation of bid/ask candle download lives in tradex.data.oanda_history (paging,
# retries, CSV cache, practice host only); these names are kept for existing callers.

def parse_ba_candles(candles: Iterable[dict]) -> pd.DataFrame:
    """Complete candles only. ``open..close`` are mid (average of bid and ask) for the engine."""
    from tradex.data.oanda_history import parse_candles
    return parse_candles(candles)


def fetch_ba_candles(instrument: str, tf: str, start: str, end: str, *, token: str | None = None,
                     http: Callable[[str, dict], dict] = _http_get, page: int = 5000) -> pd.DataFrame:
    """Bid/ask/mid candles (price=BA) over [start, end), uncached; see OandaHistory.download."""
    from tradex.data.oanda_history import OandaHistory
    h = OandaHistory(token=token, http=http, page=page, pause_s=0.0)
    return h.download(instrument, tf, pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"))


def measured_spread_pips(ba: pd.DataFrame, instrument: str) -> float:
    """Median closing bid/ask spread in pips: the measured replacement for default spreads in backtests."""
    if ba.empty:
        raise RateMissing(f"no bid/ask candles to measure the {instrument} spread")
    return float(((ba["ask_close"] - ba["bid_close"]) / pip_size(instrument)).median())


# --- price stream ------------------------------------------------------------------------

@dataclass(frozen=True)
class Tick:
    instrument: str
    time: pd.Timestamp
    bid: float
    ask: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2


def parse_stream_line(line: str | bytes) -> Tick | str | None:
    """Tick for a tradeable PRICE, "heartbeat" for HEARTBEAT, None for anything ignorable."""
    if isinstance(line, bytes):
        line = line.decode()
    line = line.strip()
    if not line:
        return None
    msg = json.loads(line)
    t = msg.get("type")
    if t == "HEARTBEAT":
        return "heartbeat"
    if t == "PRICE" and msg.get("bids") and msg.get("asks") and msg.get("tradeable", True):
        return Tick(msg["instrument"], pd.Timestamp(msg["time"]).tz_convert("UTC"),
                    float(msg["bids"][0]["price"]), float(msg["asks"][0]["price"]))
    return None


class PriceStream:
    """Streams ticks and reconnects on drop or silence.

    Oanda sends a heartbeat every 5 s, so ``heartbeat_timeout`` (7 s) of silence means the
    connection is dead; the first reconnect is immediate and later retries are at most
    2 s apart, so a drop heals well inside 10 s whenever Oanda is reachable.
    """

    def __init__(self, instruments: list[str], account_id: str | None = None, token: str | None = None, *,
                 connect: Callable[[], Iterator[bytes | str]] | None = None, heartbeat_timeout: float = 7.0,
                 retry_delays: tuple[float, ...] = (0.0, 1.0, 2.0), sleep: Callable[[float], None] = time.sleep):
        self.instruments = instruments
        self.heartbeat_timeout = heartbeat_timeout
        self.retry_delays = retry_delays
        self._sleep = sleep
        self._account, self._token = account_id, token
        self._connect = connect or self._open
        self.reconnects = 0
        self.heartbeats = 0

    def _open(self) -> Iterator[bytes]:
        acct = self._account or secrets.get("oanda_account_id")
        url = (f"https://{STREAM_HOST}/v3/accounts/{acct}/pricing/stream?"
               + urllib.parse.urlencode({"instruments": ",".join(self.instruments)}))
        req = urllib.request.Request(check_host(url), headers={"Authorization": f"Bearer {self._token or secrets.get('oanda_token')}"})
        resp = urllib.request.urlopen(req, timeout=self.heartbeat_timeout)   # per-read timeout = silence detector
        return iter(resp)

    def ticks(self, max_reconnects: int | None = None) -> Iterator[Tick]:
        failures = 0
        while True:
            try:
                for line in self._connect():
                    got = parse_stream_line(line)
                    failures = 0
                    if got == "heartbeat":
                        self.heartbeats += 1
                    elif got is not None:
                        yield got
                reason = "stream ended"
            except LiveHostRefused:
                raise
            except (OSError, ValueError) as exc:        # timeouts, resets, HTTP errors, torn JSON lines
                reason = type(exc).__name__
            self.reconnects += 1
            if max_reconnects is not None and self.reconnects > max_reconnects:
                return
            delay = self.retry_delays[min(failures, len(self.retry_delays) - 1)]
            failures += 1
            log.warning("price stream dropped; reconnecting", extra={"reason": reason, "delay_s": delay})
            self._sleep(delay)


# --- live quote book (RateSource) --------------------------------------------------------

class QuoteBook:
    """Latest bid/ask per instrument from the stream; the paper/live source of FX rates and spreads."""

    def __init__(self, max_age_s: float = 30.0, clock: Callable[[], pd.Timestamp] = lambda: pd.Timestamp.now(tz="UTC")):
        self.max_age = pd.Timedelta(seconds=max_age_s)
        self._clock = clock
        self._q: dict[str, Tick] = {}

    def update(self, tick: Tick) -> None:
        self._q[tick.instrument] = tick

    def _fresh(self, instrument: str) -> Tick | None:
        t = self._q.get(instrument)
        return t if t is not None and self._clock() - t.time <= self.max_age else None

    def spread_pips(self, symbol: str, ts=None) -> float:
        t = self._fresh(symbol)
        if t is None:
            raise RateMissing(f"no fresh Oanda quote for {symbol}")
        return (t.ask - t.bid) / pip_size(symbol)

    def usd_per_unit(self, ccy: str, ts=None) -> float:
        if ccy == "USD":
            return 1.0
        pair = _USD_PAIR.get(ccy)
        t = self._fresh(pair) if pair else None
        if t is None:
            raise RateMissing(f"no fresh Oanda quote to convert {ccy} to USD")
        return t.mid if pair.startswith(ccy) else 1.0 / t.mid


def instruments_to_stream(symbols: Iterable[str]) -> list[str]:
    """Symbols plus the USD pairs needed to convert their currencies to USD."""
    need = set(symbols)
    for s in list(need):
        need.update(_USD_PAIR[c] for c in split_pair(s) if c != "USD" and c in _USD_PAIR)
    return sorted(need)
