"""News feed adapters for the scout: Alpaca news and Massive (ex-Polygon) news.

Both implement ``tradex.scout.base.NewsFeed``. The HTTP call is injected (``HttpGet``) so
tests run on recorded fixtures; the default goes through the same urllib helper the bar
providers use, with keys read from the Keychain via ``tradex.secrets`` (``alpaca_key_id``,
``alpaca_secret``, ``massive_key``). Read-only endpoints; nothing here can place an order.

Point in time: an item counts only if it was published at or before ``end``; the feeds
filter again after parsing, so a server that ignores the ``end`` parameter cannot leak the
future into a replay. Alpaca items use ``created_at`` (``updated_at`` can be edited later).
"""
from __future__ import annotations

import time as _time
import urllib.parse
from typing import Callable, Sequence

import pandas as pd

from tradex.scout.base import NewsFeed, NewsItem

HttpGet = Callable[[str, dict[str, str]], dict]


class NewsError(Exception):
    """A news request failed. The message never contains a URL or a key."""


def urllib_get(url: str, headers: dict[str, str]) -> dict:
    from tradex.data.providers import _get_json  # lazy: keeps import cheap
    return _get_json(url, headers)


def _secret(name: str) -> str:
    from tradex import secrets
    return secrets.get(name)


def _utc(x: str) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _z(t: pd.Timestamp) -> str:
    return _utc(str(t)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _norm_headline(s: str) -> str:
    return " ".join(s.lower().split())


class AlpacaNewsFeed:
    """Alpaca market-data news: ``GET /v1beta1/news`` (free with any market-data key).

    Articles tagged with more than ``max_symbols_per_article`` tickers are roundups and are
    skipped; they say little about any one name.
    """

    base = "https://data.alpaca.markets/v1beta1/news"

    def __init__(self, get: HttpGet = urllib_get, secret: Callable[[str], str] = _secret, limit: int = 50,
                 max_pages: int = 5, symbols_per_call: int = 50, max_symbols_per_article: int = 6,
                 sleep: Callable[[float], None] = _time.sleep, page_pause_s: float = 0.35):
        self.get, self.secret, self.limit, self.max_pages = get, secret, limit, max_pages
        self.symbols_per_call, self.max_syms = symbols_per_call, max_symbols_per_article
        self.sleep, self.page_pause_s = sleep, page_pause_s

    def _headers(self) -> dict[str, str]:
        return {"APCA-API-KEY-ID": self.secret("alpaca_key_id"), "APCA-API-SECRET-KEY": self.secret("alpaca_secret")}

    def fetch(self, start: pd.Timestamp, end: pd.Timestamp, symbols: list[str] | None) -> list[NewsItem]:
        want = set(symbols) if symbols else None
        chunks: Sequence[list[str] | None] = (
            [list(symbols)[i:i + self.symbols_per_call] for i in range(0, len(symbols), self.symbols_per_call)]
            if symbols else [None])
        end_t, out, seen = _utc(str(end)), [], set()
        headers = self._headers()
        for chunk in chunks:
            token = None
            for page in range(self.max_pages):
                params = {"start": _z(start), "end": _z(end), "limit": self.limit, "sort": "asc",
                          "include_content": "false"}
                if chunk:
                    params["symbols"] = ",".join(chunk)
                if token:
                    params["page_token"] = token
                try:
                    data = self.get(f"{self.base}?{urllib.parse.urlencode(params)}", headers)
                except Exception as exc:  # noqa: BLE001 - no URL, no key in the message
                    raise NewsError(f"alpaca news request failed ({type(exc).__name__})") from None
                for art in data.get("news") or []:
                    syms = [s for s in art.get("symbols") or [] if want is None or s in want]
                    if not syms or len(art.get("symbols") or []) > self.max_syms:
                        continue
                    try:
                        pub = _utc(art["created_at"])
                    except (KeyError, ValueError):
                        continue
                    if pub > end_t or pub < _utc(str(start)):
                        continue
                    head = str(art.get("headline") or "").strip()
                    if not head:
                        continue
                    for s in syms:
                        key = (s, _norm_headline(head))
                        if key in seen:
                            continue
                        seen.add(key)
                        out.append(NewsItem(s, pub, head, f"alpaca:{art.get('source') or 'unknown'}",
                                            str(art.get("url") or "")))
                token = data.get("next_page_token")
                if not token:
                    break
                self.sleep(self.page_pause_s)
        return sorted(out, key=lambda i: (i.published, i.symbol))


_SENT = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}


class MassiveNewsFeed:
    """Massive (formerly Polygon) news: ``GET /v2/reference/news``. Free plan: 5 calls/min.

    One market-wide window query (not one call per ticker) filtered to the requested symbols
    locally, so the free tier's call budget covers a whole universe. Massive's own per-ticker
    ``insights.sentiment`` becomes ``NewsItem.sentiment`` (-1, 0, +1).
    """

    base = "https://api.massive.com/v2/reference/news"

    def __init__(self, get: HttpGet = urllib_get, secret: Callable[[str], str] = _secret, limit: int = 1000,
                 max_pages: int = 3, min_interval_s: float = 12.5, sleep: Callable[[float], None] = _time.sleep,
                 clock: Callable[[], float] = _time.monotonic, max_symbols_per_article: int = 6):
        self.get, self.secret, self.limit, self.max_pages = get, secret, limit, max_pages
        self.min_interval_s, self.sleep, self.clock, self.max_syms = min_interval_s, sleep, clock, max_symbols_per_article
        self._last: float | None = None

    def _throttle(self) -> None:
        if self._last is not None:
            wait = self.min_interval_s - (self.clock() - self._last)
            if wait > 0:
                self.sleep(wait)
        self._last = self.clock()

    def fetch(self, start: pd.Timestamp, end: pd.Timestamp, symbols: list[str] | None) -> list[NewsItem]:
        want = set(symbols) if symbols else None
        key = self.secret("massive_key")
        params = {"published_utc.gte": _z(start), "published_utc.lte": _z(end), "order": "asc",
                  "sort": "published_utc", "limit": self.limit}
        url: str | None = f"{self.base}?{urllib.parse.urlencode(params)}&apiKey={key}"
        end_t, start_t, out, seen = _utc(str(end)), _utc(str(start)), [], set()
        for _ in range(self.max_pages):
            if not url:
                break
            self._throttle()
            try:
                data = self.get(url, {})
            except Exception as exc:  # noqa: BLE001 - the URL carries the key, so it is not repeated
                raise NewsError(f"massive news request failed ({type(exc).__name__})") from None
            for art in data.get("results") or []:
                tickers = art.get("tickers") or []
                if len(tickers) > self.max_syms:
                    continue
                try:
                    pub = _utc(art["published_utc"])
                except (KeyError, ValueError):
                    continue
                if pub > end_t or pub < start_t:
                    continue
                head = str(art.get("title") or "").strip()
                if not head:
                    continue
                sent = {i.get("ticker"): _SENT.get(str(i.get("sentiment")).lower())
                        for i in art.get("insights") or [] if isinstance(i, dict)}
                publisher = (art.get("publisher") or {}).get("name") or "unknown"
                for s in tickers:
                    if want is not None and s not in want:
                        continue
                    k = (s, _norm_headline(head))
                    if k in seen:
                        continue
                    seen.add(k)
                    out.append(NewsItem(s, pub, head, f"massive:{publisher}", str(art.get("article_url") or ""),
                                        sent.get(s)))
            nxt = data.get("next_url")
            url = f"{nxt}&apiKey={key}" if nxt else None
        return sorted(out, key=lambda i: (i.published, i.symbol))


class CombinedNewsFeed:
    """Several feeds as one. A feed that fails is skipped and named in ``errors``; the same
    headline for the same symbol from two feeds counts once (the first feed's copy wins)."""

    def __init__(self, feeds: Sequence[NewsFeed]):
        self.feeds = list(feeds)
        self.errors: list[str] = []

    def fetch(self, start: pd.Timestamp, end: pd.Timestamp, symbols: list[str] | None) -> list[NewsItem]:
        self.errors = []
        out, seen = [], set()
        for f in self.feeds:
            try:
                items = f.fetch(start, end, symbols)
            except Exception as exc:  # noqa: BLE001
                self.errors.append(f"{type(f).__name__}: {type(exc).__name__}: {exc}")
                continue
            for it in items:
                k = (it.symbol, _norm_headline(it.headline))
                if k not in seen:
                    seen.add(k)
                    out.append(it)
        return sorted(out, key=lambda i: (i.published, i.symbol))
