"""Daily market scout: picks 10 to 20 stocks to watch each day.

The scout combines several sources, each scoring candidate symbols as of a point
in time. Sources must never use information published after ``asof``, so the
scout itself can be replayed in backtests (``scout_mask``) and its picks judged.

Built here:
  TechnicalSource  - unusual volume, gaps, momentum, volatility and liquidity from bars
Interfaces for thread 3 to fill in:
  NewsSource       - headlines and catalysts (Alpaca news API, Massive news)
  ChatterSource    - social chatter (Reddit, StockTwits, X)
  LLMReviewer      - Claude reads the day's headlines and scores catalysts
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from tradex.data.providers import BarProvider


@dataclass
class SourceScore:
    score: float                 # roughly -3..+3, higher = more interesting today
    reason: str
    direction: int = 0           # +1 bullish catalyst, -1 bearish, 0 unknown


@dataclass
class WatchItem:
    symbol: str
    score: float
    direction_bias: int
    reasons: list[str]
    sources: list[str]


@dataclass
class Watchlist:
    asof: str
    items: list[WatchItem]
    always_on: list[str] = field(default_factory=list)

    @property
    def symbols(self) -> list[str]:
        return [i.symbol for i in self.items]

    def to_dict(self) -> dict:
        return asdict(self)


class ScoutSource(Protocol):
    name: str

    def score(self, asof: pd.Timestamp, symbols: list[str]) -> dict[str, SourceScore]: ...


@dataclass
class NewsItem:
    symbol: str
    published: pd.Timestamp
    headline: str
    source: str
    url: str = ""
    sentiment: float | None = None


class NewsFeed(Protocol):
    def fetch(self, start: pd.Timestamp, end: pd.Timestamp, symbols: list[str] | None) -> list[NewsItem]: ...


class LLMReviewer(Protocol):
    """Reads headlines for a symbol and returns a catalyst score with a one-line reason."""

    def review(self, symbol: str, items: list[NewsItem]) -> SourceScore: ...


class NewsSource:
    """Scores symbols by recent news flow. Needs a NewsFeed (thread 3: Alpaca /v1beta1/news)
    and optionally an LLMReviewer; without a reviewer it scores headline count only."""

    name = "news"

    def __init__(self, feed: NewsFeed, reviewer: LLMReviewer | None = None, lookback: pd.Timedelta = pd.Timedelta(hours=24)):
        self.feed, self.reviewer, self.lookback = feed, reviewer, lookback

    def score(self, asof, symbols):
        items = self.feed.fetch(asof - self.lookback, asof, symbols)
        by_sym: dict[str, list[NewsItem]] = {}
        for it in items:
            if it.published <= asof:
                by_sym.setdefault(it.symbol, []).append(it)
        out = {}
        for sym, its in by_sym.items():
            if self.reviewer:
                out[sym] = self.reviewer.review(sym, its)
            else:
                out[sym] = SourceScore(min(3.0, np.log1p(len(its))), f"{len(its)} headlines in {self.lookback}")
        return out


class ChatterSource:
    """Placeholder for social chatter volume and sentiment. Thread 3 picks the provider."""

    name = "chatter"

    def score(self, asof, symbols):
        raise NotImplementedError("chatter source not connected yet")


class TechnicalSource:
    """Ranks liquid stocks by how much is happening in their bars, using only data up to ``asof``."""

    name = "technical"

    def __init__(self, provider: BarProvider | None = None, bars: dict[str, pd.DataFrame] | None = None,
                 min_price: float = 5.0, min_dollar_volume: float = 20e6, lookback_days: int = 300):
        self.provider, self.bars = provider, bars or {}
        self.min_price, self.min_dollar_volume, self.lookback_days = min_price, min_dollar_volume, lookback_days

    def _bars(self, sym: str, asof: pd.Timestamp) -> pd.DataFrame | None:
        if sym in self.bars:
            df = self.bars[sym]
        elif self.provider is not None:
            df = self.provider.get_bars(sym, "D1", str((asof - pd.Timedelta(days=self.lookback_days)).date()), str(asof.date()))
        else:
            return None
        return df[df.index < asof]  # bars stamped before asof are complete by asof for daily data

    def score(self, asof, symbols):
        out = {}
        for sym in symbols:
            df = self._bars(sym, asof)
            if df is None or len(df) < 60:
                continue
            c, v = df["close"], df["volume"]
            last = df.iloc[-1]
            dollar_vol = float((c * v).iloc[-20:].mean())
            if last["close"] < self.min_price or dollar_vol < self.min_dollar_volume:
                continue
            rel_vol = float(v.iloc[-1] / v.iloc[-21:-1].mean())
            gap = float(last["open"] / c.iloc[-2] - 1)
            mom20 = float(c.iloc[-1] / c.iloc[-21] - 1)
            tr = np.maximum(df["high"] - df["low"], np.maximum((df["high"] - c.shift()).abs(), (df["low"] - c.shift()).abs()))
            atr_pct = float(tr.iloc[-14:].mean() / c.iloc[-1])
            hi52 = float(c.iloc[-252:].max())
            near_high = c.iloc[-1] >= 0.97 * hi52
            score = 0.0
            reasons = []
            if rel_vol > 1.5:
                score += min(2.0, np.log2(rel_vol))
                reasons.append(f"volume {rel_vol:.1f}x average")
            if abs(gap) > 0.02:
                score += min(1.5, abs(gap) * 25)
                reasons.append(f"gap {gap:+.1%}")
            if abs(mom20) > 0.10:
                score += min(1.5, abs(mom20) * 5)
                reasons.append(f"20-day move {mom20:+.0%}")
            if near_high:
                score += 0.5
                reasons.append("within 3% of 52-week high")
            score += min(1.0, atr_pct * 25)  # aggressive style prefers range
            direction = int(np.sign(mom20 + gap)) if (abs(mom20) > 0.05 or abs(gap) > 0.02) else 0
            out[sym] = SourceScore(score, ", ".join(reasons) or f"ATR {atr_pct:.1%}", direction)
        return out


@dataclass
class ScoutConfig:
    min_items: int = 10
    max_items: int = 20
    min_score: float = 1.0                       # items beyond min_items must clear this
    weights: dict[str, float] = field(default_factory=lambda: {"technical": 1.0, "news": 1.0, "chatter": 0.5})
    always_on: list[str] = field(default_factory=lambda: ["SPY", "QQQ"])  # regime references, not counted


class CompositeScout:
    """Merges source scores into one watchlist of ``min_items``..``max_items`` symbols."""

    def __init__(self, universe: list[str], sources: list[ScoutSource], cfg: ScoutConfig | None = None):
        self.universe, self.sources, self.cfg = universe, sources, cfg or ScoutConfig()

    def scan(self, asof: pd.Timestamp) -> Watchlist:
        totals: dict[str, float] = {}
        reasons: dict[str, list[str]] = {}
        srcs: dict[str, list[str]] = {}
        bias: dict[str, int] = {}
        cands = [s for s in self.universe if s not in self.cfg.always_on]
        for src in self.sources:
            try:
                scores = src.score(asof, cands)
            except NotImplementedError:
                continue
            w = self.cfg.weights.get(src.name, 1.0)
            for sym, sc in scores.items():
                totals[sym] = totals.get(sym, 0.0) + w * sc.score
                reasons.setdefault(sym, []).append(f"{src.name}: {sc.reason}")
                srcs.setdefault(sym, []).append(src.name)
                bias[sym] = bias.get(sym, 0) + sc.direction
        ranked = sorted(totals.items(), key=lambda kv: -kv[1])
        picked = [kv for n, kv in enumerate(ranked) if n < self.cfg.min_items or kv[1] >= self.cfg.min_score]
        picked = picked[: self.cfg.max_items]
        items = [WatchItem(s, round(v, 3), int(np.sign(bias.get(s, 0))), reasons[s], srcs[s]) for s, v in picked]
        return Watchlist(asof=str(asof), items=items, always_on=list(self.cfg.always_on))


def scout_mask(scout: CompositeScout, dates: pd.DatetimeIndex, hold_days: int = 1) -> dict[str, pd.Series]:
    """Replay the scout over history: for each date, which symbols were on that day's watchlist.

    The returned masks plug into ``run_backtest(tradable=...)`` so a strategy only opens
    trades in names the scout picked that morning, which is how it will run live.
    """
    days = pd.DatetimeIndex(sorted(set(dates.normalize())))
    on: dict[str, list[bool]] = {s: [] for s in scout.universe}
    for d in days:
        picks = set(scout.scan(d).symbols) | set(scout.cfg.always_on)
        for s in scout.universe:
            on[s].append(s in picks)
    masks = {}
    for s, vals in on.items():
        m = pd.Series(vals, index=days)
        if hold_days > 1:
            m = m.rolling(hold_days, min_periods=1).max().astype(bool)
        masks[s] = m
    return masks
