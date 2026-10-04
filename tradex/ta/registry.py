"""Feature registry: every indicator or detector a strategy can reference by name.

A feature function receives a ``FeatureContext`` (bars plus already computed
features) and keyword parameters, and returns a Series aligned to the bars.
Every function must be causal: the value at bar t may only use bars <= t.

Any TA-Lib function is available as ``talib.<NAME>`` without registering it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
import talib
from talib import abstract

FeatureFn = Callable[..., pd.Series]


@dataclass
class FeatureContext:
    bars: pd.DataFrame
    features: dict[str, pd.Series] = field(default_factory=dict)

    def ref(self, name: str) -> pd.Series:
        if name in self.features:
            return self.features[name]
        if name in self.bars.columns:
            return self.bars[name]
        raise KeyError(f"feature {name!r} referenced before it is defined")


@dataclass
class FeatureDef:
    name: str
    fn: FeatureFn
    doc: str
    output: str  # "float" | "signal" (+1/-1/0) | "bool"


_REGISTRY: dict[str, FeatureDef] = {}

# Parameter aliases so YAML can say ``period`` for TA-Lib's ``timeperiod``.
_ALIASES = {"period": "timeperiod", "fast": "fastperiod", "slow": "slowperiod", "signal": "signalperiod"}


def register(name: str, output: str = "float"):
    def deco(fn: FeatureFn) -> FeatureFn:
        _REGISTRY[name] = FeatureDef(name=name, fn=fn, doc=(fn.__doc__ or "").strip(), output=output)
        return fn
    return deco


def is_registered(name: str) -> bool:
    if name.startswith("talib."):
        return name.split(".", 1)[1].upper() in talib.get_functions()
    return name in _REGISTRY


def registered_names() -> list[str]:
    return sorted(_REGISTRY) + [f"talib.{n}" for n in talib.get_functions()]


def describe(name: str) -> FeatureDef | None:
    return _REGISTRY.get(name)


def compute(name: str, ctx: FeatureContext, params: dict[str, Any]) -> pd.Series:
    params = dict(params)
    params.pop("tf", None)
    if name.startswith("talib."):
        return _talib(name.split(".", 1)[1].upper(), ctx, params)
    if name not in _REGISTRY:
        raise KeyError(f"unregistered feature {name!r}")
    out = _REGISTRY[name].fn(ctx, **params)
    return pd.Series(np.asarray(out, dtype=float), index=ctx.bars.index)


def _talib(fname: str, ctx: FeatureContext, params: dict[str, Any]) -> pd.Series:
    fn = abstract.Function(fname)
    output = params.pop("output", None)
    src = params.pop("input", None)  # compute on another feature instead of close
    kwargs = {_ALIASES.get(k, k): v for k, v in params.items()}
    inputs = {k: ctx.bars[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume")}
    if src is not None:
        inputs["close"] = ctx.ref(src).to_numpy(float)
    res = fn(inputs, **kwargs)
    names = fn.output_names
    if isinstance(res, list):
        if output is None:
            output = names[0]
        res = res[names.index(output)]
    return pd.Series(np.asarray(res, dtype=float), index=ctx.bars.index)


def _atr(ctx: FeatureContext, period: int = 14) -> pd.Series:
    b = ctx.bars
    return pd.Series(talib.ATR(b["high"].to_numpy(float), b["low"].to_numpy(float), b["close"].to_numpy(float), period),
                     index=b.index)


# --- generic statistics -------------------------------------------------------------

@register("stat.highest")
def highest(ctx: FeatureContext, ref: str = "high", period: int = 20) -> pd.Series:
    """Highest value of ``ref`` over the previous ``period`` bars, excluding the current bar."""
    return ctx.ref(ref).shift(1).rolling(period).max()


@register("stat.lowest")
def lowest(ctx: FeatureContext, ref: str = "low", period: int = 20) -> pd.Series:
    """Lowest value of ``ref`` over the previous ``period`` bars, excluding the current bar."""
    return ctx.ref(ref).shift(1).rolling(period).min()


@register("stat.zscore")
def zscore(ctx: FeatureContext, ref: str = "close", period: int = 20) -> pd.Series:
    """Rolling z-score of ``ref``."""
    s = ctx.ref(ref)
    return (s - s.rolling(period).mean()) / s.rolling(period).std()


@register("stat.rel_volume")
def rel_volume(ctx: FeatureContext, period: int = 20) -> pd.Series:
    """Volume divided by its average over the previous ``period`` bars (feed-independent)."""
    v = ctx.bars["volume"]
    return v / v.shift(1).rolling(period).mean()


@register("stat.atr_pct")
def atr_pct(ctx: FeatureContext, period: int = 14) -> pd.Series:
    """ATR as a fraction of close."""
    return _atr(ctx, period) / ctx.bars["close"]


@register("stat.efficiency_ratio")
def efficiency_ratio(ctx: FeatureContext, period: int = 20) -> pd.Series:
    """Kaufman efficiency ratio: 1 = straight trend, 0 = pure noise."""
    c = ctx.bars["close"]
    return (c - c.shift(period)).abs() / c.diff().abs().rolling(period).sum()


@register("stat.gap_pct")
def gap_pct(ctx: FeatureContext) -> pd.Series:
    """Open versus previous close."""
    b = ctx.bars
    return b["open"] / b["close"].shift(1) - 1


# --- swing pivots and chart patterns -------------------------------------------------

def zigzag_pivots(bars: pd.DataFrame, k_atr: float = 2.0, atr_period: int = 14) -> pd.DataFrame:
    """ATR zigzag. Returns one row per pivot with the bar it formed on and the bar it was CONFIRMED on.

    A pivot is only knowable at its confirmation bar, when price has reversed k x ATR.
    Consumers must key on ``confirmed_at`` to stay free of look-ahead.
    """
    h, l, c = bars["high"].to_numpy(float), bars["low"].to_numpy(float), bars["close"].to_numpy(float)
    atr = talib.ATR(h, l, c, atr_period)
    pivots = []
    direction = 0  # +1 looking for a high, -1 looking for a low
    ext_i = 0
    for i in range(len(c)):
        a = atr[i]
        if np.isnan(a):
            ext_i = i
            continue
        if direction == 0:
            if h[i] - l[ext_i] > k_atr * a:
                direction, ext_i = 1, i
            elif h[ext_i] - l[i] > k_atr * a:
                direction, ext_i = -1, i
            continue
        if direction == 1:
            if h[i] >= h[ext_i]:
                ext_i = i
            elif h[ext_i] - l[i] >= k_atr * a:
                pivots.append((ext_i, i, h[ext_i], 1))
                direction, ext_i = -1, i
        else:
            if l[i] <= l[ext_i]:
                ext_i = i
            elif h[i] - l[ext_i] >= k_atr * a:
                pivots.append((ext_i, i, l[ext_i], -1))
                direction, ext_i = 1, i
    return pd.DataFrame(pivots, columns=["at", "confirmed_at", "price", "kind"])


def _last_confirmed(bars: pd.DataFrame, piv: pd.DataFrame, kind: int, nth: int = 0) -> pd.Series:
    """Price of the (nth previous) confirmed pivot of ``kind`` as known at each bar."""
    p = piv[piv["kind"] == kind]
    out = np.full(len(bars), np.nan)
    conf, price = p["confirmed_at"].to_numpy(), p["price"].to_numpy()
    j = -1
    for i in range(len(bars)):
        while j + 1 < len(conf) and conf[j + 1] <= i:
            j += 1
        if j - nth >= 0:
            out[i] = price[j - nth]
    return pd.Series(out, index=bars.index)


@register("pattern.pivot_high")
def pivot_high(ctx: FeatureContext, k_atr: float = 2.0, nth: int = 0) -> pd.Series:
    """Last confirmed swing high (nth=1 for the one before)."""
    return _last_confirmed(ctx.bars, zigzag_pivots(ctx.bars, k_atr), 1, nth)


@register("pattern.pivot_low")
def pivot_low(ctx: FeatureContext, k_atr: float = 2.0, nth: int = 0) -> pd.Series:
    """Last confirmed swing low (nth=1 for the one before)."""
    return _last_confirmed(ctx.bars, zigzag_pivots(ctx.bars, k_atr), -1, nth)


@register("pattern.zone_touch", output="bool")
def zone_touch(ctx: FeatureContext, ref: str, tolerance_atr: float = 0.3, atr_period: int = 14) -> pd.Series:
    """True when the bar's range comes within ``tolerance_atr`` x ATR of the ``ref`` level."""
    lvl, atr, b = ctx.ref(ref), _atr(ctx, atr_period), ctx.bars
    tol = tolerance_atr * atr
    return ((b["low"] <= lvl + tol) & (b["high"] >= lvl - tol)).astype(float)


@register("pattern.double_top_bottom", output="signal")
def double_top_bottom(ctx: FeatureContext, k_atr: float = 2.0, tolerance_atr: float = 0.5,
                      max_bars_between: int = 60) -> pd.Series:
    """+1 on the bar that closes above the neckline of a confirmed double bottom, -1 for a double top.

    Double bottom: two confirmed swing lows within ``tolerance_atr`` x ATR of each other,
    separated by a confirmed swing high (the neckline), no more than ``max_bars_between`` apart.
    """
    b = ctx.bars
    piv = zigzag_pivots(b, k_atr)
    atr = _atr(ctx).to_numpy()
    c = b["close"].to_numpy(float)
    out = np.zeros(len(b))
    rows = piv.to_dict("records")
    armed: list[tuple[int, float, int, float]] = []  # (direction, neckline, armed_from, invalidation)
    for n in range(2, len(rows)):
        p1, mid, p2 = rows[n - 2], rows[n - 1], rows[n]
        if p1["kind"] != p2["kind"] or mid["kind"] == p2["kind"]:
            continue
        i_conf = p2["confirmed_at"]
        a = atr[i_conf]
        if np.isnan(a) or p2["at"] - p1["at"] > max_bars_between:
            continue
        if abs(p1["price"] - p2["price"]) <= tolerance_atr * a:
            direction = 1 if p2["kind"] == -1 else -1
            armed.append((direction, mid["price"], i_conf, min(p1["price"], p2["price"]) if direction == 1 else max(p1["price"], p2["price"])))
    for direction, neck, start, inval in armed:
        for i in range(start, min(len(c), start + max_bars_between)):
            if (direction == 1 and c[i] < inval) or (direction == -1 and c[i] > inval):
                break
            if (direction == 1 and c[i] > neck) or (direction == -1 and c[i] < neck):
                out[i] = direction
                break
    return pd.Series(out, index=b.index)


@register("pattern.candle_score", output="signal")
def candle_score(ctx: FeatureContext, patterns: list[str] | None = None, at_level: str | None = None,
                 tolerance_atr: float = 0.5) -> pd.Series:
    """Net bullish/bearish vote of TA-Lib candlestick detectors, optionally only near a level.

    Returns the sum of detector outputs scaled to -1..+1 per detector (so 2.0 = two bullish hits).
    """
    b = ctx.bars
    names = patterns or [f for f in talib.get_function_groups()["Pattern Recognition"]]
    o, h, l, c = (b[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    total = np.zeros(len(b))
    for n in names:
        total += getattr(talib, n.upper())(o, h, l, c) / 100.0
    s = pd.Series(total, index=b.index)
    if at_level:
        s = s.where(zone_touch(ctx, at_level, tolerance_atr) > 0, 0.0)
    return s
