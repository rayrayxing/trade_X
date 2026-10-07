"""Incremental signals: at each bar close, recompute a strategy's signals on a rolling
window of recent bars instead of the full history.

The window is at least ``mult`` (3) times the strategy's longest lookback, derived from
its features, exit ATR and rule functions. Recursive indicators never fully forget their
seed, so their lookback counts as several periods: 5 for the EMA family (alpha 2/(n+1))
and 10 for Wilder smoothing (RSI, ADX, ATR; alpha 1/n). Three lookbacks then leave a seed
error near 1e-13, which keeps every boolean signal equal to the full-history one (tested
on the seeds). Features whose state runs from the first bar (zigzag pivots) or that the table
does not know use the full history.
"""
from __future__ import annotations

import ast
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from talib import abstract

from tradex.strategy.spec import StrategySpec, compute_signals
from tradex.timeframes import duration

EMA_PERIODS = 5                         # lookback of an EMA-type indicator, in periods
WILDER_PERIODS = 10                     # Wilder smoothing forgets twice as slowly
FULL = math.inf                         # use every bar there is
EMA_TALIB = {"EMA", "DEMA", "TEMA", "T3", "TRIX", "KAMA", "MAMA", "MACD", "MACDEXT", "MACDFIX", "PPO", "APO"}
CUMULATIVE_TALIB = {"AD", "ADOSC", "OBV", "SAR", "SAREXT"}    # state runs from the first bar
_ALIASES = {"period": "timeperiod", "fast": "fastperiod", "slow": "slowperiod", "signal": "signalperiod"}


def _talib_lookback(name: str, params: dict) -> float:
    fn = abstract.Function(name)
    kw = {_ALIASES.get(k, k): v for k, v in params.items() if k not in ("output", "input", "tf")}
    fn.parameters = {k: v for k, v in kw.items() if k in fn.parameters}
    lb = float(fn.lookback)
    allp = dict(fn.parameters)
    if name in CUMULATIVE_TALIB:
        return FULL
    mult = 0
    if name in EMA_TALIB or any(k.endswith("matype") and v for k, v in allp.items()):
        mult = EMA_PERIODS
    elif any("unstable" in f.lower() for f in fn.function_flags or []) or name.startswith("HT_"):
        mult = WILDER_PERIODS           # TA-Lib's own unstable-period list: Wilder smoothing and similar
    if mult:
        periods = [float(v) for k, v in allp.items() if "period" in k and isinstance(v, (int, float))]
        lb = max(lb, mult * max(periods or [lb or 1.0]))
    return lb


def _registered_lookback(fn: str, p: dict) -> float:
    per = lambda k, d: float(p.get(k, d))  # noqa: E731
    if fn in ("stat.highest", "stat.lowest", "stat.rel_volume"):
        return per("period", 20) + 1
    if fn in ("stat.zscore", "stat.efficiency_ratio"):
        return per("period", 20) + 1
    if fn == "stat.gap_pct":
        return 1
    if fn == "stat.atr_pct":
        return WILDER_PERIODS * per("period", 14)
    if fn == "pattern.zone_touch":
        return WILDER_PERIODS * per("atr_period", 14)
    if fn == "pattern.candle_score":
        return 30 + (WILDER_PERIODS * 14 if p.get("at_level") else 0)
    return FULL                         # zigzag pivots, patterns and anything unknown: full history


def feature_lookbacks(spec: StrategySpec) -> dict[str, float]:
    """Lookback of each feature in signal-timeframe bars, including what it references."""
    stf = duration(spec.signal_tf)
    out: dict[str, float] = {}
    for name, f in spec.features.items():
        fn = f["fn"]
        params = {k: v for k, v in f.items() if k not in ("fn", "tf")}
        own = _talib_lookback(fn.split(".", 1)[1].upper(), params) if fn.startswith("talib.") \
            else _registered_lookback(fn, params)
        for rk in ("ref", "input", "at_level"):
            if f.get(rk) in out:
                own += out[f[rk]]
        ratio = duration(f.get("tf", spec.signal_tf)) / stf
        out[name] = own * ratio + math.ceil(ratio) if ratio > 1 else own
    return out


def _rule_lookback(rule: str | None) -> float:
    if not rule:
        return 0.0
    n = 0.0
    for node in ast.walk(ast.parse(rule, mode="eval")):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            nums = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, (int, float))]
            extra = 1.0 if node.func.id in ("shift", "rising", "falling", "cross_above", "cross_below") else 0.0
            n += max(nums or [0.0]) + extra
    return n


def strategy_lookback(spec: StrategySpec) -> float:
    feats = feature_lookbacks(spec)
    atr = WILDER_PERIODS * spec.exit.atr_period
    rules = max(_rule_lookback(r) for r in (spec.entry.get("long"), spec.entry.get("short"),
                                            spec.exit.long_when, spec.exit.short_when))
    return max([atr, *feats.values()]) + rules


def window_bars(spec: StrategySpec, mult: int = 3, min_bars: int = 50) -> int | None:
    """Rolling window length in signal bars, or None for the full history."""
    lb = strategy_lookback(spec)
    return None if math.isinf(lb) else max(min_bars, int(math.ceil(mult * lb)))


@dataclass
class SignalPoint:
    """A strategy's signals on one symbol at one bar close."""
    long_entry: bool
    short_entry: bool
    long_exit: bool
    short_exit: bool
    atr: float
    close: float
    bars: int                           # bars in the window


class SignalCache:
    """Signals per (strategy, symbol, close), computed once from a rolling window."""

    def __init__(self, data, mult: int = 3):
        self.data, self.mult = data, mult
        self._window: dict[str, int | None] = {}
        self._cache: dict[tuple[str, str], tuple[pd.Timestamp, SignalPoint | None]] = {}

    def window(self, spec: StrategySpec) -> int | None:
        if spec.id not in self._window:
            self._window[spec.id] = window_bars(spec, self.mult)
        return self._window[spec.id]

    def at(self, spec: StrategySpec, symbol: str, close_t: pd.Timestamp) -> SignalPoint | None:
        """Signals known at ``close_t``; None when the symbol has no bar closing then."""
        key = (spec.id, symbol)
        hit = self._cache.get(key)
        if hit is not None and hit[0] == close_t:
            return hit[1]
        bars = self.data.bars(symbol, close_t, spec.signal_tf)
        n = self.window(spec)
        if n is not None:
            bars = bars.iloc[-n:]
        pt = None
        research = getattr(self.data, "research", None)       # live research columns; None blocks the strategy
        if research is not None and research.serves(spec) and len(bars):
            bars = research.attach(spec, symbol, bars, close_t)
            if bars is None:
                self._cache[key] = (close_t, None)
                return None
        if len(bars) and bars.index[-1] + duration(spec.signal_tf) == close_t:
            sig = compute_signals(spec, bars)
            pt = SignalPoint(bool(sig.long_entry.iloc[-1]), bool(sig.short_entry.iloc[-1]),
                             bool(sig.long_exit.iloc[-1]), bool(sig.short_exit.iloc[-1]),
                             float(sig.atr.iloc[-1]), float(bars["close"].iloc[-1]), len(bars))
            if not np.isfinite(pt.atr):
                pt.atr = float("nan")
        self._cache[key] = (close_t, pt)
        return pt
