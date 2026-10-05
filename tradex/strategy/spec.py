"""Strategy YAML format: load, schema-check, apply parameters, compute features and signals."""
from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml

from tradex.strategy import expr
from tradex.ta import registry
from tradex.timeframes import TIMEFRAMES, align_higher, resample

STATUSES = ["proposed", "screened", "backtested", "validated", "paper", "live", "retired", "rejected"]
ASSET_CLASSES = ["stocks", "forex"]
BAR_NAMES = {"open", "high", "low", "close", "volume"}
WATCHLIST_TOKEN = "$watchlist"

# --- entry filters -------------------------------------------------------------------
# A filter returns a boolean Series: True where new entries are allowed.
FilterFn = Callable[[pd.DataFrame, dict], pd.Series]
FILTERS: dict[str, FilterFn] = {}


def register_filter(name: str):
    def deco(fn: FilterFn) -> FilterFn:
        FILTERS[name] = fn
        return fn
    return deco


@register_filter("no_high_impact_news_30m")
def _no_news(bars: pd.DataFrame, ctx: dict) -> pd.Series:
    """Blocks entries within 30 minutes of a high-impact event. Needs ctx['events'] (UTC timestamps).

    Without an event calendar it allows everything and the backtest report says so.
    """
    events = ctx.get("events")
    allow = pd.Series(True, index=bars.index)
    if events is None or len(events) == 0:
        ctx.setdefault("warnings", []).append("no_high_impact_news_30m: no event calendar supplied, filter inactive")
        return allow
    ev = pd.DatetimeIndex(events)
    for t in ev:
        allow &= ~((bars.index >= t - pd.Timedelta(minutes=30)) & (bars.index <= t + pd.Timedelta(minutes=30)))
    return allow


@register_filter("no_earnings_3d")
def _no_earnings(bars: pd.DataFrame, ctx: dict) -> pd.Series:
    """Blocks stock entries in the 3 calendar days before an earnings date (ctx['earnings']).

    ``ctx['earnings']`` is a list of dates, or a dict symbol -> dates read with
    ``ctx['symbol']`` (tradex.data.earnings.filter_dates). A symbol present with no dates
    (an ETF) has nothing to block; a symbol missing from the dict is reported as inactive.
    """
    dates = ctx.get("earnings")
    allow = pd.Series(True, index=bars.index)
    if isinstance(dates, dict):
        sym = ctx.get("symbol")
        if sym in dates:
            if len(dates[sym]) == 0:
                return allow
            dates = dates[sym]
        else:
            dates = None
    if dates is None or len(dates) == 0:
        ctx.setdefault("warnings", []).append("no_earnings_3d: no earnings calendar supplied, filter inactive")
        return allow
    for d in pd.DatetimeIndex(dates):
        allow &= ~((bars.index >= d - pd.Timedelta(days=3)) & (bars.index <= d))
    return allow


# --- spec -----------------------------------------------------------------------------

@dataclass
class ExitRules:
    stop_atr: float = 1.5
    target_r: float = 2.0
    max_bars: int = 48
    atr_period: int = 14
    breakeven_r: float | None = None   # move stop to entry once price reaches this multiple of risk
    trail_atr: float | None = None     # trail the stop at this many ATR once in profit
    long_when: str | None = None       # optional signal exit rules
    short_when: str | None = None
    session_close: bool = False        # stocks: flat at the US regular-session close (16:00 New York)


@dataclass
class StrategySpec:
    id: str
    version: int
    asset_class: str
    universe: list[str]
    timeframes: dict[str, str]
    features: dict[str, dict[str, Any]]
    entry: dict[str, str]
    exit: ExitRules
    holding: dict[str, Any] = field(default_factory=dict)
    filters: list[str] = field(default_factory=list)
    sizing: dict[str, Any] = field(default_factory=dict)
    search_space: dict[str, list] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    status: str = "proposed"
    hypothesis: str = ""
    family: str = "other"              # one of tradex.core.records.FAMILIES; correlated strategies share a vote
    stats: dict[str, Any] = field(default_factory=dict)   # measured hit_rate etc. once known
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def signal_tf(self) -> str:
        return self.timeframes["signal"]

    @property
    def uses_watchlist(self) -> bool:
        return WATCHLIST_TOKEN in self.universe

    @property
    def expected_hours(self) -> float:
        return float(self.holding.get("expected_hours", 24))

    @property
    def cap_risk_pct(self) -> float:
        return float(self.sizing.get("cap_risk_pct", 3.0))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StrategySpec":
        d = copy.deepcopy(d)
        ex = d.get("exit", {})
        return cls(
            id=d["id"], version=int(d.get("version", 1)), asset_class=d["asset_class"],
            universe=list(d.get("universe", [])), timeframes=dict(d["timeframes"]),
            features=dict(d.get("features", {})), entry=dict(d.get("entry", {})),
            exit=ExitRules(**{k: v for k, v in ex.items() if k in ExitRules.__dataclass_fields__}),
            holding=dict(d.get("holding", {})), filters=list(d.get("filters", [])),
            sizing=dict(d.get("sizing", {})), search_space=dict(d.get("search_space", {})),
            provenance=dict(d.get("provenance", {})), status=d.get("status", "proposed"),
            hypothesis=d.get("hypothesis", ""), family=d.get("family", "other"),
            stats=dict(d.get("stats", {}) or {}), raw=d,
        )

    @classmethod
    def load(cls, path: str | Path) -> "StrategySpec":
        return cls.from_dict(yaml.safe_load(Path(path).read_text()))

    def with_params(self, params: dict[str, Any]) -> "StrategySpec":
        """Return a copy with search-space parameters applied.

        A bare key (``stop_atr``) sets an exit rule; a dotted key
        (``features.ema_fast.period``) sets a nested value.
        """
        d = copy.deepcopy(self.raw)
        for key, val in params.items():
            if "." not in key:
                d.setdefault("exit", {})[key] = val
                continue
            node = d
            parts = key.split(".")
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = val
        return StrategySpec.from_dict(d)

    def validate(self) -> list[str]:
        """Schema check (stage 2 gate). Returns problems; empty means it passes."""
        errs = []
        if self.asset_class not in ASSET_CLASSES:
            errs.append(f"asset_class must be one of {ASSET_CLASSES}")
        if not self.universe:
            errs.append("universe is empty")
        for k, tf in self.timeframes.items():
            if tf not in TIMEFRAMES:
                errs.append(f"timeframe {k}={tf} unknown")
        if "signal" not in self.timeframes:
            errs.append("timeframes.signal is required")
        if self.status not in STATUSES:
            errs.append(f"status must be one of {STATUSES}")
        from tradex.core.records import FAMILIES
        if self.family not in FAMILIES:
            errs.append(f"family must be one of {list(FAMILIES)}")
        if "expected_hours" not in self.holding:
            errs.append("holding.expected_hours must be declared")
        if self.asset_class == "forex" and "crosses_rollover" not in self.holding:
            errs.append("forex strategies must declare holding.crosses_rollover")
        if not (self.entry.get("long") or self.entry.get("short")):
            errs.append("entry needs a long or short rule")
        if self.exit.stop_atr <= 0 or self.exit.target_r <= 0 or self.exit.max_bars <= 0:
            errs.append("exit.stop_atr, exit.target_r and exit.max_bars must be positive (every trade needs a stop, target and time limit)")
        known = set(BAR_NAMES)
        for name, f in self.features.items():
            fn = f.get("fn")
            if not fn or not registry.is_registered(fn):
                errs.append(f"feature {name}: function {fn!r} is not registered")
            ftf = f.get("tf", self.timeframes.get("signal"))
            if ftf not in TIMEFRAMES:
                errs.append(f"feature {name}: timeframe {ftf!r} unknown")
            for rk in ("ref", "input", "at_level"):
                if rk in f and f[rk] not in known:
                    errs.append(f"feature {name}: {rk}={f[rk]!r} must name an earlier feature or bar field")
            known.add(name)
        rules = [r for r in (self.entry.get("long"), self.entry.get("short"), self.exit.long_when, self.exit.short_when) if r]
        for r in rules:
            try:
                unknown = expr.names_in(r) - known
            except SyntaxError as exc:
                errs.append(f"rule {r!r} does not parse: {exc}")
                continue
            if unknown:
                errs.append(f"rule {r!r} uses undefined names {sorted(unknown)}")
        for flt in self.filters:
            if flt not in FILTERS:
                errs.append(f"filter {flt!r} is not registered")
        for key in self.search_space:
            if "." not in key and key not in ExitRules.__dataclass_fields__:
                errs.append(f"search_space key {key!r} is not an exit rule; use a dotted path for features")
        return errs

    def param_grid(self, points: int = 4, max_trials: int = 64, seed: int = 0) -> list[dict[str, Any]]:
        """Expand search_space: [lo, hi] numeric ranges become ``points`` evenly spaced values."""
        if not self.search_space:
            return [{}]
        axes = {}
        for k, v in self.search_space.items():
            if isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
                lo, hi = v
                vals = np.linspace(lo, hi, points)
                axes[k] = sorted({int(round(x)) for x in vals}) if all(isinstance(x, int) for x in v) else [round(float(x), 4) for x in vals]
            else:
                axes[k] = list(v)
        combos = [dict(zip(axes, c)) for c in itertools.product(*axes.values())]
        if len(combos) > max_trials:
            rng = np.random.default_rng(seed)
            combos = [combos[i] for i in sorted(rng.choice(len(combos), max_trials, replace=False))]
        return combos


def load_dir(path: str | Path) -> list[StrategySpec]:
    return [StrategySpec.load(p) for p in sorted(Path(path).glob("**/*.yaml"))]


# --- features and signals ------------------------------------------------------------

def compute_features(spec: StrategySpec, bars: pd.DataFrame) -> dict[str, pd.Series]:
    """Compute every declared feature on the signal timeframe, aligning higher timeframes causally."""
    stf = spec.signal_tf
    ctx_by_tf: dict[str, registry.FeatureContext] = {stf: registry.FeatureContext(bars)}
    out: dict[str, pd.Series] = {}
    for name, f in spec.features.items():
        ftf = f.get("tf", stf)
        params = {k: v for k, v in f.items() if k not in ("fn", "tf")}
        if ftf not in ctx_by_tf:
            ctx_by_tf[ftf] = registry.FeatureContext(resample(bars, ftf))
        ctx = ctx_by_tf[ftf]
        series = registry.compute(f["fn"], ctx, params)
        ctx.features[name] = series
        if ftf != stf:
            series = align_higher(bars.index, stf, series, ftf)
            ctx_by_tf[stf].features[name] = series
        out[name] = series
    return out


@dataclass
class SignalFrame:
    long_entry: pd.Series
    short_entry: pd.Series
    long_exit: pd.Series
    short_exit: pd.Series
    atr: pd.Series
    features: dict[str, pd.Series]
    warnings: list[str]


def compute_signals(spec: StrategySpec, bars: pd.DataFrame, filter_ctx: dict | None = None,
                    symbol: str | None = None) -> SignalFrame:
    """Boolean entry/exit series known at each bar's CLOSE (act at the next open).

    ``symbol`` lets per-symbol filter data (earnings dates) be picked out of ``filter_ctx``."""
    feats = compute_features(spec, bars)
    env = {k: bars[k] for k in BAR_NAMES} | feats
    idx = bars.index
    false = pd.Series(False, index=idx)
    le = expr.evaluate(spec.entry["long"], env, idx) if spec.entry.get("long") else false
    se = expr.evaluate(spec.entry["short"], env, idx) if spec.entry.get("short") else false
    fctx = dict(filter_ctx or {})
    fctx.setdefault("warnings", [])
    if symbol is not None:
        fctx["symbol"] = symbol
    for flt in spec.filters:
        allow = FILTERS[flt](bars, fctx).reindex(idx, fill_value=True)
        le, se = le & allow, se & allow
    lx = expr.evaluate(spec.exit.long_when, env, idx) if spec.exit.long_when else false
    sx = expr.evaluate(spec.exit.short_when, env, idx) if spec.exit.short_when else false
    atr = registry._atr(registry.FeatureContext(bars), spec.exit.atr_period)
    return SignalFrame(le, se, lx, sx, atr, feats, list(dict.fromkeys(fctx["warnings"])))
