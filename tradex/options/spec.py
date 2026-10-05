"""Options strategy YAML: the stock-strategy format plus an ``option`` block.

An options spec reuses ``tradex.strategy.spec`` for everything about the *underlying*:
timeframes, features, entry rules, filters, exit rules (``stop_atr``, ``target_r``,
``max_bars``, signal exits), holding, sizing, search space and provenance. The same
``compute_signals`` runs on the underlying's bars. What is new:

    asset_class: options
    structure: long_call            # long_call | long_put | naked_call
    underlying: {asset_class: stocks}
    entry: {long: "..."}            # long_call reads entry.long; long_put and naked_call read entry.short
    option:
      dte:        {min: 30, max: 60}                       # calendar days at entry
      strike:     {by: delta, target: 0.40, tolerance: 0.15}   # or {by: moneyness, target: 1.05}
      liquidity:  {min_open_interest: 100, max_spread_pct: 0.20, min_bid: 0.10}
      iv:         {max: 1.20}                              # absolute implied-volatility band (optional)
      exit:       {close_dte: 7, premium_stop_pct: 0.5, premium_target_mult: 1.0}
      naked:      {stop_premium_mult: 2.0, gap_pct: 0.20}  # naked_call only; stop is mandatory

Exit semantics: ``exit.stop_atr`` / ``target_r`` apply to the *underlying* (stop below entry for
a long call, above entry for a naked call and, mirrored, for a long put), measured in ATR as
for stocks; ``max_bars`` and the signal exits work as for stocks. The premium rules in
``option.exit`` and ``option.naked`` are additional. The first rule to fire closes the position.

``OptionStrategySpec.validate()`` returns the problems a CI check lists (as
``python -m tradex.options check <dir>``). A spec can tighten the naked-call limits but not relax
them: the 20% gap floor, a mandatory stop, no earnings hold, the squeeze limits.
"""
from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from tradex.options.contract import Right
from tradex.options.sizing import GAP_PCT_FLOOR
from tradex.strategy.spec import StrategySpec

# structure -> (entry key it reads, right, +1 long / -1 short)
STRUCTURES: dict[str, tuple[str, Right, int]] = {
    "long_call": ("long", Right.CALL, 1),
    "long_put": ("short", Right.PUT, 1),
    "naked_call": ("short", Right.CALL, -1),
}
OPTION_ASSET_CLASS = "options"
NAKED_MAX_SHORT_INTEREST = 20.0      # percent of float; a spec may be stricter, never looser
NAKED_MAX_DAYS_TO_COVER = 5.0
_STOCK_KEYS_TO_DROP = ("option", "structure", "underlying")


@dataclass
class DteRule:
    min: int = 30
    max: int = 60


@dataclass
class StrikeRule:
    by: str = "delta"                  # delta | moneyness
    target: float = 0.40               # absolute delta, or strike / spot
    tolerance: float = 0.15            # accepted distance from target, same units


@dataclass
class LiquidityRule:
    min_open_interest: float = 100.0
    min_volume: float = 0.0
    max_spread_pct: float = 0.20       # (ask - bid) / mid
    min_bid: float = 0.05              # a naked call also needs a credit worth the fees


@dataclass
class IvRule:
    min: float | None = None
    max: float | None = None


@dataclass
class OptionExit:
    close_dte: int = 7                 # close when this many calendar days remain
    premium_stop_pct: float | None = None      # long: close when premium falls this share below entry
    premium_target_mult: float | None = None   # long: close when premium has gained this multiple of entry


@dataclass
class NakedRules:
    stop_premium_mult: float | None = None     # buy-to-close when the option trades at this multiple of the credit
    gap_pct: float = GAP_PCT_FLOOR
    earnings_buffer_days: int = 1
    max_short_interest_pct_float: float = NAKED_MAX_SHORT_INTEREST
    max_days_to_cover: float = NAKED_MAX_DAYS_TO_COVER
    use_underlying_stop: bool = True           # also buy back when the underlying reaches exit.stop_atr


def _build(cls, d: dict | None, label: str, unknown: list[str]):
    d = dict(d or {})
    fields = {f.name for f in dataclasses.fields(cls)}
    for k in d:
        if k not in fields:
            unknown.append(f"{label}.{k}")
    return cls(**{k: v for k, v in d.items() if k in fields})


@dataclass
class OptionRules:
    dte: DteRule = field(default_factory=DteRule)
    strike: StrikeRule = field(default_factory=StrikeRule)
    liquidity: LiquidityRule = field(default_factory=LiquidityRule)
    iv: IvRule = field(default_factory=IvRule)
    exit: OptionExit = field(default_factory=OptionExit)
    naked: NakedRules = field(default_factory=NakedRules)
    unknown_keys: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict | None) -> "OptionRules":
        d = dict(d or {})
        unknown: list[str] = []
        known = {"dte", "strike", "liquidity", "iv", "exit", "naked"}
        unknown += [f"option.{k}" for k in d if k not in known]
        return cls(_build(DteRule, d.get("dte"), "option.dte", unknown),
                   _build(StrikeRule, d.get("strike"), "option.strike", unknown),
                   _build(LiquidityRule, d.get("liquidity"), "option.liquidity", unknown),
                   _build(IvRule, d.get("iv"), "option.iv", unknown),
                   _build(OptionExit, d.get("exit"), "option.exit", unknown),
                   _build(NakedRules, d.get("naked"), "option.naked", unknown), unknown)


@dataclass
class OptionStrategySpec:
    id: str
    version: int
    structure: str
    base: StrategySpec                 # the underlying's spec: features, entry, filters, exit, sizing
    option: OptionRules
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    # --- conveniences ------------------------------------------------------------------
    @property
    def asset_class(self) -> str:
        return OPTION_ASSET_CLASS

    @property
    def entry_key(self) -> str:
        return STRUCTURES[self.structure][0]

    @property
    def right(self) -> Right:
        return STRUCTURES[self.structure][1]

    @property
    def side(self) -> int:
        """+1 for a long option position, -1 for a short."""
        return STRUCTURES[self.structure][2]

    @property
    def is_naked(self) -> bool:
        return self.structure == "naked_call"

    @property
    def universe(self) -> list[str]:
        return self.base.universe

    @property
    def signal_tf(self) -> str:
        return self.base.signal_tf

    @property
    def status(self) -> str:
        return self.base.status

    @property
    def family(self) -> str:
        return self.base.family

    @property
    def cap_risk_pct(self) -> float:
        return self.base.cap_risk_pct

    # --- loading ------------------------------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "OptionStrategySpec":
        d = copy.deepcopy(d)
        base_d = {k: v for k, v in d.items() if k not in _STOCK_KEYS_TO_DROP}
        base_d["asset_class"] = "stocks"      # the underlying is a stock; this is how its signals are computed
        base_d.setdefault("holding", {}).setdefault("expected_hours", 24 * 14)
        return cls(id=d["id"], version=int(d.get("version", 1)), structure=str(d.get("structure", "")),
                   base=StrategySpec.from_dict(base_d), option=OptionRules.from_dict(d.get("option")), raw=d)

    @classmethod
    def load(cls, path: str | Path) -> "OptionStrategySpec":
        return cls.from_dict(yaml.safe_load(Path(path).read_text()))

    def with_params(self, params: dict[str, Any]) -> "OptionStrategySpec":
        """Copy with parameters applied: a bare key sets an ``exit`` rule, a dotted key a nested value
        (``option.strike.target``, ``features.hi.period``)."""
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
        return OptionStrategySpec.from_dict(d)

    def param_grid(self, points: int = 4, max_trials: int = 64, seed: int = 0) -> list[dict[str, Any]]:
        return self.base.param_grid(points, max_trials, seed)

    # --- schema check -------------------------------------------------------------------
    def validate(self) -> list[str]:
        errs: list[str] = []
        if self.raw.get("asset_class") != OPTION_ASSET_CLASS:
            errs.append(f"asset_class must be {OPTION_ASSET_CLASS!r}")
        und = (self.raw.get("underlying") or {}).get("asset_class", "stocks")
        if und != "stocks":
            errs.append("underlying.asset_class must be 'stocks'")
        if self.structure not in STRUCTURES:
            errs.append(f"structure must be one of {sorted(STRUCTURES)}")
        errs += self.base.validate()
        errs += [f"unknown key {k!r}" for k in self.option.unknown_keys]
        if self.structure not in STRUCTURES:
            return errs
        if not self.base.entry.get(self.entry_key):
            errs.append(f"structure {self.structure} reads entry.{self.entry_key}, which is missing")
        other = "short" if self.entry_key == "long" else "long"
        if self.base.entry.get(other):
            errs.append(f"entry.{other} is not used by {self.structure}; remove it")
        errs += self._check_option_rules()
        return errs

    def _check_option_rules(self) -> list[str]:
        o, errs = self.option, []
        if o.dte.min < 1 or o.dte.max < o.dte.min:
            errs.append("option.dte needs 1 <= min <= max")
        if o.exit.close_dte < 0 or o.exit.close_dte >= o.dte.min:
            errs.append("option.exit.close_dte must be below option.dte.min (the position would close the day it opens)")
        if o.strike.by not in ("delta", "moneyness"):
            errs.append("option.strike.by must be 'delta' or 'moneyness'")
        elif o.strike.by == "delta" and not (0.0 < o.strike.target < 1.0):
            errs.append("option.strike.target is an absolute delta and must be in (0, 1)")
        elif o.strike.by == "moneyness" and o.strike.target <= 0:
            errs.append("option.strike.target is strike / spot and must be positive")
        if o.strike.tolerance <= 0:
            errs.append("option.strike.tolerance must be positive")
        if not (0.0 < o.liquidity.max_spread_pct <= 1.0):
            errs.append("option.liquidity.max_spread_pct must be in (0, 1]")
        if o.liquidity.min_bid < 0 or o.liquidity.min_open_interest < 0 or o.liquidity.min_volume < 0:
            errs.append("option.liquidity values must not be negative")
        if o.iv.min is not None and o.iv.max is not None and o.iv.min > o.iv.max:
            errs.append("option.iv.min is above option.iv.max")
        if o.exit.premium_stop_pct is not None and not (0.0 < o.exit.premium_stop_pct <= 1.0):
            errs.append("option.exit.premium_stop_pct must be in (0, 1]")
        if o.exit.premium_target_mult is not None and o.exit.premium_target_mult <= 0:
            errs.append("option.exit.premium_target_mult must be positive")
        if self.is_naked:
            n = o.naked
            if n.stop_premium_mult is None or n.stop_premium_mult <= 1.0:
                errs.append("naked_call needs option.naked.stop_premium_mult above 1 (a buy-to-close stop is mandatory)")
            if n.gap_pct < GAP_PCT_FLOOR - 1e-12:
                errs.append(f"option.naked.gap_pct cannot be below {GAP_PCT_FLOOR:.0%}")
            if n.earnings_buffer_days < 0:
                errs.append("option.naked.earnings_buffer_days must not be negative")
            if n.max_short_interest_pct_float > NAKED_MAX_SHORT_INTEREST:
                errs.append(f"option.naked.max_short_interest_pct_float cannot exceed {NAKED_MAX_SHORT_INTEREST:g}")
            if n.max_days_to_cover > NAKED_MAX_DAYS_TO_COVER:
                errs.append(f"option.naked.max_days_to_cover cannot exceed {NAKED_MAX_DAYS_TO_COVER:g}")
            if "no_earnings_3d" not in self.base.filters:
                errs.append("naked_call must list the no_earnings_3d filter (entries stay out of the pre-earnings window)")
            if o.exit.premium_stop_pct is not None or o.exit.premium_target_mult is not None:
                errs.append("premium_stop_pct and premium_target_mult are for long options; a naked call uses option.naked")
        elif self.raw.get("option", {}).get("naked"):
            errs.append("option.naked applies to naked_call only")
        return errs


def load_dir(path: str | Path) -> list[OptionStrategySpec]:
    return [OptionStrategySpec.load(p) for p in sorted(Path(path).glob("**/*.yaml"))]
