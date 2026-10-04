"""Test-only builder for minimal valid options specs (signals are injected via ``precomputed``)."""
from __future__ import annotations

from tradex.options.spec import STRUCTURES, OptionStrategySpec


def make_spec(structure="long_call", option=None, exit=None, filters=None, universe=("X",), cap_risk_pct=1.0, **top):
    key = STRUCTURES[structure][0]
    naked = structure == "naked_call"
    opt = {"dte": {"min": 20, "max": 45},
           "strike": {"by": "delta", "target": 0.20 if naked else 0.40, "tolerance": 0.15},
           "liquidity": {"min_open_interest": 100, "max_spread_pct": 0.20, "min_bid": 0.05},
           "exit": {"close_dte": 3}}
    if naked:
        opt["naked"] = {"stop_premium_mult": 2.0}
    for k, v in (option or {}).items():
        opt[k] = {**opt.get(k, {}), **v} if isinstance(v, dict) else v
    d = {"id": f"t-{structure}", "version": 1, "asset_class": "options", "structure": structure,
         "underlying": {"asset_class": "stocks"}, "family": "other", "universe": list(universe),
         "timeframes": {"signal": "D1"}, "features": {}, "entry": {key: "close > 0"},
         "exit": {"stop_atr": 2.0, "target_r": 3.0, "max_bars": 15} | (exit or {}),
         "holding": {"expected_hours": 240},
         "filters": filters if filters is not None else (["no_earnings_3d"] if naked else []),
         "option": opt, "sizing": {"cap_risk_pct": cap_risk_pct}} | top
    return OptionStrategySpec.from_dict(d)
