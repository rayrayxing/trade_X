"""The measurements: screen, walk-forward and holdout, on the real engine, costs and gate code.

``EngineEvaluator`` adds nothing of its own to the gate. Walk-forward is ``tradex.backtest.validation.walk_forward``
with the protected thresholds and the global trial ledger; the rungs are ``tradex.research.gate.gate_checks``.
The screen and the holdout backtest use the same engine and cost model.
"""
from __future__ import annotations

import math
from typing import Any, Callable

import pandas as pd

from tradex.backtest import metrics
from tradex.backtest.engine import EngineConfig, run_backtest
from tradex.backtest.validation import Thresholds, WalkForwardConfig, walk_forward


def num(x: Any, nd: int = 4) -> float | None:
    """JSON-safe float: None for missing or NaN, a large finite number for infinity."""
    if x is None:
        return None
    x = float(x)
    if math.isnan(x):
        return None
    if math.isinf(x):
        return 1e9 if x > 0 else -1e9
    return round(x, nd)


def current_params(spec) -> dict[str, Any]:
    """The spec's own value for every search-space key: the parameter set a run without a grid evaluates."""
    out = {}
    for key in spec.search_space:
        if "." not in key:
            out[key] = getattr(spec.exit, key)
        else:
            node: Any = spec.raw
            for part in key.split("."):
                node = node.get(part) if isinstance(node, dict) else None
            out[key] = node
    return out


def brief(res_summary: dict) -> dict[str, Any]:
    keys = ("trades", "profit_factor", "sharpe", "max_drawdown", "total_return", "win_rate", "costs_share_of_gross")
    return {k: (int(res_summary[k]) if k == "trades" else num(res_summary.get(k))) for k in keys}


class EngineEvaluator:
    def __init__(self, thresholds: Thresholds | None = None, wf: WalkForwardConfig | None = None,
                 engine_cfg: EngineConfig | None = None, costs_for: Callable[[Any], Any] | None = None):
        self.thresholds = thresholds or Thresholds.from_config()
        self.wf = wf or WalkForwardConfig()
        self.engine_cfg = engine_cfg or EngineConfig()
        self._costs_for = costs_for

    def costs(self, spec):
        if self._costs_for:
            return self._costs_for(spec)
        from tradex.research.gate import _costs
        return _costs(spec)

    def screen(self, spec, frames: dict[str, pd.DataFrame], trials, run_id: str) -> dict[str, Any]:
        res = run_backtest(spec, frames, self.costs(spec), self.engine_cfg)
        out = brief(metrics.summarize(res.equity, res.trades, res.daily_returns))
        params = current_params(spec)
        # the screen evaluated this parameter set on real bars, so the deflated Sharpe's N counts it
        trials.record(spec.id, spec.version, params, run_id=run_id, source="loop_screen",
                      sharpe=metrics.sharpe(res.daily_returns, annualise=False), n_obs=int(len(res.daily_returns)),
                      trades=out["trades"])
        return out | {"params": params, "warnings": res.warnings[:5]}

    def walk_forward(self, spec, frames: dict[str, pd.DataFrame], trials, data_key: str):
        return walk_forward(spec, frames, costs=self.costs(spec), engine_cfg=self.engine_cfg, wf=self.wf,
                            thresholds=self.thresholds, trials=trials, data_key=data_key)

    def holdout(self, spec, params: dict[str, Any], frames: dict[str, pd.DataFrame], start: pd.Timestamp
                ) -> dict[str, Any]:
        cfg = EngineConfig(**{**self.engine_cfg.__dict__, "start": start, "end": None})
        res = run_backtest(spec, frames, self.costs(spec), cfg, params=params or None)
        return brief(metrics.summarize(res.equity, res.trades, res.daily_returns)) | {
            "params": params, "days": int(len(res.daily_returns)), "warnings": res.warnings[:5]}
