"""Walk-forward validation and the stage-3 promotion gate.

For each fold, every parameter set in the strategy's search space is backtested on
the training window, the best one is picked, and it is then run once on the
following unseen test window. Only test-window results count. A gap equal to the
strategy's maximum holding time separates train and test so no trade straddles them.
The stitched out-of-sample returns are scored with the deflated Sharpe ratio, using
the number of parameter sets ever tried for the strategy (the global trial ledger,
tradex.research.trials), so a large search, or many small ones, cannot pass by luck.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from tradex.backtest import metrics
from tradex.backtest.engine import EngineConfig, run_backtest
from tradex.costs.models import CostModel, model_for
from tradex.strategy.spec import StrategySpec, compute_signals
from tradex.timeframes import duration


@dataclass
class Thresholds:
    """Stage-3 gate. Defaults from the thread-1 design doc; tune from simulation."""
    min_trades: int = 100
    min_profit_factor: float = 1.2
    min_dsr: float = 0.95
    max_drawdown: float = -0.50            # aggressive style: tolerate deep but not ruinous drawdowns
    min_positive_folds: float = 0.5        # share of test folds that must make money

    @classmethod
    def from_config(cls, path=None) -> "Thresholds":
        """Load the protected thresholds file (config/gates/thresholds.yaml); defaults if absent."""
        from pathlib import Path

        import yaml
        p = Path(path) if path else Path(__file__).resolve().parents[2] / "config" / "gates" / "thresholds.yaml"
        if not p.exists():
            return cls()
        d = yaml.safe_load(p.read_text()) or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class WalkForwardConfig:
    n_folds: int = 5
    initial_train_frac: float = 0.4
    anchored: bool = True                  # anchored = train on all history so far; else rolling window
    grid_points: int = 4
    max_trials: int = 48
    min_train_trades: int = 10
    seed: int = 0


@dataclass
class FoldResult:
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    best_params: dict
    train_sharpe: float
    test_sharpe: float
    test_return: float
    test_trades: int


@dataclass
class ValidationReport:
    strategy_id: str
    version: int
    status: str
    reasons: list[str]
    n_trials: int                          # distinct parameter sets ever tried (DSR's N)
    oos: dict
    folds: list[FoldResult]
    param_stability: float
    recommended_params: dict
    warnings: list[str] = field(default_factory=list)
    n_trials_this_run: int = 0
    oos_returns: pd.Series | None = field(default=None, repr=False)
    oos_trades: pd.DataFrame | None = field(default=None, repr=False)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("oos_returns", None)
        d.pop("oos_trades", None)
        return d


def _splits(index: pd.DatetimeIndex, wf: WalkForwardConfig, embargo: pd.Timedelta):
    n = len(index)
    first_test = int(n * wf.initial_train_frac)
    edges = np.linspace(first_test, n, wf.n_folds + 1).astype(int)
    for k in range(wf.n_folds):
        a, b = edges[k], edges[k + 1]
        if b - a < 2:
            continue
        test_start = index[a]
        test_end = index[b] if b < n else index[-1] + pd.Timedelta(seconds=1)
        train_end = test_start - embargo
        train_start = index[0] if wf.anchored else index[max(0, a - first_test)]
        yield k, train_start, train_end, test_start, test_end


def walk_forward(
    spec: StrategySpec,
    data: dict[str, pd.DataFrame],
    costs: CostModel | None = None,
    engine_cfg: EngineConfig | None = None,
    wf: WalkForwardConfig | None = None,
    thresholds: Thresholds | None = None,
    tradable: dict[str, pd.Series] | None = None,
    filter_ctx: dict | None = None,
    trials=None,
    data_key: str | None = None,
) -> ValidationReport:
    """``trials`` is a TrialLedger; None uses the global one (``TRADEX_TRIALS_DB`` or data/research/)."""
    from tradex.research.trials import TrialLedger
    trials = trials if trials is not None else TrialLedger()
    run_id = TrialLedger.new_run_id()
    wf = wf or WalkForwardConfig()
    th = thresholds or Thresholds.from_config()
    base_cfg = engine_cfg or EngineConfig()
    costs = costs or model_for(spec.asset_class)
    grid = spec.param_grid(wf.grid_points, wf.max_trials, wf.seed)
    prior_trials = trials.count(spec.id)
    embargo = duration(spec.signal_tf) * (spec.exit.max_bars + 1)
    timeline = pd.DatetimeIndex(sorted(set().union(*[set(b.index) for b in data.values() if len(b)])))

    # Signals depend only on parameters, not on the window, so compute them once per parameter set.
    variants = []
    warnings: list[str] = []
    if spec.version > 1 and prior_trials == 0:
        warnings.append(f"trial ledger holds no earlier trials of {spec.id} although it is at version {spec.version}: "
                        f"N restarts at this run's {len(grid)} parameter sets, so the deflation may be too light")
    for params in grid:
        s = spec.with_params(params)
        sig = {sym: compute_signals(s, bars, filter_ctx) for sym, bars in data.items() if len(bars)}
        for sym, sf in sig.items():
            warnings += [f"{sym}: {w}" for w in sf.warnings]
        variants.append((params, s, sig))

    def run(s, sig, start, end):
        cfg = EngineConfig(**{**base_cfg.__dict__, "start": start, "end": end})
        return run_backtest(s, data, costs, cfg, tradable, filter_ctx, precomputed=sig)

    folds: list[FoldResult] = []
    trial_srs: list[list[float]] = []
    oos_parts, oos_trades = [], []
    chosen = []
    for k, tr_s, tr_e, te_s, te_e in _splits(timeline, wf, embargo):
        if tr_e <= tr_s:
            continue
        scored = []
        for params, s, sig in variants:
            res = run(s, sig, tr_s, tr_e)
            sr = metrics.sharpe(res.daily_returns, annualise=False)
            scored.append((sr, len(res.trades), params, s, sig))
        trial_srs.append([x[0] for x in scored])
        trials.record_many([dict(strategy_id=spec.id, version=spec.version, params=x[2], run_id=run_id, fold=k,
                                 window=(tr_s, tr_e), data_key=data_key, sharpe=float(x[0]), trades=x[1])
                            for x in scored])
        eligible = [x for x in scored if x[1] >= wf.min_train_trades] or scored
        best = max(eligible, key=lambda x: x[0])
        test = run(best[3], best[4], te_s, te_e)
        test.params = best[2]
        oos_parts.append(test.daily_returns)
        if not test.trades.empty:
            oos_trades.append(test.trades.assign(fold=k))
        chosen.append(tuple(sorted(best[2].items())))
        folds.append(FoldResult(
            fold=k, train_start=str(tr_s), train_end=str(tr_e), test_start=str(te_s), test_end=str(te_e),
            best_params=best[2], train_sharpe=best[0] * np.sqrt(metrics.PERIODS_PER_YEAR),
            test_sharpe=metrics.sharpe(test.daily_returns),
            test_return=float(test.equity.iloc[-1] / test.equity.iloc[0] - 1) if len(test.equity) > 1 else 0.0,
            test_trades=int(len(test.trades)),
        ))

    oos_ret = pd.concat(oos_parts).sort_index() if oos_parts else pd.Series(dtype=float)
    oos_ret = oos_ret[~oos_ret.index.duplicated()]
    trades = pd.concat(oos_trades, ignore_index=True) if oos_trades else pd.DataFrame()
    equity = (1 + oos_ret).cumprod() * base_cfg.initial_equity
    if len(equity):
        equity = pd.concat([pd.Series([base_cfg.initial_equity], index=[equity.index[0] - pd.Timedelta(days=1)]), equity])
    oos = metrics.summarize(equity, trades, oos_ret) if len(equity) else metrics.trade_stats(trades)
    n_trials = max(len(grid), trials.count(spec.id))
    var = float(np.mean([np.var(x, ddof=1) for x in trial_srs if len(x) > 1])) if any(len(x) > 1 for x in trial_srs) else 0.0
    var = max(var, trials.sharpe_variance(spec.id))         # a narrow re-run must not switch the deflation off
    oos["dsr"] = metrics.deflated_sharpe(oos_ret, n_trials, var)
    oos["expected_max_sharpe_annual"] = metrics.expected_max_sharpe(n_trials, var) * np.sqrt(metrics.PERIODS_PER_YEAR)
    oos["positive_folds"] = float(np.mean([f.test_return > 0 for f in folds])) if folds else 0.0

    reasons = []
    if oos.get("trades", 0) < th.min_trades:
        reasons.append(f"only {oos.get('trades', 0)} out-of-sample trades (need {th.min_trades})")
    if oos.get("profit_factor", 0) < th.min_profit_factor:
        reasons.append(f"profit factor {oos.get('profit_factor', 0):.2f} after costs (need {th.min_profit_factor})")
    if oos["dsr"] < th.min_dsr:
        reasons.append(f"deflated Sharpe {oos['dsr']:.2f} (need {th.min_dsr})")
    if oos.get("max_drawdown", 0) < th.max_drawdown:
        reasons.append(f"max drawdown {oos['max_drawdown']:.0%} (limit {th.max_drawdown:.0%})")
    if oos["positive_folds"] < th.min_positive_folds:
        reasons.append(f"only {oos['positive_folds']:.0%} of test folds profitable")

    stability = max(chosen.count(c) for c in set(chosen)) / len(chosen) if chosen else 0.0
    recommended = dict(max(set(chosen), key=chosen.count)) if chosen else {}
    return ValidationReport(
        strategy_id=spec.id, version=spec.version, status="validated" if not reasons else "rejected",
        reasons=reasons, n_trials=n_trials, oos=oos, folds=folds, param_stability=stability,
        recommended_params=recommended, warnings=list(dict.fromkeys(warnings)), n_trials_this_run=len(grid),
        oos_returns=oos_ret, oos_trades=trades,
    )
