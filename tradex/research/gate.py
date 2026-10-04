"""Phase-1 gate run: every runnable seed and catalog Build entry, on real bars only.

Each runnable strategy goes through walk-forward with full costs (moomoo SG fees and
spread/slippage estimates for US stocks; Oanda bid/ask spreads for FX when that history
exists), the global trial ledger (DSR's N), and the protected stage-3 thresholds in
config/gates/thresholds.yaml. Entries whose data we do not have are reported as
"needs data" and are not run on anything else; entries that are not standalone signals,
or need a model or engine feature not built yet, say so.

    python -m tradex.research.gate            # writes research/results/phase1_gate.{json,md}
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from tradex.backtest.engine import EngineConfig
from tradex.backtest.validation import Thresholds, WalkForwardConfig, walk_forward
from tradex.costs.models import model_for
from tradex.data.opend import DEFAULT_CACHE as OPEND_CACHE, load_cached
from tradex.research import catalog, panels
from tradex.research.trials import TrialLedger
from tradex.strategy.spec import StrategySpec

ROOT = Path(__file__).resolve().parents[2]
SEEDS = ROOT / "strategies" / "seeds"
SPECS = ROOT / "research" / "specs"
RESULTS = ROOT / "research" / "results"
OANDA_CACHE = ROOT / "data" / "cache" / "oanda"

SECTOR = {"NVDA": "XLK", "AMD": "XLK", "AAPL": "XLK", "MSFT": "XLK", "INTC": "XLK", "CSCO": "XLK", "ORCL": "XLK",
          "GOOGL": "XLK", "META": "XLK",   # XLC only exists from 2018; the old GICS home is used throughout
          "AMZN": "XLY", "TSLA": "XLY", "HD": "XLY", "LOW": "XLY", "KO": "XLP", "PEP": "XLP", "WMT": "XLP",
          "COST": "XLP", "XOM": "XLE", "CVX": "XLE", "JPM": "XLF", "BAC": "XLF", "GS": "XLF", "MS": "XLF",
          "V": "XLF", "MA": "XLF", "MRK": "XLV", "PFE": "XLV", "UNH": "XLV", "JNJ": "XLV"}

# Ladder order of the stage-3 gate; the first failing rung is reported.
RUNGS = ("trades", "profit_factor", "deflated_sharpe", "max_drawdown", "folds_profitable")


@dataclass
class Plan:
    catalog_id: str
    spec: Path | None = None
    builder: str | None = None       # name of a data builder below
    verdict: str | None = None       # set when not run: "needs data" | "not built" | "not a standalone signal"
    why: str = ""


def _seed(name):
    return SEEDS / f"{name}.yaml"


def _spec(name):
    return SPECS / f"{name}.yaml"


NO_OANDA = "Oanda practice bid/ask history not downloaded (no oanda_token in the Keychain yet)"
PLANS = [
    Plan("donchian-channel-breakout", _seed("fx-donchian-breakout-h4"), "fx"),
    Plan("ema-pullback-swing-seed", _seed("stk-ema-pullback-swing"), "us_d1"),
    Plan("macd-trend-with-candle-confirmation-seed", _seed("stk-macd-trend-candle"), "us_d1"),
    Plan("trend-pullback-with-engulfing-candle-seed", _seed("fx-trend-pullback-engulfing"), "fx"),
    Plan("rsi-2-pullback-in-uptrend-seed", _seed("stk-rsi2-meanrev"), "us_d1"),
    Plan("rsi-range-reversion-seed", _seed("fx-rsi-range-reversion"), "fx"),
    Plan("breakout-with-volume-seed", _seed("stk-breakout-volume"), "us_d1"),
    Plan("double-bottom-breakout-seed", _seed("stk-double-bottom-breakout"), "us_d1"),
    Plan("52-week-high-momentum", _spec("stk-52w-high-momentum"), "high_ratio"),
    Plan("relative-strength-vs-sector", _spec("stk-sector-relative-strength"), "sector_rs"),
    Plan("pairs-trading-distance-and-cointegration", _spec("stk-pairs-zscore"), "pairs"),
    Plan("gap-and-go-after-news-gap", _spec("stk-gap-and-go-h1"), "sessions"),
    Plan("overnight-vs-intraday-return-split", _spec("etf-overnight-hold"), "sessions"),
    Plan("joint-time-series-and-cross-sectional-strategy", _spec("etf-ts-xs-momentum"), "momentum_xs"),
    Plan("deep-momentum-network-lstm-trained-on-sharpe", verdict="not built", why="needs a trained LSTM model; no model training in phase 1"),
    Plan("momentum-transformer-with-changepoints", verdict="not built", why="needs a trained transformer model"),
    Plan("slow-momentum-with-fast-reversion-changepoint-detection", verdict="not built", why="needs the Gaussian-process changepoint model"),
    Plan("dynamic-momentum-learning-adaptive-lookback", verdict="not built", why="needs the adaptive-lookback learner"),
    Plan("learning-to-rank-cross-sectional-momentum", verdict="not built", why="needs a trained ranking model and a point-in-time universe"),
    Plan("deep-learning-statistical-arbitrage-residual-portfolios", verdict="not built", why="needs a trained model and factor returns"),
    Plan("deep-chart-pattern-recognition-head-and-shoulders-triangles", verdict="not built", why="needs a trained pattern model"),
    Plan("large-tick-trend-filter-trade-trend-only-where-tick-size-is-large", verdict="not a standalone signal", why="a filter for trend strategies"),
    Plan("uncertainty-gated-stock-ranker-skip-when-model-unsure", verdict="not a standalone signal", why="a gate on a ranker that does not exist yet"),
    Plan("hidden-markov-regime-allocation", verdict="not a standalone signal", why="regime input for the ensemble"),
    Plan("realised-covariance-regime-detection", verdict="not a standalone signal", why="regime detector, an input to allocation"),
    Plan("regime-switching-volatility-forecast-for-sizing", verdict="not a standalone signal", why="sizing input"),
    Plan("long-calls-or-puts-on-high-conviction-plans", verdict="not a standalone signal", why="execution style; needs live option chains"),
    Plan("intraday-momentum-first-half-hour-predicts-last", verdict="not built",
         why="trades the last half hour; the engine fills exits at the next open, so it cannot exit at the 16:00 close yet"),
    Plan("earnings-day-jump-continuation", verdict="needs data", why="historical earnings dates"),
    Plan("pre-earnings-run-up-and-iv-crush-options", verdict="needs data", why="historical earnings dates and option implied volatility"),
    Plan("pre-fomc-announcement-drift", verdict="needs data", why="historical FOMC dates (data/calendar covers 2026-2027 only)"),
    Plan("llm-news-sentiment-long-short", verdict="needs data", why="news headline history"),
    Plan("chatgpt-headline-scoring", verdict="needs data", why="news headline history"),
    Plan("short-high-borrow-fee-high-short-interest-names", verdict="needs data", why="borrow-fee and short-interest history (not downloaded in phase 1)"),
    Plan("short-seller-flow-signal", verdict="needs data", why="short-volume history (not downloaded in phase 1)"),
]


# --- data builders -------------------------------------------------------------------------

def _us(symbols, tf, cache=OPEND_CACHE) -> dict[str, pd.DataFrame]:
    out = {}
    for s in symbols:
        b = load_cached(s, tf, cache)
        if b is not None and len(b):
            out[s] = b
    return out


def build(builder: str, spec: StrategySpec, cache=OPEND_CACHE) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Real bars for the spec's universe plus the research columns it reads. Returns (data, missing symbols)."""
    syms = [s for s in spec.universe if not s.startswith("$")]
    if builder == "fx":
        return {}, syms
    data = _us(syms, spec.signal_tf, cache)
    missing = [s for s in syms if s not in data]
    if builder == "high_ratio":
        hr = pd.DataFrame({s: panels.high_ratio(b) for s, b in data.items()})
        xs = panels.xs_percentile(hr)
        data = {s: panels.with_columns(b, {"hi_ratio": hr[s], "hi_ratio_xs": xs[s]}) for s, b in data.items()}
    elif builder == "momentum_xs":
        m = pd.DataFrame({s: panels.momentum(b) for s, b in data.items()})
        xs = panels.xs_percentile(m)
        data = {s: panels.with_columns(b, {"mom": m[s], "mom_xs": xs[s]}) for s, b in data.items()}
    elif builder == "sector_rs":
        etfs = _us(sorted(set(SECTOR.values())), "D1", cache)
        data = {s: panels.with_columns(b, panels.relative_strength(b, etfs[SECTOR[s]]))
                for s, b in data.items() if SECTOR.get(s) in etfs}
    elif builder == "pairs":
        from tradex.research.universe import PAIRS
        out = {}
        for a, b in PAIRS:
            if a in data and b in data:
                z = panels.pair_zscore(data[a], data[b])
                out[a] = panels.with_columns(data[a], {"pair_z": z})
                out[b] = panels.with_columns(data[b], {"pair_z": -z.reindex(data[b].index)})
        data = out
    elif builder == "sessions":
        data = {s: panels.with_columns(b, panels.session_columns(b)) for s, b in data.items()}
    return data, missing


# --- gate --------------------------------------------------------------------------------

def gate_checks(oos: dict, th: Thresholds) -> list[dict]:
    vals = {"trades": oos.get("trades", 0), "profit_factor": oos.get("profit_factor", 0.0),
            "deflated_sharpe": oos.get("dsr", 0.0), "max_drawdown": oos.get("max_drawdown", 0.0),
            "folds_profitable": oos.get("positive_folds", 0.0)}
    need = {"trades": th.min_trades, "profit_factor": th.min_profit_factor, "deflated_sharpe": th.min_dsr,
            "max_drawdown": th.max_drawdown, "folds_profitable": th.min_positive_folds}
    return [{"rung": r, "value": vals[r], "need": need[r], "ok": bool(vals[r] >= need[r])} for r in RUNGS]


def _f(x, nd=3):
    if x is None:
        return None
    x = float(x)
    return None if np.isnan(x) else (x if np.isinf(x) else round(x, nd))


def run_plan(plan: Plan, entry: catalog.CatalogEntry, ledger: TrialLedger, wf: WalkForwardConfig,
             th: Thresholds, cfg: EngineConfig, cache=OPEND_CACHE) -> dict:
    row = {"catalog_id": plan.catalog_id, "catalog_name": entry.name, "catalog_status": entry.status}
    if plan.spec is None:
        return row | {"result": plan.verdict, "why": plan.why}
    spec = StrategySpec.load(plan.spec)
    row |= {"strategy_id": spec.id, "spec": str(plan.spec.relative_to(ROOT)), "timeframe": spec.signal_tf,
            "market": "US stocks/ETFs (moomoo)" if spec.asset_class == "stocks" else "FX (Oanda)"}
    if plan.builder == "fx" and not any((OANDA_CACHE / f"{s}_{spec.signal_tf}.csv").exists() for s in spec.universe):
        return row | {"result": "needs data", "why": NO_OANDA}
    data, missing = build(plan.builder, spec, cache)
    if not data:
        return row | {"result": "needs data", "why": f"no cached bars for {missing}"}
    t0 = time.time()
    rep = walk_forward(spec, data, costs=model_for(spec.asset_class), engine_cfg=cfg, wf=wf, thresholds=th,
                       trials=ledger, data_key=f"opend:{spec.signal_tf}:{','.join(sorted(data))}")
    checks = gate_checks(rep.oos, th)
    failing = [c["rung"] for c in checks if not c["ok"]]
    oos = rep.oos
    if not failing:
        row["robustness"] = robustness(plan, spec, ledger, wf, th, cfg, cache)
    return row | {
        "result": "pass" if not failing else "fail", "failing_rung": failing[0] if failing else None,
        "failing": failing, "reasons": rep.reasons,
        "symbols": sorted(data), "missing_symbols": missing,
        "bars_used": int(sum(len(b) for b in data.values())),
        "data_from": str(min(b.index[0] for b in data.values()).date()),
        "data_to": str(max(b.index[-1] for b in data.values()).date()),
        "oos_trades": int(oos.get("trades", 0)), "profit_factor": _f(oos.get("profit_factor")),
        "sharpe": _f(oos.get("sharpe")), "deflated_sharpe": _f(oos.get("dsr")), "psr": _f(oos.get("psr")),
        "n_trials": rep.n_trials, "n_trials_this_run": rep.n_trials_this_run,
        "max_drawdown": _f(oos.get("max_drawdown")), "total_return": _f(oos.get("total_return")),
        "win_rate": _f(oos.get("win_rate")), "costs_share_of_gross": _f(oos.get("costs_share_of_gross")),
        "folds_profitable": f"{sum(f.test_return > 0 for f in rep.folds)}/{len(rep.folds)}",
        "folds": [{"test": f"{f.test_start[:10]}..{f.test_end[:10]}", "params": f.best_params,
                   "test_return": _f(f.test_return), "trades": f.test_trades} for f in rep.folds],
        "checks": checks, "warnings": rep.warnings[:10], "seconds": round(time.time() - t0, 1),
    }


def _brief(rep) -> dict:
    o = rep.oos
    return {"oos_trades": int(o.get("trades", 0)), "profit_factor": _f(o.get("profit_factor")),
            "sharpe": _f(o.get("sharpe")), "deflated_sharpe": _f(o.get("dsr")), "max_drawdown": _f(o.get("max_drawdown")),
            "folds_profitable": f"{sum(f.test_return > 0 for f in rep.folds)}/{len(rep.folds)}",
            "gate": "pass" if not rep.reasons else "fail", "reasons": rep.reasons}


def robustness(plan: Plan, spec: StrategySpec, ledger, wf, th, cfg, cache=OPEND_CACHE) -> dict:
    """Extra checks for a pass, not part of the stage-3 gate: double spread and slippage, and the
    same rules on large caps outside the spec's hand-picked universe (a hindsight-selection check)."""
    from tradex.research.universe import LARGE_CAPS
    out = {}
    data, _ = build(plan.builder, spec, cache)
    rep = walk_forward(spec, data, costs=model_for(spec.asset_class, stress=2.0), engine_cfg=cfg, wf=wf,
                       thresholds=th, trials=ledger)
    out["costs_x2"] = _brief(rep)
    if spec.asset_class == "stocks" and spec.signal_tf == "D1" and plan.builder == "us_d1":
        others = [s for s in LARGE_CAPS if s not in spec.universe]
        alt = StrategySpec.from_dict({**spec.raw, "universe": others})
        data, _ = build(plan.builder, alt, cache)
        if data:
            rep = walk_forward(alt, data, costs=model_for(spec.asset_class), engine_cfg=cfg, wf=wf, thresholds=th,
                               trials=ledger)
            out["other_large_caps"] = _brief(rep) | {"symbols": sorted(data)}
    return out


def render_md(rows: list[dict], meta: dict) -> str:
    ran = [r for r in rows if r["result"] in ("pass", "fail")]
    passed = [r for r in ran if r["result"] == "pass"]
    lines = [
        "# Phase-1 gate run", "",
        f"Run {meta['run_at']} on real bars only (moomoo OpenD, qfq-adjusted). Stage-3 thresholds from "
        f"`config/gates/thresholds.yaml`: {meta['thresholds']}.",
        f"Walk-forward: {meta['walk_forward']}. Engine: {meta['engine']}. "
        "Costs: moomoo SG fees and default spread/slippage for US stocks; FX would use Oanda bid/ask.",
        "", f"**{len(passed)} of {len(ran)} strategies run pass the gate.** "
        f"{sum(r['result'] == 'needs data' for r in rows)} need data, "
        f"{sum(r['result'] in ('not built', 'not a standalone signal') for r in rows)} are not built or not standalone.", "",
    ] + [f"Note on `{r['strategy_id']}`: it passes, but fails the robustness check(s) "
         f"{', '.join(k for k, v in r.get('robustness', {}).items() if v['gate'] != 'pass')}; treat it as unproven."
         for r in passed if any(v["gate"] != "pass" for v in r.get("robustness", {}).values())] + [
        "",
        "## Strategies run", "",
        "| Strategy | Catalog entry | Market | Bars used | OOS trades | PF | Sharpe | DSR (N) | Max DD | Folds + | Result | Failing rung |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in ran:
        lines.append(f"| `{r['strategy_id']}` | {r['catalog_name']} | {r['market']} {r['timeframe']}, {len(r['symbols'])} symbols, "
                     f"{r['data_from']}..{r['data_to']} | {r['bars_used']:,} | {r['oos_trades']} | {r['profit_factor']} | "
                     f"{r['sharpe']} | {r['deflated_sharpe']} ({r['n_trials']}) | {r['max_drawdown']} | {r['folds_profitable']} | "
                     f"**{r['result']}** | {r['failing_rung'] or '-'} |")
    lines += ["", "Failing reasons:", ""]
    for r in ran:
        lines.append(f"- `{r['strategy_id']}`: " + ("; ".join(r["reasons"]) or "none"))
    robust = [r for r in ran if r.get("robustness")]
    if robust:
        lines += ["", "## Robustness of the passes (not part of the gate)", ""]
        for r in robust:
            for k, v in r["robustness"].items():
                label = {"costs_x2": "spread and slippage doubled",
                         "other_large_caps": f"same rules on {len(v.get('symbols', []))} other large caps"}[k]
                lines.append(f"- `{r['strategy_id']}`, {label}: {v['oos_trades']} trades, PF {v['profit_factor']}, "
                             f"Sharpe {v['sharpe']}, DSR {v['deflated_sharpe']}, max DD {v['max_drawdown']}, "
                             f"folds {v['folds_profitable']}: **{v['gate']}**" +
                             (f" ({'; '.join(v['reasons'])})" if v["reasons"] else ""))
    lines += ["", "## Not run", "", "| Catalog entry | Status in catalog | Result | Why |", "|---|---|---|---|"]
    for r in rows:
        if r["result"] not in ("pass", "fail"):
            lines.append(f"| {r['catalog_name']} | {r['catalog_status']} | {r['result']} | {r['why']} |")
    lines += ["", "## Caveats", ""] + [f"- {c}" for c in meta["caveats"]]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    only = set(argv or sys.argv[1:])
    entries = {e.id: e for e in catalog.load()}
    unknown = [p.catalog_id for p in PLANS if p.catalog_id not in entries]
    if unknown:
        raise SystemExit(f"plans name catalog entries that do not exist: {unknown}")
    th = Thresholds.from_config()
    wf = WalkForwardConfig()
    cfg = EngineConfig()
    ledger = TrialLedger()
    rows = []
    for plan in PLANS:
        if only and plan.catalog_id not in only:
            continue
        row = run_plan(plan, entries[plan.catalog_id], ledger, wf, th, cfg)
        print(f"{row.get('strategy_id', plan.catalog_id)}: {row['result']} "
              f"{row.get('failing_rung') or row.get('why', '')}", flush=True)
        rows.append(row)
    meta = {
        "run_at": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC"),
        "thresholds": {k: getattr(th, k) for k in th.__dataclass_fields__},
        "walk_forward": {k: getattr(wf, k) for k in wf.__dataclass_fields__},
        "engine": {"initial_equity": cfg.initial_equity, "risk_pct": cfg.risk_pct, "max_leverage": cfg.max_leverage},
        "caveats": [
            "US large caps are today's names, so cross-sectional and pairs results carry survivorship bias.",
            "The stock seeds' universes were picked in 2026 and lean to that period's biggest winners (NVDA, AMD,"
            " TSLA, META, AMZN). Their trial count does not include that choice, so a pass that fails on other"
            " large caps (see robustness) is most likely hindsight selection, not an edge.",
            "No historical earnings calendar: the seeds' no_earnings_3d filter was inactive.",
            "Stock spreads are the cost model's defaults (1 bp half-spread + 2 bp slippage), not measured quotes.",
            "Fixed moomoo fees (US$0.99 + 9% GST per order) weigh heavily at the US$10,000 test equity.",
            "OpenD history starts 2006-09 for daily bars and 2018-09 for 60-minute bars.",
            "DSR's N is every parameter set ever recorded for the strategy in the trial ledger.",
        ],
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    if not only:
        (RESULTS / "phase1_gate.json").write_text(json.dumps({"meta": meta, "rows": rows}, indent=1, default=str))
        (RESULTS / "phase1_gate.md").write_text(render_md(rows, meta))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
