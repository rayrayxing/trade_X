"""Phase-1 gate run: every runnable seed and catalog Build entry, on real bars only.

Each runnable strategy goes through walk-forward with full costs (moomoo SG fees and
spread/slippage estimates for US stocks; spreads measured from Oanda bid/ask candles for
FX), the global trial ledger (DSR's N), and the protected stage-3 thresholds in
config/gates/thresholds.yaml. Entries whose data we do not have are reported as
"needs data" and are not run on anything else; entries that are not standalone signals,
or need a model or engine feature not built yet, say so.

US stock and ETF plans run on a point-in-time liquidity screen (tradex.research.screen)
over the whole candidate pool, not on the symbols their spec names, and with the
earnings calendar (tradex.data.earnings) feeding the ``no_earnings_3d`` filter and the
position reviewer's pre-earnings exit.

The proposed strategies (strategies/proposed/) read their research columns from
tradex.research.builders, which takes calendars and rate histories through
tradex.research.sources and reports "needs data" when one is absent.

    python -m tradex.research.gate            # writes research/results/phase1_gate.{json,md}
    python -m tradex.research.gate stk-pead-ear   # one strategy by strategy id or catalog id; prints, writes research/results/single/<id>.json
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
from tradex.data import earnings as earn
from tradex.data.opend import DEFAULT_CACHE as OPEND_CACHE, load_cached
from tradex.research import builders, catalog, panels, screen as scr
from tradex.research.sources import DataUnavailable
from tradex.research.trials import TrialLedger
from tradex.research.universe import ETFS, PAIRS, STOCKS
from tradex.strategy.spec import StrategySpec
from tradex.timeframes import duration

ROOT = Path(__file__).resolve().parents[2]
SEEDS = ROOT / "strategies" / "seeds"
SPECS = ROOT / "research" / "specs"
PROPOSED = ROOT / "strategies" / "proposed"
RESULTS = ROOT / "research" / "results"
OANDA_CACHE = ROOT / "data" / "cache" / "oanda"
EARNINGS_CACHE = earn.DEFAULT_CACHE
PREVIOUS = RESULTS / "phase1_gate_run2.json"     # run 2 (screened universes, earnings, FX bid/ask), for the comparison

SECTOR = {"NVDA": "XLK", "AMD": "XLK", "AAPL": "XLK", "MSFT": "XLK", "INTC": "XLK", "CSCO": "XLK", "ORCL": "XLK",
          "GOOGL": "XLK", "META": "XLK",   # XLC only exists from 2018; the old GICS home is used throughout
          "AMZN": "XLY", "TSLA": "XLY", "HD": "XLY", "LOW": "XLY", "KO": "XLP", "PEP": "XLP", "WMT": "XLP",
          "COST": "XLP", "XOM": "XLE", "CVX": "XLE", "JPM": "XLF", "BAC": "XLF", "GS": "XLF", "MS": "XLF",
          "V": "XLF", "MA": "XLF", "MRK": "XLV", "PFE": "XLV", "UNH": "XLV", "JNJ": "XLV"}
# The rest of the S&P 100 pool, in the nine original SPDRs (pre-2018 GICS homes: media in
# XLY, telecom in XLK, REITs in XLF; later reclassifications are ignored).
SECTOR |= {s: "XLK" for s in ("ACN", "ADBE", "AVGO", "CRM", "IBM", "INTU", "NOW", "QCOM", "TXN", "PLTR", "PYPL",
                              "T", "VZ", "TMUS")}
SECTOR |= {s: "XLY" for s in ("MCD", "NKE", "SBUX", "TGT", "BKNG", "GM", "F", "NFLX", "CMCSA", "DIS", "CHTR")}
SECTOR |= {s: "XLP" for s in ("PG", "CL", "MDLZ", "MO", "PM")}
SECTOR |= {"COP": "XLE", "LIN": "XLB", "DUK": "XLU", "NEE": "XLU", "SO": "XLU"}
SECTOR |= {s: "XLF" for s in ("AIG", "AXP", "BLK", "BNY", "BRK.B", "C", "COF", "MET", "SCHW", "USB", "WFC", "AMT", "SPG")}
SECTOR |= {s: "XLV" for s in ("ABBV", "ABT", "AMGN", "BMY", "CVS", "DHR", "GILD", "ISRG", "LLY", "MDT", "TMO")}
SECTOR |= {s: "XLI" for s in ("BA", "CAT", "DE", "EMR", "FDX", "GD", "GE", "HON", "LMT", "MMM", "RTX", "UNP", "UPS", "UBER")}

# Screen size per pool: one fixed choice, made before any screened run (not tuned).
SCREEN_N = {"stocks": 30, "etfs": 10, "all": 30}

# Ladder order of the stage-3 gate; the first failing rung is reported.
RUNGS = ("trades", "profit_factor", "deflated_sharpe", "max_drawdown", "folds_profitable")


@dataclass
class Plan:
    catalog_id: str
    spec: Path | None = None
    builder: str | None = None       # name of a data builder below
    verdict: str | None = None       # set when not run: "needs data" | "not built" | "not a standalone signal"
    why: str = ""
    pool: str | None = None          # liquidity-screened candidate pool: "stocks" | "etfs" | "all"


def _seed(name):
    return SEEDS / f"{name}.yaml"


def _spec(name):
    return SPECS / f"{name}.yaml"


def _proposed(name):
    return PROPOSED / f"{name}.yaml"


NO_OANDA = "Oanda practice bid/ask history not downloaded (python -m tradex.research.universe oanda)"
PLANS = [
    Plan("donchian-channel-breakout", _seed("fx-donchian-breakout-h4"), "fx"),
    Plan("ema-pullback-swing-seed", _seed("stk-ema-pullback-swing"), "us_d1", pool="all"),
    Plan("macd-trend-with-candle-confirmation-seed", _seed("stk-macd-trend-candle"), "us_d1", pool="all"),
    Plan("trend-pullback-with-engulfing-candle-seed", _seed("fx-trend-pullback-engulfing"), "fx"),
    Plan("rsi-2-pullback-in-uptrend-seed", _seed("stk-rsi2-meanrev"), "us_d1", pool="etfs"),
    Plan("rsi-range-reversion-seed", _seed("fx-rsi-range-reversion"), "fx"),
    Plan("breakout-with-volume-seed", _seed("stk-breakout-volume"), "us_d1", pool="stocks"),
    Plan("double-bottom-breakout-seed", _seed("stk-double-bottom-breakout"), "us_d1", pool="stocks"),
    Plan("52-week-high-momentum", _spec("stk-52w-high-momentum"), "high_ratio", pool="stocks"),
    Plan("relative-strength-vs-sector", _spec("stk-sector-relative-strength"), "sector_rs", pool="stocks"),
    Plan("pairs-trading-distance-and-cointegration", _spec("stk-pairs-zscore"), "pairs", pool="stocks"),
    Plan("gap-and-go-after-news-gap", _spec("stk-gap-and-go-h1"), "sessions", pool="stocks"),
    Plan("overnight-vs-intraday-return-split", _spec("etf-overnight-hold"), "sessions", pool="etfs"),
    Plan("joint-time-series-and-cross-sectional-strategy", _spec("etf-ts-xs-momentum"), "momentum_xs", pool="etfs"),
    Plan("joint-time-series-and-cross-sectional-strategy", _spec("fx-ts-xs-momentum"), "fx_momentum_xs"),
    Plan("intraday-momentum-first-half-hour-predicts-last", _spec("etf-intraday-momentum"), "sessions", pool="etfs"),
    Plan("intraday-momentum-first-half-hour-predicts-last", _spec("etf-intraday-momentum-m30"), "sessions", pool="etfs"),
    Plan("earnings-day-jump-continuation", _spec("stk-earnings-jump"), "earn_jump", pool="stocks"),
    # proposed strategies built on the injected-calendar / rate / regime features (research/proposals.md).
    # Stock ones run on the liquidity screen like the specs above; the ETF ones keep their spec universe
    # (the four index ETFs and the nine original sector SPDRs: the whole set, not a pick).
    Plan("earnings-day-jump-continuation", _proposed("stk-earnings-jump-continuation"), "earnings", pool="stocks"),
    Plan("post-earnings-announcement-drift", _proposed("stk-pead-ear"), "earnings", pool="stocks"),
    Plan("pre-fomc-announcement-drift", _proposed("etf-pre-fomc-drift-h1"), "fomc_window"),
    Plan("hidden-markov-regime-allocation", _proposed("etf-risk-on-trend"), "regime_etf"),
    Plan("realised-covariance-regime-detection", _proposed("etf-corr-calm-trend"), "regime_etf"),
    Plan("buy-equity-after-vix-spike-above-30", _proposed("etf-panic-rebound"), "regime_etf"),
    Plan("buy-equity-after-vix-spike-above-30", _proposed("etf-vix-panic-rebound"), "regime_vix"),
    Plan("momentum-with-crash-protection-vol-scaled", _proposed("stk-rs-momentum-crash-protected"), "rs_crash", pool="stocks"),
    Plan("joint-time-series-and-cross-sectional-strategy", _proposed("fx-currency-strength-momentum"), "fx_strength"),
    Plan("g10-carry-long-high-rate-short-low-rate", _proposed("fx-carry-trend"), "fx_carry"),
    Plan("carry-with-volatility-filter", _proposed("fx-carry-vol-filter"), "fx_carry"),
    Plan("rate-differential-trend-fx", _proposed("fx-rate-diff-trend"), "fx_carry"),
    Plan("deep-momentum-network-lstm-trained-on-sharpe", verdict="not built", why="needs a trained LSTM model; no model training in phase 1"),
    Plan("momentum-transformer-with-changepoints", verdict="not built", why="needs a trained transformer model"),
    Plan("slow-momentum-with-fast-reversion-changepoint-detection", verdict="not built", why="needs the Gaussian-process changepoint model"),
    Plan("dynamic-momentum-learning-adaptive-lookback", verdict="not built", why="needs the adaptive-lookback learner"),
    Plan("learning-to-rank-cross-sectional-momentum", verdict="not built", why="needs a trained ranking model and a point-in-time universe"),
    Plan("deep-learning-statistical-arbitrage-residual-portfolios", verdict="not built", why="needs a trained model and factor returns"),
    Plan("deep-chart-pattern-recognition-head-and-shoulders-triangles", verdict="not built", why="needs a trained pattern model"),
    Plan("large-tick-trend-filter-trade-trend-only-where-tick-size-is-large", verdict="not a standalone signal", why="a filter for trend strategies"),
    Plan("uncertainty-gated-stock-ranker-skip-when-model-unsure", verdict="not a standalone signal", why="a gate on a ranker that does not exist yet"),
    Plan("regime-switching-volatility-forecast-for-sizing", verdict="not a standalone signal", why="sizing input"),
    Plan("long-calls-or-puts-on-high-conviction-plans", verdict="not a standalone signal", why="execution style; needs live option chains"),
    Plan("pre-earnings-run-up-and-iv-crush-options", verdict="needs data",
         why="option implied-volatility history and option prices (earnings dates now exist; OpenD F10 has IV only around each report)"),
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


def pool_symbols(pool: str) -> list[str]:
    return {"stocks": STOCKS, "etfs": ETFS, "all": list(dict.fromkeys(ETFS + STOCKS))}[pool]


def _xs_mask(values: pd.DataFrame, members: pd.DataFrame | None) -> pd.DataFrame:
    """Cross-sectional inputs limited to the universe of the day (no rank among non-members)."""
    if members is None:
        return values
    m = members.reindex(index=values.index, columns=values.columns).fillna(False).astype(bool)
    return values.where(m)


def build(builder: str, spec: StrategySpec, cache=OPEND_CACHE, members: pd.DataFrame | None = None,
          raw: dict[str, pd.DataFrame] | None = None) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Real bars for the spec's universe plus the research columns it reads. Returns (data, missing symbols).

    ``members`` (date x symbol, from the liquidity screen) limits cross-sectional ranks to
    the universe of the day; ``raw`` passes bars already loaded."""
    syms = [s for s in spec.universe if not s.startswith("$")]
    if spec.asset_class == "forex":
        ccys = _rate_ccys()
        financed = [s for s in syms if all(c in ccys for c in s.split("_"))]
        if builder in builders.BUILDERS:
            data, _ = builders.BUILDERS[builder](StrategySpec.from_dict({**spec.raw, "universe": financed}), cache)
            return data, [s for s in syms if s not in data]
        data = _fx(financed, spec.signal_tf)
        missing = [s for s in syms if s not in data]
        if builder == "fx_momentum_xs" and data:
            m = pd.DataFrame({s: panels.momentum(b) for s, b in data.items()})
            xs = panels.xs_percentile(m)
            data = {s: panels.with_columns(b, {"mom": m[s], "mom_xs": xs[s]}) for s, b in data.items()}
        return data, missing
    if builder in builders.BUILDERS:
        return builders.BUILDERS[builder](spec, cache, members=members)
    data = {s: raw[s] for s in syms if s in raw} if raw is not None else _us(syms, spec.signal_tf, cache)
    missing = [s for s in syms if s not in data]
    if builder == "high_ratio":
        hr = pd.DataFrame({s: panels.high_ratio(b) for s, b in data.items()})
        xs = panels.xs_percentile(_xs_mask(hr, members))
        data = {s: panels.with_columns(b, {"hi_ratio": hr[s], "hi_ratio_xs": xs[s]}) for s, b in data.items()}
    elif builder == "momentum_xs":
        m = pd.DataFrame({s: panels.momentum(b) for s, b in data.items()})
        xs = panels.xs_percentile(_xs_mask(m, members))
        data = {s: panels.with_columns(b, {"mom": m[s], "mom_xs": xs[s]}) for s, b in data.items()}
    elif builder == "sector_rs":
        etfs = _us(sorted(set(SECTOR.values())), "D1", cache)
        data = {s: panels.with_columns(b, panels.relative_strength(b, etfs[SECTOR[s]]))
                for s, b in data.items() if SECTOR.get(s) in etfs}
    elif builder == "pairs":
        out = {}
        for a, b in PAIRS:
            if a in data and b in data:
                z = panels.pair_zscore(data[a], data[b])
                out[a] = panels.with_columns(data[a], {"pair_z": z})
                out[b] = panels.with_columns(data[b], {"pair_z": -z.reindex(data[b].index)})
        data = out
    elif builder == "sessions":
        data = {s: panels.with_columns(b, panels.session_columns(b, duration(spec.signal_tf))) for s, b in data.items()}
    elif builder == "earn_jump":
        e = earn.Earnings(EARNINGS_CACHE)
        data = {s: panels.with_columns(b, panels.earnings_columns(b, e.load(s))) for s, b in data.items()
                if s not in ETFS}
    return data, missing


# --- FX (Oanda practice bid/ask history) -----------------------------------------------------

USD_LEGS = {"EUR": ("EUR_USD", False), "GBP": ("GBP_USD", False), "AUD": ("AUD_USD", False),
            "NZD": ("NZD_USD", False), "JPY": ("USD_JPY", True), "CAD": ("USD_CAD", True), "CHF": ("USD_CHF", True)}


def _rate_table():
    """The policy-rate history read by both the carry features (builders.RATES_FILE) and FX
    financing, so a signal and the cost of carrying it come from the same file."""
    from tradex.costs.models import PolicyRates
    if builders.RATES_FILE.exists():
        t = pd.read_csv(builders.RATES_FILE, comment="#")
        t.columns = [str(c).strip().lower() for c in t.columns]
        return PolicyRates(t)
    return PolicyRates()


def _rate_ccys() -> set[str]:
    """Currencies with a policy-rate history for financing. Pairs outside it are left out
    rather than financed with a guessed rate."""
    return set(_rate_table()._by_ccy)


def _costs(spec: StrategySpec, **overrides):
    """Cost model for a spec run outside ``prepare`` (the research loop): FX financing from the same
    rate history the carry features read; spreads are the cost model's defaults."""
    if spec.asset_class == "forex":
        overrides.setdefault("rates", _rate_table())
    return model_for(spec.asset_class, **overrides)


def _fx(pairs, tf) -> dict[str, pd.DataFrame]:
    from tradex.data.oanda_history import OandaHistory
    h = OandaHistory(OANDA_CACHE)
    out = {}
    for p in pairs:
        b = h.load(p, tf)
        if b is not None and len(b):
            out[p] = b
    return out


def fx_costs_and_rates(data: dict[str, pd.DataFrame], stress: float = 1.0):
    """Cost model paying the spread Oanda quoted at each bar's open, and USD conversion
    series from the H1 mid closes (stamped at the bar's close, so never ahead of time)."""
    from tradex.costs.models import pip_size
    spreads = {s: ((b["ask_open"] - b["bid_open"]) / pip_size(s)).clip(lower=0) for s, b in data.items()
               if {"ask_open", "bid_open"} <= set(b.columns)}
    rates = {}
    for ccy, (pair, invert) in USD_LEGS.items():
        h1 = _fx([pair], "H1").get(pair)
        if h1 is None:
            continue
        c = h1["close"].copy()
        c.index = c.index + pd.Timedelta(hours=1)
        rates[ccy] = 1.0 / c if invert else c
    return model_for("forex", measured_spreads=spreads, stress=stress, rates=_rate_table()), rates


# --- run context: universe screen, earnings, costs ---------------------------------------------

@dataclass
class Prepared:
    data: dict[str, pd.DataFrame]
    missing: list[str]
    costs: object
    cfg: EngineConfig
    tradable: dict[str, pd.Series] | None = None
    filter_ctx: dict | None = None
    variant: str | None = None
    universe_rule: str = "spec universe"
    members: dict | None = None


def _screened(plan: Plan, spec: StrategySpec, wf: WalkForwardConfig, cache, n: int):
    """Screen the pool at each fold start; rebuild until the timeline of the kept symbols gives the same dates."""
    screen = scr.Screen(plan.pool, n)
    raw = _us(pool_symbols(plan.pool), spec.signal_tf, cache)
    if not raw:
        return None
    embargo = duration(spec.signal_tf) * (spec.exit.max_bars + 1)
    keep = sorted(raw)
    for _ in range(4):
        timeline = pd.DatetimeIndex(sorted(set().union(*[set(raw[s].index) for s in keep])))
        dates = scr.rebalance_dates(timeline, wf, embargo, screen.lookback)
        members = scr.membership(raw, dates, screen)
        ever = scr.ever_members(members)
        if plan.builder == "pairs":
            legs = {s for a, b in PAIRS if a in ever and b in ever for s in (a, b)}
            ever = sorted(legs) or ever
        if ever == keep:
            break
        keep = ever
    sub = {s: raw[s] for s in keep}
    alt = StrategySpec.from_dict({**spec.raw, "universe": keep})
    idx = pd.DatetimeIndex(sorted(set().union(*[set(b.index) for b in sub.values()])))
    mf = scr.member_frame(members, keep, idx)
    data, _ = build(plan.builder, alt, cache, members=mf, raw=sub)
    tradable = scr.tradable_masks(members, data)
    if plan.builder == "pairs":
        partner = {a: b for a, b in PAIRS} | {b: a for a, b in PAIRS}
        tradable = {s: m & tradable.get(partner[s], m & False) for s, m in tradable.items()}
    return data, tradable, members, screen, len(raw)


def prepare(plan: Plan, spec: StrategySpec, wf: WalkForwardConfig, cfg: EngineConfig, cache=OPEND_CACHE,
            stress: float = 1.0, screen_n: int | None = None) -> Prepared:
    if spec.asset_class == "forex":
        data, missing = build(plan.builder, spec, cache)
        costs, rates = fx_costs_and_rates(data, stress) if data else (model_for("forex"), {})
        run_cfg = EngineConfig(**{**cfg.__dict__, "fx_rates": rates})
        return Prepared(data, missing, costs, run_cfg, variant="oanda-bidask-spreads",
                        universe_rule=f"{len(data)} pairs named by the spec")
    costs = model_for(spec.asset_class, stress=stress) if stress != 1.0 else model_for(spec.asset_class)
    tradable = members = None
    if plan.pool:
        got = _screened(plan, spec, wf, cache, screen_n or SCREEN_N[plan.pool])
        if got is None:
            return Prepared({}, [], costs, cfg)
        data, tradable, members, screen, pool_n = got
        missing: list[str] = []
        rule = (f"top {screen.n} of {pool_n} {plan.pool} by trailing {screen.lookback}-day median dollar volume, "
                f"re-screened at {len(members)} dates")
        variant = screen.label
    else:
        data, missing = build(plan.builder, spec, cache)
        rule, variant = "spec universe", None
    stocks = [s for s in data if s not in ETFS]
    dates = earn.filter_dates(list(data), etfs=ETFS, cache_dir=EARNINGS_CACHE)
    filter_ctx = {"earnings": dates}
    events = {s: list(dates[s]) for s in stocks if s in dates}
    run_cfg = EngineConfig(**{**cfg.__dict__, "events": events or None})
    if stocks:
        variant = f"{variant or 'spec'}|earnings"
    return Prepared(data, missing, costs, run_cfg, tradable, filter_ctx, variant, rule, members)


def _walk(spec, prep: Prepared, ledger, wf, th, tag: str = ""):
    return walk_forward(spec, prep.data, costs=prep.costs, engine_cfg=prep.cfg, wf=wf, thresholds=th,
                        tradable=prep.tradable, filter_ctx=prep.filter_ctx, trials=ledger,
                        data_key=f"{spec.signal_tf}:{prep.universe_rule}", variant=(prep.variant or "") + tag or None)


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


def _market(spec: StrategySpec) -> str:
    return "US stocks/ETFs (moomoo)" if spec.asset_class == "stocks" else "FX (Oanda practice bid/ask)"


def run_plan(plan: Plan, entry: catalog.CatalogEntry, ledger: TrialLedger, wf: WalkForwardConfig,
             th: Thresholds, cfg: EngineConfig, cache=OPEND_CACHE) -> dict:
    row = {"catalog_id": plan.catalog_id, "catalog_name": entry.name, "catalog_status": entry.status}
    if plan.spec is None:
        return row | {"result": plan.verdict, "why": plan.why}
    spec = StrategySpec.load(plan.spec)
    row |= {"strategy_id": spec.id, "spec": str(plan.spec.relative_to(ROOT)), "timeframe": spec.signal_tf,
            "market": _market(spec)}
    if plan.builder in ("fx", "fx_momentum_xs") and \
            not any((OANDA_CACHE / f"{s}_{spec.signal_tf}.csv").exists() for s in spec.universe):
        return row | {"result": "needs data", "why": NO_OANDA}
    try:
        if plan.builder in builders.BUILDERS:
            builders.preflight(plan.builder)
        prep = prepare(plan, spec, wf, cfg, cache)
    except DataUnavailable as exc:
        return row | {"result": "needs data", "why": str(exc)}
    if not prep.data:
        return row | {"result": "needs data", "why": f"no cached bars for {prep.missing or plan.pool}"}
    t0 = time.time()
    rep = _walk(spec, prep, ledger, wf, th)
    checks = gate_checks(rep.oos, th)
    failing = [c["rung"] for c in checks if not c["ok"]]
    oos = rep.oos
    if not failing:
        row["robustness"] = robustness(plan, spec, ledger, wf, th, cfg, cache)
    traded = sorted(rep.oos_trades["symbol"].unique()) if rep.oos_trades is not None and len(rep.oos_trades) else []
    return row | {
        "result": "pass" if not failing else "fail", "failing_rung": failing[0] if failing else None,
        "failing": failing, "reasons": rep.reasons, "universe_rule": prep.universe_rule, "variant": prep.variant,
        "symbols": sorted(prep.data), "missing_symbols": prep.missing, "symbols_traded_oos": traded,
        "universe_by_date": {str(k.date()): v for k, v in (prep.members or {}).items()},
        "bars_used": int(sum(len(b) for b in prep.data.values())),
        "data_from": str(min(b.index[0] for b in prep.data.values()).date()),
        "data_to": str(max(b.index[-1] for b in prep.data.values()).date()),
        "oos_trades": int(oos.get("trades", 0)), "profit_factor": _f(oos.get("profit_factor")),
        "sharpe": _f(oos.get("sharpe")), "deflated_sharpe": _f(oos.get("dsr")), "psr": _f(oos.get("psr")),
        "n_trials": rep.n_trials, "n_trials_this_run": rep.n_trials_this_run,
        "max_drawdown": _f(oos.get("max_drawdown")), "total_return": _f(oos.get("total_return")),
        "win_rate": _f(oos.get("win_rate")), "costs_share_of_gross": _f(oos.get("costs_share_of_gross")),
        "exit_reasons": rep.oos_trades["exit_reason"].str.replace(r"\s*[-+]?\d.*$", "", regex=True).value_counts().to_dict()
        if rep.oos_trades is not None and len(rep.oos_trades) else {},
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
    """Extra checks for a pass, not part of the stage-3 gate: double spread and slippage, and
    (screened plans) a universe twice as wide. Each is recorded as its own trial variant."""
    out = {"costs_x2": _brief(_walk(spec, prepare(plan, spec, wf, cfg, cache, stress=2.0), ledger, wf, th, "|costs_x2"))}
    if plan.pool:
        prep = prepare(plan, spec, wf, cfg, cache, screen_n=2 * SCREEN_N[plan.pool])
        out["screen_x2"] = _brief(_walk(spec, prep, ledger, wf, th)) | {"universe_rule": prep.universe_rule}
    return out


LABELS = {"costs_x2": "spread and slippage doubled", "screen_x2": "screened universe twice as wide"}


def _changes(rows: list[dict], previous: dict[str, dict]) -> list[str]:
    out = ["| Strategy | Run 2 | This run | Result change |",
           "|---|---|---|---|"]
    for r in rows:
        if r["result"] not in ("pass", "fail"):
            continue
        p = previous.get(r["strategy_id"])
        now = (f"{r['oos_trades']} trades, PF {r['profit_factor']}, Sharpe {r['sharpe']}, DSR {r['deflated_sharpe']} "
               f"(N {r['n_trials']}), {len(r['symbols'])} symbols")
        if p is None:
            out.append(f"| `{r['strategy_id']}` | not run | {now} | new: **{r['result']}** |")
            continue
        was = (f"{p['oos_trades']} trades, PF {p['profit_factor']}, Sharpe {p['sharpe']}, DSR {p['deflated_sharpe']} "
               f"(N {p['n_trials']}), {len(p['symbols'])} symbols")
        out.append(f"| `{r['strategy_id']}` | {was} | {now} | {p['result']} -> **{r['result']}** |")
    return out


def render_md(rows: list[dict], meta: dict) -> str:
    ran = [r for r in rows if r["result"] in ("pass", "fail")]
    passed = [r for r in ran if r["result"] == "pass"]
    lines = [
        "# Phase-1 gate run", "",
        f"Run {meta['run_at']} on real bars only (moomoo OpenD, qfq-adjusted; Oanda practice bid/ask candles for FX). "
        f"Stage-3 thresholds from `config/gates/thresholds.yaml`: {meta['thresholds']}.",
        f"Walk-forward: {meta['walk_forward']}. Engine: {meta['engine']}. "
        "Costs: moomoo SG fees and default spread/slippage for US stocks; for FX the spread Oanda quoted at each "
        "bar's open plus 0.2 pip slippage, and financing from official central-bank policy rates (data/cache/macro).",
        f"US universes: {meta['universe']}",
        "", f"**{len(passed)} of {len(ran)} strategies run pass the gate.** "
        f"{sum(r['result'] == 'needs data' for r in rows)} need data, "
        f"{sum(r['result'] in ('not built', 'not a standalone signal') for r in rows)} are not built or not standalone.", "",
    ] + [f"Note on `{r['strategy_id']}`: it passes, but fails the robustness check(s) "
         f"{', '.join(LABELS[k] for k, v in r.get('robustness', {}).items() if v['gate'] != 'pass')}; treat it as unproven."
         for r in passed if any(v["gate"] != "pass" for v in r.get("robustness", {}).values())] + [
        "",
        "## Strategies run", "",
        "| Strategy | Catalog entry | Market | Universe | Bars used | OOS trades | PF | Sharpe | DSR (N) | Max DD | Folds + | Result | Failing rung |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in ran:
        lines.append(f"| `{r['strategy_id']}` | {r['catalog_name']} | {r['market']} {r['timeframe']}, "
                     f"{r['data_from']}..{r['data_to']} | {r['universe_rule']}; {len(r['symbols'])} symbols ever in it | "
                     f"{r['bars_used']:,} | {r['oos_trades']} | {r['profit_factor']} | "
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
                lines.append(f"- `{r['strategy_id']}`, {LABELS[k]}: {v['oos_trades']} trades, PF {v['profit_factor']}, "
                             f"Sharpe {v['sharpe']}, DSR {v['deflated_sharpe']}, max DD {v['max_drawdown']}, "
                             f"folds {v['folds_profitable']}: **{v['gate']}**" +
                             (f" ({'; '.join(v['reasons'])})" if v["reasons"] else ""))
    lines += ["", "## Not run", "", "| Catalog entry | Status in catalog | Result | Why |", "|---|---|---|---|"]
    for r in rows:
        if r["result"] not in ("pass", "fail"):
            lines.append(f"| {r['catalog_name']} | {r['catalog_status']} | {r['result']} | {r['why']} |")
    if meta.get("previous"):
        lines += ["", "## What changed vs run 2", ""] + [f"- {c}" for c in meta["changes"]] + [""] + \
            _changes(rows, meta["previous"])
    lines += ["", "## Caveats", ""] + [f"- {c}" for c in meta["caveats"]]
    return "\n".join(lines) + "\n"


def earnings_coverage(cache=EARNINGS_CACHE) -> dict:
    e = earn.Earnings(cache)
    src: dict[str, int] = {}
    late = []
    n = 0
    for s in STOCKS:
        t = e.load(s)
        if t is None:
            continue
        n += 1
        for k, v in t["source"].value_counts().items():
            src[k] = src.get(k, 0) + int(v)
        b = load_cached(s, "D1")
        if len(t) and b is not None and len(b) and pd.Timestamp(t["date"].min()) > b.index[0].tz_localize(None) + pd.Timedelta(days=200):
            late.append(f"{s} from {t['date'].min()}")
    return {"symbols": n, "reports_by_source": src, "starts_late": late}


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
        if only and plan.catalog_id not in only and (plan.spec is None or plan.spec.stem not in only):
            continue
        row = run_plan(plan, entries[plan.catalog_id], ledger, wf, th, cfg)
        print(f"{row.get('strategy_id', plan.catalog_id)}: {row['result']} "
              f"{row.get('failing_rung') or row.get('why', '')}", flush=True)
        rows.append(row)
    cov = earnings_coverage()
    previous = {}
    if PREVIOUS.exists():
        previous = {r["strategy_id"]: r for r in json.loads(PREVIOUS.read_text())["rows"] if r.get("strategy_id")
                    and r["result"] in ("pass", "fail")}
    meta = {
        "run_at": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC"),
        "thresholds": {k: getattr(th, k) for k in th.__dataclass_fields__},
        "walk_forward": {k: getattr(wf, k) for k in wf.__dataclass_fields__},
        "engine": {"initial_equity": cfg.initial_equity, "risk_pct": cfg.risk_pct, "max_leverage": cfg.max_leverage},
        "universe": (f"screened at every walk-forward test-fold start (and yearly before the first) from a pool of "
                     f"{len(STOCKS)} stocks (the S&P 100 as of Oct 2026 plus the first run's names) and {len(ETFS)} ETFs; "
                     f"top {SCREEN_N['stocks']} stocks, {SCREEN_N['etfs']} ETFs, or {SCREEN_N['all']} of both, ranked by "
                     "trailing 60-day median dollar volume using bars before the screen date only."),
        "earnings_coverage": cov,
        "previous": previous,
        "changes": [
            "Merged with the proposed-strategy work: the 11 strategies in strategies/proposed/ run here, built by "
            "tradex.research.builders from injected calendars and rate histories. The three stock ones run on the "
            "same liquidity screen as the seeds; the ETF ones keep their full index/sector-SPDR universe.",
            "Policy rates for all eight currencies from the central banks' own data (FRED for the Fed, ECB, BoE, RBA, "
            "BoC Valet, SNB data portal; BoJ and RBNZ via the BIS policy-rate dataset), cached in data/cache/macro. "
            "The carry features and FX financing read this one file. CAD, CHF and NZD pairs now run, financed.",
            "The bundled cost table tradex/costs/policy_rates.csv is rebuilt from the same official series (the old "
            "one was written from memory: it missed the 2026 ECB, RBA and Fed moves and dated RBA changes a day early).",
            "Scheduled FOMC meetings 2006-2027 from federalreserve.gov feed the pre-FOMC drift strategy; Cboe VIX daily "
            "history feeds `etf-vix-panic-rebound`, the literal VIX>30 form of the panic-rebound entry (new).",
            "Earnings strategies from the proposals read the run-2 earnings calendar (OpenD release dates and timing, "
            "SEC filing-date proxies before that).",
            "30-minute OpenD bars for the 22 ETFs (no new history quota: 124 of 300 still used): "
            "`etf-intraday-momentum-m30` takes the first half hour as its signal and enters at 15:30 (new).",
            "Trial ledger: every parameter set of every variant run here is recorded, so N grows for re-run strategies.",
        ],
        "caveats": [
            "Survivorship bias remains: the candidate pool is today's S&P 100 and today's ETFs. OpenD has no delisted "
            "names, so stocks that left the index or failed since 2006 can never be picked by the screen. Results on "
            "screened universes are less hand-picked than the first run, not survivorship-free.",
            "The S&P 100 list is the October 2026 membership as written in tradex/research/universe.py (checked against "
            "OpenD's S&P 500 plate); the screen ranks within it, but membership itself is today's.",
            "Earnings dates before OpenD's coverage (about 2013 on) are SEC filing dates, a proxy: they can lag the "
            "release by a day and their before/after-market timing is unknown, so both filters treat them as before the "
            "open (conservative)." + (f" Coverage starts late for: {', '.join(cov['starts_late'])}." if cov['starts_late'] else ""),
            "FX financing is the official policy-rate differential minus Oanda's admin fee; Oanda does not publish "
            "historical financing rates, so actual swap rates (which track interbank rates, not policy rates) differ.",
            "JPY and NZD rates are the BIS compilation of the BoJ and RBNZ series (rbnz.govt.nz refuses scripted "
            "downloads; BoJ has no policy-rate series). The BIS JPY series holds 0.05% (the 0-0.1% call-rate "
            "guideline) until 2016-09-21 and -0.1% from then. Swiss rates before 2019-06-13 are the SNB's 3-month "
            "Libor target-range midpoint, dated at month end (up to a month late, never early).",
            "FOMC statement times are not on the Fed's calendar pages: 14:00 New York from 2013 (12:30 on 2011-2012 "
            "press-conference days, 14:15 before). The 60-minute bars the pre-FOMC strategy uses start in 2018.",
            "FX seeds' no_high_impact_news_30m filter is inactive: there is no historical macro calendar before 2026.",
            "Stock spreads are the cost model's defaults (1 bp half-spread + 2 bp slippage), not measured quotes.",
            "Fixed moomoo fees (US$0.99 + 9% GST per order) weigh heavily at the US$10,000 test equity.",
            "OpenD history starts 2006-09 for daily bars and 2018-09 for 60-minute bars; Oanda history here is 10 years.",
            "DSR's N is every parameter set ever recorded for the strategy in the trial ledger, across all variants.",
        ],
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    if not only:
        slim = {k: v for k, v in meta.items() if k != "previous"} | {"previous_run": str(PREVIOUS.relative_to(ROOT))}
        (RESULTS / "phase1_gate.json").write_text(json.dumps({"meta": slim, "rows": rows}, indent=1, default=str))
        (RESULTS / "phase1_gate.md").write_text(render_md(rows, meta))
    else:   # a single-strategy run keeps its own result file and leaves the full-run report alone
        (RESULTS / "single").mkdir(exist_ok=True)
        slim = {k: v for k, v in meta.items() if k != "previous"}
        for r in rows:
            (RESULTS / "single" / f"{r.get('strategy_id', r['catalog_id'])}.json").write_text(
                json.dumps({"meta": slim, "row": r}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
