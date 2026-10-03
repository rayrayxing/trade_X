"""Command line: tradex check | backtest | validate | select | fetch | compare-feeds."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from tradex.backtest import metrics
from tradex.backtest.engine import EngineConfig, run_backtest
from tradex.backtest.validation import ValidationReport, WalkForwardConfig, walk_forward
from tradex.data.providers import AlpacaProvider, CachedProvider, CsvProvider, MassiveProvider, OandaProvider
from tradex.strategy.spec import StrategySpec, load_dir


def _load_data(spec: StrategySpec, data_dir: str, symbols: list[str] | None) -> dict[str, pd.DataFrame]:
    prov = CsvProvider(data_dir)
    syms = symbols or [s for s in spec.universe if not s.startswith("$")]
    out = {}
    for s in syms:
        p = prov.path(s, spec.signal_tf)
        if p.exists():
            out[s] = prov.get_bars(s, spec.signal_tf)
        else:
            print(f"warning: no data file {p}", file=sys.stderr)
    if not out:
        raise SystemExit("no data found; run `tradex fetch` first")
    return out


def save_validation(rep: ValidationReport, spec: StrategySpec, out: Path) -> Path:
    d = out / spec.id
    d.mkdir(parents=True, exist_ok=True)
    payload = rep.to_dict() | {"asset_class": spec.asset_class, "symbols": spec.universe,
                               "cap_risk_pct": spec.cap_risk_pct}
    (d / "validation.json").write_text(json.dumps(payload, indent=2, default=str))
    if rep.oos_returns is not None:
        rep.oos_returns.rename("ret").to_csv(d / "oos_returns.csv", index_label="date")
    if rep.oos_trades is not None and not rep.oos_trades.empty:
        rep.oos_trades.to_csv(d / "oos_trades.csv", index=False)
    return d


def load_records(reports: Path, reference_risk_pct: float = 1.0):
    from tradex.selection.allocator import StrategyRecord
    recs = []
    for vj in sorted(reports.glob("*/validation.json")):
        v = json.loads(vj.read_text())
        rp = vj.parent / "oos_returns.csv"
        tp = vj.parent / "oos_trades.csv"
        rets = pd.read_csv(rp, index_col=0, parse_dates=True)["ret"] if rp.exists() else pd.Series(dtype=float)
        trades = pd.read_csv(tp) if tp.exists() else pd.DataFrame()
        recs.append(StrategyRecord(
            id=v["strategy_id"], asset_class=v["asset_class"], status=v["status"], oos_returns=rets,
            oos_trades=trades, dsr=v["oos"].get("dsr", 0.0), reference_risk_pct=reference_risk_pct,
            cap_risk_pct=v.get("cap_risk_pct", 3.0), symbols=v.get("symbols", []),
        ))
    return recs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tradex")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="schema-check strategy files")
    c.add_argument("path", default="strategies", nargs="?")

    for name in ("backtest", "validate"):
        b = sub.add_parser(name)
        b.add_argument("spec")
        b.add_argument("--data", required=True, help="directory of {SYMBOL}_{TF}.csv files")
        b.add_argument("--symbols", nargs="*")
        b.add_argument("--start")
        b.add_argument("--end")
        b.add_argument("--equity", type=float, default=10_000)
        b.add_argument("--risk-pct", type=float, default=1.0)
        b.add_argument("--leverage", type=float, default=1.0)
        b.add_argument("--out", default="reports")
        if name == "validate":
            b.add_argument("--folds", type=int, default=5)
            b.add_argument("--grid-points", type=int, default=4)

    s = sub.add_parser("select", help="allocate the risk budget across validated strategies")
    s.add_argument("--reports", default="reports")
    s.add_argument("--regime-bars", required=True, help="CSV of the regime reference (e.g. SPY_D1.csv)")
    s.add_argument("--mode", default="research", choices=["research", "paper", "live"])
    s.add_argument("--out", default="reports/allocation.json")

    f = sub.add_parser("fetch", help="download bars into a CSV cache (keys from environment)")
    f.add_argument("--provider", required=True, choices=["alpaca", "massive", "oanda"])
    f.add_argument("--symbols", nargs="+", required=True)
    f.add_argument("--tf", default="D1")
    f.add_argument("--start", required=True)
    f.add_argument("--end", required=True)
    f.add_argument("--out", default="data/cache")

    cf = sub.add_parser("compare-feeds", help="Alpaca IEX vs Massive consolidated bars")
    cf.add_argument("--symbols", nargs="+", default=["SPY", "QQQ", "IWM", "NVDA", "TSLA", "AMD", "AAPL", "MSFT", "META", "AMZN"])
    cf.add_argument("--tf", default="M15", choices=["D1", "M15", "H1"])
    cf.add_argument("--start", required=True)
    cf.add_argument("--end", required=True)
    cf.add_argument("--cache", default="data/cache")
    cf.add_argument("--out", default="reports/feed_comparison.md")

    a = ap.parse_args(argv)

    if a.cmd == "check":
        bad = 0
        for spec in load_dir(a.path):
            errs = spec.validate()
            print(f"{'FAIL' if errs else 'ok  '} {spec.id}")
            for e in errs:
                print(f"     - {e}")
            bad += bool(errs)
        return 1 if bad else 0

    if a.cmd in ("backtest", "validate"):
        spec = StrategySpec.load(a.spec)
        errs = spec.validate()
        if errs:
            raise SystemExit("schema check failed:\n  " + "\n  ".join(errs))
        data = _load_data(spec, a.data, a.symbols)
        cfg = EngineConfig(initial_equity=a.equity, risk_pct=a.risk_pct, max_leverage=a.leverage,
                           start=a.start, end=a.end)
        out = Path(a.out)
        if a.cmd == "backtest":
            res = run_backtest(spec, data, cfg=cfg)
            summary = metrics.summarize(res.equity, res.trades)
            d = out / spec.id
            d.mkdir(parents=True, exist_ok=True)
            res.trades.to_csv(d / "trades.csv", index=False)
            res.equity.to_csv(d / "equity.csv", index_label="ts")
            (d / "backtest.json").write_text(json.dumps(summary | {"warnings": res.warnings}, indent=2, default=str))
            print(json.dumps(summary, indent=2, default=str))
        else:
            rep = walk_forward(spec, data, engine_cfg=cfg, wf=WalkForwardConfig(n_folds=a.folds, grid_points=a.grid_points))
            d = save_validation(rep, spec, out)
            print(f"{spec.id}: {rep.status.upper()}  dsr={rep.oos['dsr']:.2f} trades={rep.oos.get('trades')} "
                  f"pf={rep.oos.get('profit_factor', 0):.2f}  -> {d}")
            for r in rep.reasons:
                print(f"  - {r}")
        return 0

    if a.cmd == "select":
        from tradex.selection.allocator import AllocationConfig, allocate, blend_report
        from tradex.selection.regime import classify_regime, daily_labels
        recs = load_records(Path(a.reports))
        bars = CsvProvider(Path(a.regime_bars).parent).get_bars(*Path(a.regime_bars).stem.rsplit("_", 1))
        reg = classify_regime(bars)
        snap = allocate(recs, reg["label"].iloc[-1], daily_labels(reg), AllocationConfig(mode=a.mode))
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(snap.to_json())
        print(snap.to_json())
        print(json.dumps(blend_report(recs, snap), indent=2))
        return 0

    if a.cmd == "fetch":
        inner = {"alpaca": AlpacaProvider, "massive": MassiveProvider, "oanda": OandaProvider}[a.provider]()
        prov = CachedProvider(inner, a.out, a.provider)
        for sym in a.symbols:
            df = prov.get_bars(sym, a.tf, a.start, a.end)
            print(f"{sym}: {len(df)} bars")
        return 0

    if a.cmd == "compare-feeds":
        from tradex.data.compare import compare_feeds, render_report, to_session_dates
        iex = CachedProvider(AlpacaProvider(feed="iex"), a.cache, "alpaca_iex")
        ref = CachedProvider(MassiveProvider(), a.cache, "massive")
        results = []
        for sym in a.symbols:
            x, y = iex.get_bars(sym, a.tf, a.start, a.end), ref.get_bars(sym, a.tf, a.start, a.end)
            if a.tf == "D1":
                x, y = to_session_dates(x), to_session_dates(y)
            try:
                results.append(compare_feeds(x, y, sym, a.tf))
            except ValueError as exc:
                print(f"{sym}: {exc}", file=sys.stderr)
        text = render_report(results)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text)
        Path(a.out).with_suffix(".json").write_text(json.dumps([r.to_dict() for r in results], indent=2))
        print(text)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
