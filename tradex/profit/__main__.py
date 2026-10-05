"""``python -m tradex.profit <report>``: read-only reports over a ledger. Nothing here writes to it."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from tradex.core.ledger import Ledger


def _frames(data: str, tf: str) -> dict[str, pd.DataFrame]:
    from tradex.data.providers import CsvProvider
    prov = CsvProvider(data)
    out = {}
    for p in sorted(Path(data).glob(f"*_{tf}.csv")):
        sym = p.name[: -len(f"_{tf}.csv")]
        out[sym] = prov.get_bars(sym, tf)
    return out


def _series_rate_fn(frames: dict[str, pd.DataFrame]):
    """USD per unit of a currency from the recorded XXX_USD / USD_XXX closes. No fallback: a currency without a
    series raises, and the report that needed it says so."""
    from tradex.core.replay import usd_per_unit_series
    series = usd_per_unit_series(frames)

    def rate(ccy: str, ts: pd.Timestamp) -> float:
        if ccy == "USD":
            return 1.0
        s = series.get(ccy)
        if s is None:
            raise LookupError(f"no {ccy}/USD series in the data directory")
        i = s.index.searchsorted(ts, side="right")
        if i == 0:
            raise LookupError(f"{ccy}/USD has no bar at or before {ts}")
        return float(s.iloc[i - 1])
    return rate


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tradex.profit")
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("exits", help="compare the exit engine with the plain stop/target/time-stop on the ledger's plans")
    e.add_argument("--ledger", required=True)
    e.add_argument("--data", required=True, help="directory of {SYMBOL}_{TF}.csv bars")
    e.add_argument("--config", default="config/exits.yaml")
    e.add_argument("--which", choices=["accepted", "blocked", "all"], default="accepted")
    e.add_argument("--extra-leg-cost-r", type=float, default=0.0)

    c = sub.add_parser("calibrate", help="probability calibration on forward trades (refuses on thin evidence)")
    c.add_argument("--ledger", required=True)
    c.add_argument("--min-trades", type=int, default=100)

    k = sub.add_parser("costs", help="realised stock fills against the default cost model (forex needs a live spread "
                                     "source, so run it from the runtime)")
    k.add_argument("--ledger", required=True)

    f = sub.add_parser("filters", help="what each gate's blocked plans would have earned, with verdicts")
    f.add_argument("--ledger", required=True)
    f.add_argument("--min-plans", type=int, default=20)

    r = sub.add_parser("radar", help="ruin radar on the latest ensemble snapshot")
    r.add_argument("--ledger", required=True)
    r.add_argument("--data", required=True, help="directory of {SYMBOL}_{TF}.csv bars with the history to replay")
    r.add_argument("--tf", default="D1")
    r.add_argument("--policy", default="config/risk/policy.yaml")
    r.add_argument("--windows", default="config/shock_windows.yaml")
    r.add_argument("--json", action="store_true")

    a = ap.parse_args(argv)
    led = Ledger(a.ledger, read_only=True)

    if a.cmd == "exits":
        from tradex.profit.exits import ExitPolicy, evaluate_on_ledger, load_exit_policy
        frames_by_tf: dict[str, dict[str, pd.DataFrame]] = {}

        def bars_for(sym: str, tf: str):
            if tf not in frames_by_tf:
                frames_by_tf[tf] = _frames(a.data, tf)
            return frames_by_tf[tf].get(sym)
        out = evaluate_on_ledger(led, bars_for, {"plain": ExitPolicy.baseline(), "engine": load_exit_policy(a.config)},
                                 which=a.which, baseline="plain", extra_leg_cost_r=a.extra_leg_cost_r)
        out.pop("per_plan")
        print(json.dumps(out, indent=2))
        return 0

    if a.cmd == "calibrate":
        from tradex.profit.calibration import CalibrationConfig, forward_trades, fit_from_ledger
        cfg = CalibrationConfig(min_trades=a.min_trades)
        m = fit_from_ledger(led, cfg)
        print(json.dumps({**m.to_dict(), "reliability": m.reliability(forward_trades(led, label=cfg.label))}, indent=2))
        return 0

    if a.cmd == "costs":
        from tradex.costs.models import model_for
        from tradex.profit.costcal import calibrate_costs, calibration_report
        print(json.dumps(calibration_report(calibrate_costs(led, {"stocks": model_for("stocks")})), indent=2))
        return 0

    if a.cmd == "filters":
        from tradex.profit.whatif import dashboard_payload
        print(json.dumps(dashboard_payload(led, a.min_plans), indent=2, default=str))
        return 0

    if a.cmd == "radar":
        import yaml
        from tradex.core.replay import factor_returns
        from tradex.profit.ruin import load_shock_windows, radar_from_snapshot
        snap = led.db.execute("SELECT payload FROM events WHERE kind='snapshot' AND book='ensemble' "
                              "ORDER BY time DESC, seq DESC LIMIT 1").fetchone()
        if snap is None:
            raise SystemExit("no ensemble snapshot in the ledger yet")
        frames = _frames(a.data, a.tf)
        if not frames:
            raise SystemExit(f"no *_{a.tf}.csv bars in {a.data}")
        ac = {s: ("forex" if "_" in s else "stocks") for s in frames}
        pol = yaml.safe_load(Path(a.policy).read_text())
        rep = radar_from_snapshot(json.loads(snap[0]), factor_returns(frames, ac), load_shock_windows(a.windows),
                                  _series_rate_fn(frames), pol)
        print(json.dumps(rep.to_dict(), indent=2) if a.json else rep.telegram())
        return 0 if rep.level != "breach" else 2
    return 1


if __name__ == "__main__":
    sys.exit(main())
