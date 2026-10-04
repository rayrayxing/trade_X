import numpy as np
import pandas as pd
import pytest

from tradex.backtest.validation import Thresholds
from tradex.data.synthetic import synthetic_bars
from tradex.research import catalog, gate, panels
from tradex.strategy.spec import StrategySpec, load_dir


def test_every_plan_names_a_catalog_entry_and_covers_all_build_and_seed_entries():
    entries = {e.id: e for e in catalog.load()}
    planned = {p.catalog_id for p in gate.PLANS}
    assert planned <= set(entries)
    assert {e.id for e in entries.values() if e.status_kind in ("Build", "Seed")} <= planned
    for p in gate.PLANS:
        assert (p.spec is None) == (p.verdict is not None)
        if p.spec:
            assert p.spec.exists()


def test_research_specs_validate():
    specs = load_dir(gate.SPECS)
    assert len(specs) >= 6
    for s in specs:
        assert s.validate() == [], s.id


def test_gate_checks_report_ladder_order():
    th = Thresholds()
    oos = {"trades": 50, "profit_factor": 1.5, "dsr": 0.2, "max_drawdown": -0.6, "positive_folds": 0.8}
    checks = gate.gate_checks(oos, th)
    assert [c["rung"] for c in checks] == list(gate.RUNGS)
    assert [c["rung"] for c in checks if not c["ok"]] == ["trades", "deflated_sharpe", "max_drawdown"]


def _panel(n=400):
    return {s: synthetic_bars(n, seed=i) for i, s in enumerate("ABCDEF")}


def test_cross_sectional_columns_are_causal():
    data = _panel()
    cut = {s: b.iloc[:300] for s, b in data.items()}
    full = panels.xs_percentile(pd.DataFrame({s: panels.high_ratio(b, 100) for s, b in data.items()}))
    part = panels.xs_percentile(pd.DataFrame({s: panels.high_ratio(b, 100) for s, b in cut.items()}))
    pd.testing.assert_frame_equal(full.iloc[:300], part)
    z_full = panels.pair_zscore(data["A"], data["B"], 100, 30)
    z_part = panels.pair_zscore(cut["A"], cut["B"], 100, 30)
    pd.testing.assert_series_equal(z_full.iloc[:300], z_part)
    assert full.iloc[-1].between(0, 1).all()


def test_session_columns_on_opend_stamped_hours():
    ny = pd.DatetimeIndex([f"2026-09-{d} {h}" for d in (29, 30) for h in
                           ("09:30", "10:30", "11:30", "12:30", "13:30", "14:30", "15:00")], tz="America/New_York")
    idx = ny.tz_convert("UTC").astype("datetime64[ns, UTC]")
    close = np.arange(14, dtype=float) + 100
    bars = pd.DataFrame({"open": close - 0.5, "high": close + 1, "low": close - 1, "close": close, "volume": 1.0}, index=idx)
    c = panels.session_columns(bars)
    assert c["first_bar"].tolist() == [1, 0, 0, 0, 0, 0, 0] * 2
    assert c["last_full"].tolist() == [0, 0, 0, 0, 0, 1, 0] * 2
    assert c["gap"].iloc[0] == 0 and c["gap"].iloc[7] == pytest.approx(bars["open"].iloc[7] / bars["close"].iloc[6] - 1)


def test_fx_plans_report_needs_data_without_oanda_history(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "OANDA_CACHE", tmp_path)
    entries = {e.id: e for e in catalog.load()}
    plan = next(p for p in gate.PLANS if p.builder == "fx")
    row = gate.run_plan(plan, entries[plan.catalog_id], None, None, Thresholds(), None)
    assert row["result"] == "needs data" and "oanda" in row["why"].lower()


def test_stock_plan_runs_on_cached_bars(tmp_path, monkeypatch):
    from tradex.backtest.engine import EngineConfig
    from tradex.backtest.validation import WalkForwardConfig
    from tradex.data.providers import CsvProvider
    from tradex.research.trials import TrialLedger
    prov = CsvProvider(tmp_path)
    for i, s in enumerate(["SPY", "QQQ", "IWM"]):
        prov.save(s, "D1", synthetic_bars(900, seed=i))
    entries = {e.id: e for e in catalog.load()}
    plan = next(p for p in gate.PLANS if p.catalog_id == "rsi-2-pullback-in-uptrend-seed")
    row = gate.run_plan(plan, entries[plan.catalog_id], TrialLedger(tmp_path / "t.sqlite"),
                        WalkForwardConfig(n_folds=2, grid_points=2), Thresholds(), EngineConfig(), cache=tmp_path)
    assert row["result"] in ("pass", "fail") and row["symbols"] == ["IWM", "QQQ", "SPY"]
    assert row["n_trials"] == 4 and row["bars_used"] == 2700
    assert row["failing_rung"] in gate.RUNGS + (None,)
