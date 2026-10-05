import json

import numpy as np
import pytest

from pfx import H, T0, plan
from tradex.core.ledger import Ledger
from tradex.core.records import Close, Counterfactual, Fill
from tradex.profit.calibration import (CalibrationConfig, ForwardTrade, TradeProbabilityModel, fit_from_ledger,
                                       fit_isotonic, fit_platt, forward_trades)


def trades(n, true_of, seed=0, raw_lo=0.3, raw_hi=0.7, loss_r=0.9):
    rng = np.random.default_rng(seed)
    raw = rng.uniform(raw_lo, raw_hi, n)
    out = []
    for i, p in enumerate(raw):
        y = int(rng.random() < true_of(p))
        out.append(ForwardTrade(f"d{i}", f"2026-04-{1 + i // 40:02d}T{i % 24:02d}:00:00+00:00", float(p), y,
                                2.0 if y else -loss_r))
    return out


def test_below_the_minimum_returns_the_base_rate_flagged_uncalibrated():
    ts = trades(60, lambda p: 0.3)
    m = TradeProbabilityModel.fit(ts)
    c = m.predict(0.55, reward_risk=2.0)
    assert (c.p, c.calibrated, c.method, c.n_forward) == (0.55, False, "base_rate", 60)
    assert "need 100" in c.reason and c.ev_r == pytest.approx(0.55 * 2 - 0.45)
    pl = plan(p_target=0.55)
    assert m.annotate(pl) is pl                              # the plan is untouched


def test_no_trades_and_single_class_are_refused():
    assert not TradeProbabilityModel.fit([]).calibrated
    same = [ForwardTrade(f"d{i}", "t", 0.5, 0, -1.0) for i in range(150)]
    m = TradeProbabilityModel.fit(same)
    assert not m.calibrated and "same outcome" in m.reason


def test_platt_pulls_an_overconfident_base_rate_down_to_the_forward_rate():
    ts = trades(400, lambda p: 0.55 * p, seed=1)             # the finaliser says ~0.5, the truth is ~0.27
    m = TradeProbabilityModel.fit(ts, CalibrationConfig(method="platt"))
    assert m.calibrated and m.method == "platt"
    for raw in (0.35, 0.5, 0.65):
        assert m.predict(raw).p == pytest.approx(0.55 * raw, abs=0.07)
    assert m.beats_raw and m.annotate(plan(p_target=0.5)).p_source == "model"
    assert m.annotate(plan(p_target=0.5)).p_target < 0.35


def test_isotonic_is_monotone_and_used_from_its_minimum():
    ts = trades(800, lambda p: np.clip(0.2 + 0.6 * (p - 0.3) / 0.4, 0, 1) ** 1.5, seed=2)
    m = TradeProbabilityModel.fit(ts)
    assert m.method == "isotonic"
    grid = [m.predict(x).p for x in np.linspace(0.3, 0.7, 40)]
    assert all(b >= a - 1e-9 for a, b in zip(grid, grid[1:]))
    assert m.predict(0.3).p < m.predict(0.7).p
    short = TradeProbabilityModel.fit(ts[:200])
    assert short.method == "platt"                            # isotonic needs 300
    forced = TradeProbabilityModel.fit(ts[:200], CalibrationConfig(method="isotonic"))
    assert forced.method == "platt" and "isotonic needs" in forced.reason


def test_a_base_rate_that_was_already_right_is_not_replaced_unless_it_improves_out_of_sample():
    ts = trades(600, lambda p: p, seed=3)
    m = TradeProbabilityModel.fit(ts, CalibrationConfig(method="platt"))
    assert m.predict(0.5).p == pytest.approx(0.5, abs=0.08)
    pl = plan(p_target=0.5)
    out = m.annotate(pl)
    assert out is pl or abs(out.p_target - 0.5) < 0.08


def test_platt_without_spread_in_the_base_rate_falls_back_to_the_forward_rate():
    a, b = fit_platt(np.full(200, 0.4), np.array([1.0] * 50 + [0.0] * 150))
    assert a == 0.0 and 1 / (1 + np.exp(-b)) == pytest.approx(0.25, abs=0.01)


def test_isotonic_pav_pools_violators():
    k, f = fit_isotonic(np.array([1.0, 2, 3, 4, 5]), np.array([0.0, 1, 0, 1, 1]))
    assert list(f) == pytest.approx([0.0, 0.5, 0.5, 1.0, 1.0])
    assert np.all(np.diff(f) >= 0)


def test_ev_in_r_uses_the_measured_loss_size():
    ts = trades(300, lambda p: 0.4, seed=4, loss_r=0.6)
    m = TradeProbabilityModel.fit(ts)
    assert m.loss_r == pytest.approx(0.6, abs=0.01)
    c = m.predict(0.5, reward_risk=2.0)
    assert c.ev_r == pytest.approx(c.p * 2.0 - (1 - c.p) * 0.6, abs=1e-3)
    assert c.p_low < c.p
    few = TradeProbabilityModel.fit(trades(100, lambda p: 0.9, seed=5))
    assert few.loss_r == 1.0                                  # fewer than 20 losing trades: nothing measured


def test_reliability_table_and_serialisation():
    ts = trades(300, lambda p: 0.5 * p, seed=6)
    m = TradeProbabilityModel.fit(ts)
    rows = m.reliability(ts)
    assert sum(r["n"] for r in rows) == 300
    json.dumps(m.to_dict())


def _closed(led, i, qty, close_qtys, reasons, rs, p_target=0.5, book="ensemble", t=None):
    t = t or (T0 + i * H)
    pl = plan(decision_id=f"2026-03-{1 + i % 28:02d}-{i:04d}", t=t, p_target=p_target, book=book)
    led.append(pl)
    led.append(Fill(pl.decision_id, pl.decision_id + "-e", t.isoformat(), pl.symbol, 1, qty, 100.0, 0.0, 0.0, book))
    for j, (q, why, r) in enumerate(zip(close_qtys, reasons, rs)):
        led.append(Close(pl.decision_id, (t + (j + 1) * H).isoformat(), pl.symbol, 101.0, q, 1.0, r, why, book))
    return pl


def test_forward_trades_come_from_complete_ensemble_trades_only():
    led = Ledger(":memory:", git_commit="t")
    _closed(led, 0, 100, [100], ["target"], [2.0])
    _closed(led, 1, 100, [100], ["stop"], [-1.0])
    _closed(led, 2, 100, [50], ["target"], [1.0])                          # half still open: not yet a result
    _closed(led, 3, 100, [50, 50], ["target", "stop"], [1.0, 0.0])         # T1 then breakeven: counts, reached target
    _closed(led, 4, 100, [100], ["target"], [2.0], book="virtual:x")      # other books are not forward results
    led.append(Counterfactual("2026-03-01-0000", T0.isoformat(), "calendar", T0.isoformat(), "target", 2.0))
    got = {t.decision_id[-4:]: t for t in forward_trades(led)}
    assert set(got) == {"0000", "0001", "0003"}
    assert (got["0000"].y, got["0001"].y, got["0003"].y) == (1, 0, 1)
    assert got["0003"].r == 1.0
    assert {t.y for t in forward_trades(led, label="profit")} == {1, 0}
    until = (T0 + 1.5 * H).isoformat()
    assert [t.decision_id[-4:] for t in forward_trades(led, until=until)] == ["0000"]
    assert fit_from_ledger(led).n == 3 and not fit_from_ledger(led).calibrated
