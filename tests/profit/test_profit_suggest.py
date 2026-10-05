import ast
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from pfx import plan
from tradex.profit.calibration import CalibratedProbability
from tradex.profit.suggest import (OpenTrade, PyramidPolicy, SizingSuggestion, VolTargetPolicy, cap_to_suggestion,
                                   pyramid_plan, pyramid_suggestion, realised_vol, vol_target_suggestion)


def trade(**kw):
    base = dict(direction=1, entry_price=100.0, initial_stop=98.0, stop=100.0, qty=1000.0, initial_qty=1000.0,
                price=102.5, adds_done=0, atr=1.0, entry_atr=1.0)
    return OpenTrade(**(base | kw))


def worst_case_loss(t, q, add_price=None):
    """Independent re-derivation: loss in USD if everything is stopped at the shared stop."""
    px = t.price if add_price is None else add_price
    d = t.direction
    return -(t.qty * d * (t.stop - t.entry_price) + q * d * (t.stop - px)) * t.base_to_usd


# --- pyramiding ---------------------------------------------------------------------------------------

def test_no_add_before_the_trade_is_far_enough_in_profit():
    s = pyramid_suggestion(trade(price=100.8))
    assert not s.ok and s.qty == 0 and "needs" in s.blockers[0]


def test_no_add_until_the_stop_is_at_breakeven():
    s = pyramid_suggestion(trade(stop=98.5))
    assert not s.ok and any("breakeven" in b for b in s.blockers)
    assert pyramid_suggestion(trade(stop=98.5), PyramidPolicy(require_breakeven_stop=False)).qty >= 0


def test_no_add_into_a_volatility_spike_or_after_the_last_add():
    assert any("volatility" in b for b in pyramid_suggestion(trade(atr=2.5)).blockers)
    assert not pyramid_suggestion(trade(adds_done=2)).ok
    assert pyramid_suggestion(trade(adds_done=1, price=104.5)).ok          # the second add needs 2R


def test_add_is_sized_so_the_whole_position_cannot_lose_more_than_the_budget():
    t = trade()
    s = pyramid_suggestion(t)
    init_risk = 2.0 * 1000.0
    # add risk fraction 0.5: the add alone risks at most 1000 USD to the shared stop (unit risk 2.5)
    assert s.qty == pytest.approx(0.5 * init_risk / 2.5)
    assert worst_case_loss(t, s.qty) <= 1.0 * init_risk + 1e-9
    assert s.risk_usd == pytest.approx(s.qty * 2.5, abs=0.01) and s.authority == "suggestion"


def test_locked_profit_buys_a_bigger_add_but_never_a_bigger_worst_case():
    low = pyramid_suggestion(trade(stop=100.0, price=102.5), PyramidPolicy(add_risk_fraction=5.0))
    high = pyramid_suggestion(trade(stop=101.0, price=102.5), PyramidPolicy(add_risk_fraction=5.0))
    assert high.qty > low.qty
    for t, s in ((trade(stop=100.0), low), (trade(stop=101.0), high)):
        assert worst_case_loss(t, s.qty) <= 2000.0 + 1e-6


@settings(max_examples=120, deadline=None)
@given(d=st.sampled_from([1, -1]), stop_r=st.floats(0.0, 1.5), price_r=st.floats(1.0, 4.0), q0=st.floats(1, 1e5),
       frac=st.floats(0.1, 1.0), total=st.floats(0.2, 1.5), bu=st.floats(0.5, 2.0))
def test_worst_case_never_exceeds_the_budget(d, stop_r, price_r, q0, frac, total, bu):
    e, risk = 100.0, 2.0
    t = OpenTrade(d, e, e - d * risk, e + d * stop_r * risk, q0, q0, e + d * price_r * risk, 0, 1.0, 1.0, bu)
    pol = PyramidPolicy(add_risk_fraction=frac, max_total_risk_fraction=total, add_at_r=(1.0,), max_adds=1)
    s = pyramid_suggestion(t, pol)
    init = risk * q0 * bu
    if s.ok:
        assert worst_case_loss(t, s.qty) <= total * init * (1 + 1e-9) + 1e-6
        assert s.qty * abs(t.price - t.stop) * bu <= frac * init * (1 + 1e-9) + 1e-6
    else:
        assert s.qty == 0


def test_short_pyramid_mirrors_long():
    a = pyramid_suggestion(trade())
    b = pyramid_suggestion(OpenTrade(-1, -100.0, -98.0, -100.0, 1000.0, 1000.0, -102.5, 0, 1.0, 1.0))
    assert a.qty == pytest.approx(b.qty)


def test_the_add_on_plan_shares_the_stop_and_only_keeps_targets_ahead():
    base = plan(entry=100.0, stop=98.0, targets=(104.0, 108.0), cost_r=0.05)
    t = trade(price=104.5)
    s = pyramid_suggestion(t)
    add = pyramid_plan(base, t, s, "2026-03-03-0009", "2026-03-03T10:00:00+00:00")
    assert add.stop == 100.0 and add.entry_price == 104.5 and add.targets == [108.0]
    assert add.decision_id == "2026-03-03-0009" and add.reward_risk > 0
    done = pyramid_plan(plan(targets=(104.0,)), t, s, "x", "t")
    assert done is None                                                  # target 1 is behind price
    assert pyramid_plan(base, t, pyramid_suggestion(trade(price=100.5)), "x", "t") is None


def test_cap_to_suggestion_only_lowers():
    s = SizingSuggestion("pyramid", qty=300.0)
    assert cap_to_suggestion(1000.0, s) == 300.0 and cap_to_suggestion(100.0, s) == 100.0


# --- vol targeting -------------------------------------------------------------------------------------

def curve(n, daily_vol, seed=0, start=10_000.0):
    r = np.random.default_rng(seed).normal(0, daily_vol, n)
    return start * np.cumprod(1 + r)


def test_realised_vol_annualises_daily_returns():
    v, n = realised_vol(curve(200, 0.01, seed=1), 120)
    assert n == 120 and v == pytest.approx(0.01 * np.sqrt(252), rel=0.25)


def test_too_little_history_means_no_vol_targeting():
    s = vol_target_suggestion(curve(10, 0.01), 1.0)
    assert not s.ok and s.scale == 1.0 and s.risk_pct == 1.0 and "need 30" in s.blockers[0]


def test_high_vol_shrinks_and_the_shrink_factor_is_usable_as_is():
    s = vol_target_suggestion(curve(120, 0.03, seed=2), 1.0)
    assert s.scale < 1.0 and s.shrink_factor == s.scale and s.risk_pct == pytest.approx(s.scale)
    assert s.scale >= VolTargetPolicy().min_scale * 0.4                       # the drawdown de-risk can go lower


def test_low_vol_does_not_upsize_without_a_calibrated_edge():
    c = curve(120, 0.002, seed=3)
    s = vol_target_suggestion(c, 1.0)
    assert s.scale <= 1.0 and any("no calibrated edge" in r for r in s.reasons)
    unc = CalibratedProbability(0.55, 0.0, 0.55, False, "base_rate", 10, None)
    assert vol_target_suggestion(c, 1.0, edge=unc).scale <= 1.0


def test_calibrated_edge_allows_upsizing_up_to_fractional_kelly_on_the_lower_bound():
    c = curve(120, 0.002, seed=3)
    edge = CalibratedProbability(0.5, 0.45, 0.55, True, "platt", 200, 0.5)
    s = vol_target_suggestion(c, 1.0, edge=edge, reward_risk=2.0)
    assert 1.0 < s.scale <= VolTargetPolicy().max_scale
    thin = CalibratedProbability(0.5, 0.34, 0.55, True, "platt", 120, 0.2)        # lower bound near break-even
    assert vol_target_suggestion(c, 1.0, edge=thin, reward_risk=2.0).scale <= 1.0 + 1e-9
    assert s.shrink_factor == 1.0                                                  # the gate takes shrink only, today


def test_drawdown_derisks():
    c = curve(120, 0.01, seed=4)
    flat = vol_target_suggestion(c, 1.0, peak=float(c[-1]))
    deep = vol_target_suggestion(c, 1.0, peak=float(c[-1]) / 0.8)
    assert deep.scale < flat.scale and any("drawdown" in r for r in deep.reasons)


def test_zero_vol_is_not_divided_by():
    s = vol_target_suggestion([100.0] * 60, 1.0)
    assert not s.ok and s.scale == 1.0


# --- the module cannot size or send anything -----------------------------------------------------------

def test_profit_package_never_imports_the_gate_or_execution():
    root = Path(__file__).resolve().parents[2] / "tradex" / "profit"
    banned = ("tradex.execution", "tradex.risk.gate")
    seen = []
    for f in root.glob("*.py"):
        for node in ast.walk(ast.parse(f.read_text())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                [node.module or ""] if isinstance(node, ast.ImportFrom) else []
            for n in names:
                seen.append((f.name, n))
                assert not n.startswith(banned), f"{f.name} imports {n}"
    assert any(n == "tradex.risk.sizing" for _, n in seen)               # the scan really read the imports
