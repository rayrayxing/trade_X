import numpy as np
import pandas as pd
import pytest

from pfx import H, T0, plan, with_history
from test_profit_costcal import stock_ledger
from tradex.core.ledger import Ledger
from tradex.core.records import EquitySnapshot
from tradex.costs.models import MoomooStockCosts
from tradex.positions.review import Action, ActionKind
from tradex.profit.calibration import CalibrationConfig
from tradex.profit.costcal import CalibratedCosts, CostCalConfig
from tradex.profit.exits import ExitPolicy
from tradex.profit.hooks import ProfitHooks
from tradex.profit.suggest import VolTargetPolicy


def manage(h, bars, entry_t, base=(), **kw):
    args = dict(direction=1, entry_price=100.0, entry_time=entry_t, initial_stop=98.0, stop=98.0,
                targets=[104.0, 108.0], max_bars=10, cost_r=0.0, qty0=100.0, qty=100.0, bars=bars,
                base=list(base) or [Action(ActionKind.HOLD, "within plan")], tf_duration=H)
    return h.manage("ensemble", "d1", **(args | kw))


def test_everything_off_is_a_pass_through():
    h = ProfitHooks()
    pl = plan()
    assert h.annotate_plan(pl) is pl and h.attach_target(pl) == 104.0
    assert h.size_factor(Ledger(":memory:", git_commit="t"), "ensemble", 3.0) == 1.0
    bars, t = with_history([(100, 102.5, 99.9, 102.4)])
    base = [Action(ActionKind.HOLD, "within plan")]
    assert manage(h, bars, t, base) == base


def test_manage_merges_the_reviewer_and_the_engine_into_one_close_and_the_tightest_stop():
    h = ProfitHooks(exit_policy=ExitPolicy(shock_atr_mult=None))
    bars, t = with_history([(100, 102.5, 99.9, 102.4)])                      # +1.2R close: engine moves the stop to entry
    out = manage(h, bars, t, [Action(ActionKind.MOVE_STOP, "reviewer", price=99.0)])
    assert [a.kind for a in out] == [ActionKind.MOVE_STOP] and out[0].price == 100.0
    loose = manage(h, bars, t, [Action(ActionKind.MOVE_STOP, "reviewer", price=101.0)])
    assert loose[0].price == 101.0                                            # the reviewer's was tighter
    closing = manage(h, bars, t, [Action(ActionKind.CLOSE, "stale")])
    assert [a.kind for a in closing] == [ActionKind.CLOSE] and closing[0].reason == "stale"
    already = manage(h, bars, t, stop=100.5)                                  # a stop already tighter than the engine's
    assert all(a.kind != ActionKind.MOVE_STOP for a in already)


def test_engine_close_wins_over_a_reviewer_stop_move():
    h = ProfitHooks(exit_policy=ExitPolicy())
    bars, t = with_history([(100, 100.2, 96.5, 96.8)])                        # volatility shock
    out = manage(h, bars, t, [Action(ActionKind.MOVE_STOP, "reviewer", price=99.0)], stop=90.0)
    assert [a.kind for a in out] == [ActionKind.CLOSE] and "volatility shock" in out[0].reason


def test_partial_target_becomes_a_reduce_of_what_is_open_now_and_is_not_repeated():
    h = ProfitHooks(exit_policy=ExitPolicy(shock_atr_mult=None))
    bars, t = with_history([(100, 104.4, 99.9, 104.0)])
    out = manage(h, bars, t)
    red = [a for a in out if a.kind == ActionKind.REDUCE]
    assert red and red[0].price == 104.0 and red[0].fraction == pytest.approx(0.5)
    h.note_reduce("ensemble", "d1")
    again = manage(h, bars, t, qty=50.0)                                      # half is out: target 2 is next
    assert not [a for a in again if a.kind == ActionKind.REDUCE]
    assert any(a.kind == ActionKind.MOVE_STOP for a in again)                 # breakeven after target 1


def test_vol_factor_shrinks_a_volatile_book_and_needs_history():
    led = Ledger(":memory:", git_commit="t")
    rng = np.random.default_rng(3)
    eq = 10_000.0
    h = ProfitHooks(vol_policy=VolTargetPolicy(target_annual_vol=0.05, min_obs=30))
    assert h.size_factor(led, "ensemble", 3.0) == 1.0                         # no snapshots: no vol targeting
    for i in range(80):
        eq *= 1 + rng.normal(0, 0.02)
        led.append(EquitySnapshot((T0 + pd.Timedelta(days=i)).isoformat(), "ensemble", eq, eq, 0.0, [], {}, {}, "h"))
    f = h.size_factor(led, "ensemble", 3.0)
    assert 0.0 < f < 1.0
    assert h.size_factor(led, "virtual:x", 3.0) == 1.0                        # another book's curve is not read


def test_daily_refits_the_probability_model_and_swaps_calibrated_costs_in_place():
    led = stock_ledger(60, ss_mult=2.0)
    costs = {"stocks": MoomooStockCosts()}
    base = costs["stocks"]
    h = ProfitHooks(calibration=CalibrationConfig(min_trades=20), cost_cfg=CostCalConfig(min_fills=20))
    t = T0 + 100 * H
    h.daily(led, costs, None, t)
    assert h.prob_model is not None and h.prob_model.n == 0                   # no closed trades in this ledger
    assert isinstance(costs["stocks"], CalibratedCosts) and costs["stocks"].spread_mult > 1.2
    first = costs["stocks"].spread_mult
    h.daily(led, costs, None, t)
    assert costs["stocks"].base is base and costs["stocks"].spread_mult == pytest.approx(first)
    # as of an earlier time only the earlier fills are visible
    h2 = ProfitHooks(cost_cfg=CostCalConfig(min_fills=20))
    c2 = {"stocks": MoomooStockCosts()}
    h2.daily(led, c2, None, T0 + 10 * H)
    assert not isinstance(c2["stocks"], CalibratedCosts) and h2.last_cost_report["stocks"]["n_fills"] == 11


def test_annotate_writes_the_exit_ladder_into_the_plan_and_touches_no_price():
    h = ProfitHooks(exit_policy=ExitPolicy())
    pl = plan()
    out = h.annotate_plan(pl)
    assert "managed by: stop 98; T1 104 (50%); T2 108 (50%)" in out.invalidation
    assert (out.entry_price, out.stop, out.targets, out.max_bars) == (pl.entry_price, pl.stop, pl.targets, pl.max_bars)
