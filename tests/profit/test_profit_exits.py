import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from pfx import H, T0, bars_from, history, mirror, plan, random_walk_bars, with_history
from tradex.core.counterfactual import CounterfactualTracker
from tradex.core.ledger import Ledger
from tradex.core.records import Fill, Veto, Verdict
from tradex.positions.review import ActionKind
from tradex.profit.exits import (ExitPolicy, ExitState, atr_series, compare_policies, evaluate_exit,
                                 evaluate_on_ledger, exit_ladder, load_exit_policy, plan_outcome, simulate_exit)

NO_VOL = dict(shock_atr_mult=None)


def state_for(bars_entry_t, **kw):
    base = dict(direction=1, entry_price=100.0, initial_stop=98.0, stop=98.0, targets=(104.0, 108.0),
                entry_time=bars_entry_t, max_bars=10)
    return ExitState(**(base | kw))


# --- policy and config ------------------------------------------------------------------------------

def test_shipped_config_is_the_default_policy():
    assert load_exit_policy("config/exits.yaml") == ExitPolicy()


def test_policy_validation():
    with pytest.raises(ValueError):
        ExitPolicy(trail="loose")
    with pytest.raises(ValueError):
        ExitPolicy(partials=(0.6, 0.5))
    with pytest.raises(ValueError):
        ExitPolicy.from_dict({"nonsense": 1})
    assert ExitPolicy.from_dict({"partials": [0.25, 0.25]}).fractions(3) == (0.25, 0.25, 0.5)
    assert ExitPolicy().fractions(1) == (1.0,)           # one target: exits whole there
    assert ExitPolicy.baseline().fractions(2) == (1.0,)[:0] + (1.0,)


def test_ladder_text_names_every_level():
    lad = exit_ladder(plan(), ExitPolicy())
    t = lad.text()
    assert "T1 104 (50%)" in t and "T2 108 (50%)" in t and "breakeven at 1R" in t and "time stop 10 bars" in t


# --- evaluate_exit ------------------------------------------------------------------------------------

def test_breakeven_moves_stop_after_a_close_at_1r_and_covers_cost():
    bars, t = with_history([(100, 102.2, 99.8, 102.1)])
    st_ = state_for(t, cost_r=0.1)
    acts = evaluate_exit(ExitPolicy(trail="none", **NO_VOL), st_, bars)
    mv = [a for a in acts if a.kind == ActionKind.MOVE_STOP]
    assert len(mv) == 1 and mv[0].price == pytest.approx(100.0 + 0.1 * 2.0)


def test_no_breakeven_below_threshold_and_hold_is_returned():
    bars, t = with_history([(100, 101.0, 99.8, 100.9)])
    acts = evaluate_exit(ExitPolicy(trail="none", **NO_VOL), state_for(t), bars)
    assert [a.kind for a in acts] == [ActionKind.HOLD]


def test_atr_trail_arms_late_and_trails_the_best_price():
    # ATR 1.0 in history. Best high 106 -> 3 ATR trail = 103; best close 105.5 = 2.75R >= 1.5R
    bars, t = with_history([(100, 103, 99.9, 102.9), (103, 106, 102.8, 105.5)])
    pol = ExitPolicy(partials=(), **NO_VOL)
    acts = evaluate_exit(pol, state_for(t, targets=(120.0,)), bars)
    stop = [a.price for a in acts if a.kind == ActionKind.MOVE_STOP][0]
    assert stop == pytest.approx(106 - 3 * float(atr_series(bars).iloc[-1]))
    assert stop > 100.0


def test_structure_trail_uses_the_lowest_low_of_the_last_bars():
    rows = [(100, 102, 99.9, 101.9), (101.9, 104, 101.0, 103.9), (103.9, 106, 103.2, 105.8)]
    bars, t = with_history(rows)
    pol = ExitPolicy(trail="structure", structure_lookback=2, breakeven_r=None, **NO_VOL)
    acts = evaluate_exit(pol, state_for(t, targets=(120.0,)), bars)
    stop = [a.price for a in acts if a.kind == ActionKind.MOVE_STOP][0]
    atr = float(atr_series(bars).iloc[-1])
    assert stop == pytest.approx(101.0 - 0.1 * atr)       # lowest low of the last 2 bars, less the buffer


def test_stop_never_loosens_and_never_sits_on_the_last_close():
    bars, t = with_history([(100, 103, 99.9, 102.9), (103, 106, 102.8, 105.5)])
    # a stop already above any trail candidate is left alone
    acts = evaluate_exit(ExitPolicy(**NO_VOL), state_for(t, stop=105.0, targets=(120.0,)), bars)
    assert all(a.kind != ActionKind.MOVE_STOP for a in acts)
    # a candidate closer than min_stop_gap_atr to the close is refused
    acts = evaluate_exit(ExitPolicy(trail_atr_mult=0.1, **NO_VOL), state_for(t, targets=(120.0,)), bars)
    stops = [a.price for a in acts if a.kind == ActionKind.MOVE_STOP]
    assert all(105.5 - s >= 0.25 for s in stops)


def test_time_stop_closes_at_the_bar_after_max_bars():
    rows = [(100, 100.4, 99.6, 100.0)] * 3
    bars, t = with_history(rows)
    assert evaluate_exit(ExitPolicy(**NO_VOL), state_for(t, max_bars=4), bars)[0].kind == ActionKind.HOLD
    acts = evaluate_exit(ExitPolicy(**NO_VOL), state_for(t, max_bars=3), bars)
    assert acts[0].kind == ActionKind.CLOSE and "time stop" in acts[0].reason
    assert evaluate_exit(ExitPolicy(time_stop=False, **NO_VOL), state_for(t, max_bars=3), bars)[0].kind == ActionKind.HOLD


def test_no_progress_stop():
    rows = [(100, 100.4, 99.6, 100.1)] * 5
    bars, t = with_history(rows)
    pol = ExitPolicy(progress_frac=0.5, progress_min_r=0.3, **NO_VOL)
    assert evaluate_exit(pol, state_for(t, max_bars=10), bars)[0].kind == ActionKind.CLOSE
    assert evaluate_exit(pol, state_for(t, max_bars=20), bars)[0].kind == ActionKind.HOLD


def test_volatility_shock_closes_a_position_hit_by_a_wide_adverse_bar():
    bars, t = with_history([(100, 100.4, 99.6, 100.0), (100, 100.2, 96.5, 96.8)])     # range 3.7 ATR, down
    acts = evaluate_exit(ExitPolicy(), state_for(t, stop=90.0), bars)
    assert acts[0].kind == ActionKind.CLOSE and "volatility shock" in acts[0].reason
    up_bar, t2 = with_history([(100, 100.4, 99.6, 100.0), (100, 103.7, 99.9, 103.5)])    # wide, but in favour
    assert evaluate_exit(ExitPolicy(), state_for(t2, stop=90.0, targets=(120.0,)), up_bar)[0].kind != ActionKind.CLOSE


def test_expansion_tightens_the_trail():
    h = history(30, rng=1.0)
    wide = bars_from([(100, 104, 99, 103.8), (103.8, 110, 103, 109.5), (109.5, 113, 108, 112.5)], h.index[-1] + H)
    bars = pd.concat([h, wide])
    t = wide.index[0]
    loose = evaluate_exit(ExitPolicy(shock_atr_mult=None, expand_mult=None), state_for(t, targets=(150.0,), entry_atr=1.0), bars)
    tight = evaluate_exit(ExitPolicy(shock_atr_mult=None, expand_mult=1.5, expand_trail_mult=1.0),
                          state_for(t, targets=(150.0,), entry_atr=1.0), bars)
    s = lambda acts: [a.price for a in acts if a.kind == ActionKind.MOVE_STOP][0]  # noqa: E731
    assert s(tight) > s(loose)


def test_target_reached_without_a_resting_order_asks_for_a_reduce_then_a_close():
    bars, t = with_history([(100, 104.5, 99.9, 104.0)])
    acts = evaluate_exit(ExitPolicy(**NO_VOL), state_for(t), bars)
    red = [a for a in acts if a.kind == ActionKind.REDUCE]
    assert red and red[0].price == 104.0 and red[0].fraction == 0.5
    acts2 = evaluate_exit(ExitPolicy(**NO_VOL), state_for(t, targets_done=1, remaining=0.5), bars)
    assert acts2[0].kind != ActionKind.REDUCE


# --- simulate_exit ------------------------------------------------------------------------------------

def run(rows, policy=None, gap_aware=True, **kw):
    bars, t = with_history(rows)
    return simulate_exit(policy or ExitPolicy(**NO_VOL), state_for(t, **kw), bars, gap_aware=gap_aware)


def test_stop_target_and_costs():
    r = run([(100, 100.5, 97.5, 98.2)], ExitPolicy.baseline())
    assert (r.reason, r.r_multiple) == ("stop", -1.0)
    r = run([(100, 104.2, 99.9, 104.0)], ExitPolicy.baseline(), targets=(104.0,), cost_r=0.1)
    assert r.reason == "target" and r.r_multiple == pytest.approx(2.0 - 0.1)


def test_stop_wins_when_stop_and_target_are_in_one_bar():
    r = run([(100, 105, 97, 101)], ExitPolicy.baseline(), targets=(104.0,))
    assert r.reason == "stop" and r.r_multiple == -1.0


def test_gap_through_the_stop_fills_at_the_open_and_costs_more_than_1r():
    r = run([(96.0, 96.5, 95.5, 96.2)], ExitPolicy.baseline())
    assert r.reason == "stop_gap" and r.r_multiple == pytest.approx(-2.0)
    r2 = run([(96.0, 96.5, 95.5, 96.2)], ExitPolicy.baseline(), gap_aware=False)
    assert r2.r_multiple == -1.0                       # the tracker's convention


def test_gap_through_the_target_fills_at_the_open():
    r = run([(105.0, 105.5, 104.8, 105.2)], ExitPolicy.baseline(), targets=(104.0,))
    assert r.r_multiple == pytest.approx(2.5)


def test_partial_target_then_breakeven_stop_locks_half_the_profit():
    rows = [(100, 104.2, 99.9, 103.5),                  # T1 (104) hit: half off, 2R on that half
            (103.5, 103.8, 99.8, 100.2),                # close moves... stop at breakeven set after bar 1
            (100.2, 100.3, 99.5, 99.8)]                 # breakeven stop (100) hit
    r = run(rows, ExitPolicy(trail="none", **NO_VOL))
    assert [l.reason for l in r.legs] == ["target_1", "stop"]
    assert r.gross_r == pytest.approx(0.5 * 2.0 + 0.5 * 0.0)


def test_partial_then_second_target():
    r = run([(100, 104.2, 99.9, 103.5), (103.5, 108.5, 103.4, 108.0)], ExitPolicy(trail="none", **NO_VOL))
    assert r.gross_r == pytest.approx(0.5 * 2.0 + 0.5 * 4.0) and r.legs[-1].reason == "target"


def test_stop_set_at_a_close_applies_from_the_next_bar_only():
    # bar 1 closes +1.1R (breakeven armed at 100) but its own low was 99.2: the new stop must not act on bar 1
    bars, t = with_history([(100, 102.3, 99.2, 102.2), (102.2, 102.4, 101.0, 101.5), (101.5, 101.8, 101.0, 101.4)])
    r = simulate_exit(ExitPolicy(trail="none", partials=(), **NO_VOL), state_for(t, targets=(110.0,), max_bars=3), bars)
    assert r.reason == "time_stop" and len(r.legs) == 1
    assert r.stop_moves[0][0] == t and r.stop_moves[0][1] == 100.0
    assert r.mae_r < 0                                  # the pierced level was below the new stop


def test_open_at_end_is_flagged_and_marked_to_the_last_close():
    r = run([(100, 100.5, 99.5, 100.4)], ExitPolicy.baseline(), max_bars=50)
    assert r.open_at_end and r.reason == "end_of_data"


def test_extra_leg_cost():
    rows = [(100, 104.2, 99.9, 103.5), (103.5, 108.5, 103.4, 108.0)]
    bars, t = with_history(rows)
    a = simulate_exit(ExitPolicy(trail="none", **NO_VOL), state_for(t), bars)
    b = simulate_exit(ExitPolicy(trail="none", **NO_VOL), state_for(t), bars, extra_leg_cost_r=0.05)
    assert a.r_multiple - b.r_multiple == pytest.approx(0.05)


@pytest.mark.parametrize("seed", range(6))
def test_baseline_policy_reproduces_the_counterfactual_tracker(seed):
    """The exit engine with every rule off IS the plan as the existing tracker follows it."""
    bars = random_walk_bars(400, seed, vol=0.6)
    agree = 0
    for j in range(20, 360, 17):
        entry = float(bars["close"].iloc[j])
        t = bars.index[j] + H
        d = 1 if (j + seed) % 2 else -1
        pl = plan(direction=d, entry=entry, stop=entry - d * 1.5, targets=(entry + d * 3.0,), max_bars=12, cost_r=0.07,
                  t=t, decision_id=f"d{j}")
        tr = CounterfactualTracker(H)
        tr.track(pl, "x")
        got = []
        for k in range(j + 1, len(bars)):
            r = bars.iloc[k]
            got += tr.on_bar(pl.symbol, bars.index[k] + H, float(r["high"]), float(r["low"]), float(r["close"]))
            if got:
                break
        mine = plan_outcome(pl, ExitPolicy.baseline(), bars, gap_aware=False)
        if got:
            assert mine.r_multiple == pytest.approx(got[0].r_multiple, abs=1e-4)
            assert mine.reason.replace("_", "") == got[0].exit_reason.replace("_", "")
            agree += 1
    assert agree > 10


# --- no look-ahead ------------------------------------------------------------------------------------

POLICIES = [ExitPolicy(), ExitPolicy(trail="structure", partials=(0.3,), expand_mult=1.4, progress_frac=0.5),
            ExitPolicy(trail="tighter", breakeven_r=0.5, shock_atr_mult=2.0)]


@pytest.mark.parametrize("pol", POLICIES)
@pytest.mark.parametrize("seed", range(5))
def test_decisions_up_to_a_bar_do_not_depend_on_later_bars(pol, seed):
    bars = random_walk_bars(300, seed, vol=0.7)
    entry_i, cut = 100, 130
    t0 = bars.index[entry_i]
    st0 = state_for(t0, entry_price=float(bars["open"].iloc[entry_i]), initial_stop=float(bars["open"].iloc[entry_i]) - 2,
                    stop=float(bars["open"].iloc[entry_i]) - 2,
                    targets=(float(bars["open"].iloc[entry_i]) + 3, float(bars["open"].iloc[entry_i]) + 6), max_bars=60)
    a = simulate_exit(pol, st0, bars)
    future_scrambled = bars.copy()
    rng = np.random.default_rng(99)
    junk = future_scrambled.iloc[cut + 1:].copy()
    junk[["open", "high", "low", "close"]] = junk[["open", "high", "low", "close"]].to_numpy()[::-1] * rng.uniform(0.5, 1.5)
    future_scrambled.iloc[cut + 1:] = junk.to_numpy()
    b = simulate_exit(pol, st0, future_scrambled)
    cut_t = bars.index[cut]
    assert [m for m in a.stop_moves if m[0] <= cut_t] == [m for m in b.stop_moves if m[0] <= cut_t]
    assert [l for l in a.legs if l.time <= cut_t] == [l for l in b.legs if l.time <= cut_t]
    # and the per-bar decision is a function of the bars up to it
    for k in (110, 120, cut):
        prefix = bars.iloc[: k + 1]
        assert evaluate_exit(pol, st0, prefix) == evaluate_exit(pol, st0, future_scrambled.iloc[: k + 1])


def test_atr_is_causal():
    bars = random_walk_bars(120, 3)
    full = atr_series(bars, 14)
    for k in (30, 60, 90):
        assert atr_series(bars.iloc[: k + 1], 14).iloc[-1] == pytest.approx(full.iloc[k])


@settings(max_examples=60, deadline=None)
@given(seed=st.integers(0, 10_000), d=st.sampled_from([1, -1]), be=st.sampled_from([None, 0.5, 1.0]),
       trail=st.sampled_from(["none", "atr", "structure", "tighter"]))
def test_stop_is_monotone_toward_price_and_r_is_bounded(seed, d, be, trail):
    bars = random_walk_bars(200, seed, vol=0.7)
    i = 80
    e = float(bars["open"].iloc[i])
    pol = ExitPolicy(breakeven_r=be, trail=trail)
    st_ = state_for(bars.index[i], direction=d, entry_price=e, initial_stop=e - d * 2.0, stop=e - d * 2.0,
                    targets=(e + d * 3.0, e + d * 6.0), max_bars=50)
    res = simulate_exit(pol, st_, bars, gap_aware=False)
    stops = [2.0 * -1] + [d * (m[1] - e) for m in res.stop_moves]
    assert all(b > a for a, b in zip(stops, stops[1:]))              # strictly toward price
    assert res.r_multiple >= -1.0 - 1e-9                              # no gaps: never worse than the initial stop


@settings(max_examples=40, deadline=None)
@given(seed=st.integers(0, 10_000))
def test_short_is_the_mirror_of_long(seed):
    bars = random_walk_bars(160, seed, vol=0.7)
    i, e = 60, float(bars["open"].iloc[60])
    long_ = simulate_exit(ExitPolicy(), state_for(bars.index[i], entry_price=e, initial_stop=e - 2, stop=e - 2,
                                                  targets=(e + 3, e + 6), max_bars=40), bars)
    mb = mirror(bars)
    short = simulate_exit(ExitPolicy(), state_for(bars.index[i], direction=-1, entry_price=-e, initial_stop=-e + 2,
                                                  stop=-e + 2, targets=(-e - 3, -e - 6), max_bars=40), mb)
    assert short.r_multiple == pytest.approx(long_.r_multiple, abs=1e-6)
    assert [l.reason for l in short.legs] == [l.reason for l in long_.legs]


# --- ledger-backed evaluation -------------------------------------------------------------------------

def ledger_with_plans(bars, n=30):
    led = Ledger(":memory:", git_commit="t")
    for j in range(40, 40 + 7 * n, 7):
        e = float(bars["close"].iloc[j])
        pl = plan(entry=e, stop=e - 2, targets=(e + 3, e + 6), max_bars=15, t=bars.index[j] + H, decision_id=f"2026-03-{(j % 28) + 1:02d}-{j:04d}")
        led.append(pl)
        if j % 2:
            led.append(Verdict(pl.decision_id, pl.time, "accepted", 1000.0, 20.0, 0.2, [], {}))
            led.append(Fill(pl.decision_id, pl.decision_id + "-e", pl.time, pl.symbol, 1, 1000.0, e, 0.0, 0.0))
        else:
            led.append(Veto(pl.decision_id, pl.time, "calendar", "NFP"))
    return led


def test_compare_policies_on_a_ledger_pairs_each_plan_with_its_baseline():
    bars = random_walk_bars(500, 11, vol=0.6)
    led = ledger_with_plans(bars)
    pols = {"baseline": ExitPolicy.baseline(), "engine": ExitPolicy()}
    out = evaluate_on_ledger(led, lambda s, tf: bars, pols, which="all")
    assert out["evaluated"] + out["skipped_no_bars_or_open"] == 30
    b, e = out["policies"]["baseline"], out["policies"]["engine"]
    assert b["n"] == e["n"] == out["evaluated"] and "delta_mean_r" in e and "delta_mean_r" not in b
    acc = evaluate_on_ledger(led, lambda s, tf: bars, pols, which="accepted")
    blk = evaluate_on_ledger(led, lambda s, tf: bars, pols, which="blocked")
    assert acc["plans"] + blk["plans"] == 30 and acc["plans"] > 0 and blk["plans"] > 0


def test_missing_bars_are_skipped_not_guessed():
    bars = random_walk_bars(300, 2)
    led = ledger_with_plans(bars, n=10)
    out = evaluate_on_ledger(led, lambda s, tf: None, {"baseline": ExitPolicy.baseline()}, which="all")
    assert out["evaluated"] == 0 and out["skipped_no_bars_or_open"] == 10
    assert compare_policies([], lambda s, tf: bars, {"b": ExitPolicy.baseline()})["evaluated"] == 0
