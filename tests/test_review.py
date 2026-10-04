import pandas as pd

from tradex.positions.review import (ActionKind, MarketSnapshot, OpenPosition, PositionReviewer, ReviewPolicy,
                                     hit_probability)

T0 = pd.Timestamp("2026-01-05 15:00", tz="UTC")


def pos(**kw):
    d = dict(symbol="EUR_USD", asset_class="forex", strategy_id="s", direction=1, qty=10_000, entry_time=T0,
             entry_price=1.10, stop=1.09, target=1.12, initial_risk=0.01, max_bars=48, bars_held=1, best_price=1.10)
    return OpenPosition(**(d | kw))


def snap(hours=1, price=1.10, atr=0.002, **kw):
    return MarketSnapshot(time=T0 + pd.Timedelta(hours=hours), price=price, atr=atr, bar_hours=1, **kw)


def kinds(actions):
    return [a.kind for a in actions]


def test_time_limit():
    a = PositionReviewer().review(pos(bars_held=48), snap(48))
    assert kinds(a) == [ActionKind.CLOSE] and "time limit" in a[0].reason


def test_hard_ceiling_even_with_long_max_bars():
    a = PositionReviewer().review(pos(max_bars=10_000, bars_held=300), snap(24 * 11, price=1.105))
    assert kinds(a) == [ActionKind.CLOSE] and "hard ceiling" in a[0].reason


def test_stale_trade_closed():
    a = PositionReviewer().review(pos(bars_held=30), snap(30, price=1.101))
    assert "stale" in a[0].reason


def test_target_unreachable_near_end():
    a = PositionReviewer().review(pos(bars_held=44), snap(44, price=1.1050, atr=0.001))
    assert a[0].kind == ActionKind.CLOSE and "target unlikely" in a[0].reason


def test_rollover_close_when_charge_exceeds_expected_gain():
    p = pos(bars_held=3)
    s = snap(3, price=1.1005, next_rollover_time=T0 + pd.Timedelta(hours=3, minutes=30), next_rollover_cost_usd=50.0)
    a = PositionReviewer().review(p, s)
    assert a[0].kind == ActionKind.CLOSE and "rollover" in a[0].reason


def test_breakeven_and_trail():
    r = PositionReviewer(ReviewPolicy(breakeven_r=1.0, trail_atr=2.0))
    a = r.review(pos(bars_held=5, best_price=1.115), snap(5, price=1.112))
    move = [x for x in a if x.kind == ActionKind.MOVE_STOP][0]
    assert move.price == max(1.10, 1.115 - 2 * 0.002)


def test_event_inside_window():
    p = pos(asset_class="stocks", symbol="NVDA", entry_price=100, stop=95, target=110, initial_risk=5, best_price=100)
    s = snap(2, price=101, atr=2, next_event_time=T0 + pd.Timedelta(hours=10), next_event_kind="earnings")
    assert "earnings" in PositionReviewer().review(p, s)[0].reason
    a = PositionReviewer(ReviewPolicy(event_action="reduce")).review(p, s)
    assert a[0].kind == ActionKind.REDUCE


def test_hit_probability_monotonic():
    assert hit_probability(0, 1, 1) == 1.0
    assert hit_probability(1, 1, 100) > hit_probability(1, 1, 4) > hit_probability(5, 1, 4)
