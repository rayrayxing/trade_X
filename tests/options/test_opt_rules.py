import datetime as dt

import pandas as pd
import pytest

from opt_fixtures import FakeRisk
from tradex.execution.checks import ShortInfo
from tradex.options.providers import EarningsInfo
from tradex.options.rules import NakedCallPolicy, blocker_kind, naked_call_blockers, stop_problem

ASOF = pd.Timestamp("2026-10-05 14:00", tz="UTC")           # Monday 10:00 New York
POL = NakedCallPolicy()
CALM = ShortInfo(short_interest_pct_float=3.0, days_to_cover=1.0)


def test_policy_defaults_match_ray_brief():
    assert POL.gap_pct == 0.20 and POL.stop_required and POL.block_through_earnings
    assert POL.max_short_interest_pct_float == 20.0 and POL.max_days_to_cover == 5.0 and POL.require_known_data


def test_policy_from_repo_policy_file_reads_the_squeeze_limits():
    pol = NakedCallPolicy.from_policy()
    assert pol.max_short_interest_pct_float == 20.0 and pol.max_days_to_cover == 5.0 and pol.gap_pct >= 0.20


def test_policy_file_can_tighten_but_gap_never_drops_below_floor(tmp_path):
    f = tmp_path / "p.yaml"
    f.write_text("shorts: {max_short_interest_pct_float: 15, max_days_to_cover: 4}\n"
                 "options:\n  naked_call: {gap_pct: 0.10, earnings_buffer_days: 3, max_days_to_cover: 3}\n")
    pol = NakedCallPolicy.from_policy(f)
    assert pol.max_short_interest_pct_float == 15 and pol.max_days_to_cover == 3
    assert pol.earnings_buffer_days == 3 and pol.gap_pct == 0.20


def test_stop_problem():
    assert stop_problem(None, 1.5, POL) == "naked call has no buy-to-close stop"
    assert stop_problem(0.0, 1.5, POL)
    assert "not above the credit" in stop_problem(1.5, 1.5, POL)
    assert stop_problem(3.0, 1.5, POL) is None
    assert stop_problem(None, 1.5, NakedCallPolicy(stop_required=False)) is None


def test_clear_name_passes():
    assert naked_call_blockers(POL, "X", ASOF, dt.date(2026, 10, 16), FakeRisk(EarningsInfo(None), CALM)) == []


def test_earnings_inside_window_blocks_including_buffer():
    horizon = dt.date(2026, 10, 16)
    for d in (dt.date(2026, 10, 5), dt.date(2026, 10, 10), horizon, dt.date(2026, 10, 17)):
        r = naked_call_blockers(POL, "X", ASOF, horizon, FakeRisk(EarningsInfo(d), CALM))
        assert r and r[0].startswith("earnings:"), d
    for d in (dt.date(2026, 10, 18), dt.date(2026, 11, 20)):
        assert naked_call_blockers(POL, "X", ASOF, horizon, FakeRisk(EarningsInfo(d), CALM)) == []


def test_earnings_already_passed_does_not_block():
    assert naked_call_blockers(POL, "X", ASOF, dt.date(2026, 10, 16), FakeRisk(EarningsInfo(dt.date(2026, 10, 1)), CALM)) == []


def test_unknown_earnings_blocks_unless_policy_allows():
    assert naked_call_blockers(POL, "X", ASOF, dt.date(2026, 10, 16), FakeRisk(None, CALM)) == ["unknown_data: X: earnings date unknown"]
    lax = NakedCallPolicy(require_known_data=False)
    assert naked_call_blockers(lax, "X", ASOF, dt.date(2026, 10, 16), FakeRisk(None, CALM)) == []


def test_earnings_rule_can_be_disabled_only_by_policy_object():
    pol = NakedCallPolicy(block_through_earnings=False)
    r = FakeRisk(EarningsInfo(dt.date(2026, 10, 7)), CALM)
    assert naked_call_blockers(pol, "X", ASOF, dt.date(2026, 10, 16), r) == []


def test_heavily_shorted_names_blocked():
    h = dt.date(2026, 10, 16)
    r = naked_call_blockers(POL, "X", ASOF, h, FakeRisk(EarningsInfo(None), ShortInfo(short_interest_pct_float=25.0)))
    assert len(r) == 1 and r[0].startswith("squeeze:") and "25%" in r[0]
    r = naked_call_blockers(POL, "X", ASOF, h, FakeRisk(EarningsInfo(None), ShortInfo(days_to_cover=6.5)))
    assert len(r) == 1 and "days to cover" in r[0]
    both = naked_call_blockers(POL, "X", ASOF, h, FakeRisk(EarningsInfo(None), ShortInfo(short_interest_pct_float=30, days_to_cover=9)))
    assert len(both) == 2


def test_squeeze_limits_are_inclusive_at_the_limit():
    ok = ShortInfo(short_interest_pct_float=20.0, days_to_cover=5.0)
    assert naked_call_blockers(POL, "X", ASOF, dt.date(2026, 10, 16), FakeRisk(EarningsInfo(None), ok)) == []


def test_unknown_short_interest_blocks():
    h = dt.date(2026, 10, 16)
    for si in (None, ShortInfo()):
        r = naked_call_blockers(POL, "X", ASOF, h, FakeRisk(EarningsInfo(None), si))
        assert r == ["unknown_data: X: short-interest data unknown"]


def test_no_provider_blocks():
    r = naked_call_blockers(POL, "X", ASOF, dt.date(2026, 10, 16), None)
    assert len(r) == 1 and r[0].startswith("unknown_data:")


def test_date_is_read_in_new_york_time():
    # 02:00 UTC on the 6th is still the evening of the 5th in New York
    asof = pd.Timestamp("2026-10-06 02:00", tz="UTC")
    r = FakeRisk(EarningsInfo(dt.date(2026, 10, 5)), CALM)
    assert naked_call_blockers(POL, "X", asof, dt.date(2026, 10, 16), r)


def test_blocker_kind_priority():
    assert blocker_kind([]) == ""
    assert blocker_kind(["unknown_data: x", "squeeze: y"]) == "squeeze"
    assert blocker_kind(["squeeze: y", "earnings: z"]) == "earnings"
    assert blocker_kind(["unknown_data: x"]) == "unknown_data"
