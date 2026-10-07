"""Decision IDs after a restart: the per-day counter continues from the ledger, never reuses an ID."""
from collections import Counter

import pandas as pd

from test_spine import _frames, _measured, _two_family_specs
from tradex.core.ledger import Ledger
from tradex.core.loop import CoreConfig
from tradex.core.records import DecisionIds, Veto
from tradex.core.replay import run_replay


def _plans(led):
    return [r["decision_id"] for r in led.rows(kind="plan")]


def test_same_day_restart_on_the_same_ledger_continues_numbering(tmp_path):
    frames = _frames()
    start = frames["AAA"].index[260]
    end = frames["AAA"].index[261]                          # the one close of 2019-01-01 (daily bars)
    day = end
    path = tmp_path / "live.sqlite"
    first = Ledger(path, git_commit="t")
    run_replay(_measured(_two_family_specs()), frames, first, start, end, cfg=CoreConfig(mode="paper"), use_es=False)
    before = _plans(first)
    first.close()
    assert before and all(d.startswith(day.strftime("%Y-%m-%d")) for d in before)

    again = Ledger(path, git_commit="t")                     # a restart: new process, same ledger file
    run_replay(_measured(_two_family_specs()), frames, again, start, end, cfg=CoreConfig(mode="paper"), use_es=False)
    after = _plans(again)
    assert len(after) == 2 * len(before)
    assert not [d for d, n in Counter(after).items() if n > 1]           # no ID used twice
    new = after[len(before):]
    assert int(new[0][-4:]) == max(int(d[-4:]) for d in before) + 1      # numbering continues
    assert again.verify() == (True, None)


def test_replay_keeps_its_own_count_and_other_days_start_at_one():
    led = Ledger(":memory:", git_commit="t")
    led.append(Veto("2026-10-05-0007", "2026-10-05T10:00:00+00:00", "x", "y"))
    led.append(Veto("oanda-trade-12", "2026-10-05T10:00:00+00:00", "x", "y"))       # not a core ID: ignored
    ids = DecisionIds.from_ledger(led)
    assert ids.next(pd.Timestamp("2026-10-05 11:00", tz="UTC")) == "2026-10-05-0008"
    assert ids.next(pd.Timestamp("2026-10-06 00:00", tz="UTC")) == "2026-10-06-0001"
    assert DecisionIds().next(pd.Timestamp("2026-10-05 11:00", tz="UTC")) == "2026-10-05-0001"
