"""Measured hit rates (G7): paper/live vote on a real-data win rate or not at all."""
import pandas as pd
import pytest
import yaml

from conftest import simple_spec
from gapkit import recorded  # noqa: F401  (keeps the gap kit importable the same way as its siblings)
from tradex.decision.ensemble import measured_hit_rate
from tradex.research.loop.specstore import YamlSpecStore
from tradex.research.loop.state import LoopState
from tradex.research.measure import MIN_TRADES, hit_rate_from_evidence, measure, write_stats
from tradex.runtime.build import split_unmeasured


@pytest.mark.parametrize("stats,want", [({}, None), ({"hit_rate": 0.57}, 0.57), ({"hit_rate": "x"}, None),
                                        ({"hit_rate": 0}, None), ({"hit_rate": 1.0}, None), (None, None)])
def test_measured_hit_rate(stats, want):
    assert measured_hit_rate(stats) == want


def test_split_unmeasured_names_the_strategies_without_one():
    a, b = simple_spec(long="close > 0"), simple_spec(long="close > 1")
    b.id = "other"
    a.stats = {"hit_rate": 0.5}
    run, held = split_unmeasured([a, b])
    assert [s.id for s in run] == [a.id] and held == ["other"]


def test_evidence_needs_enough_trades_and_a_win_rate():
    ok = {"win_rate_lower": 0.41, "win_rate": 0.47, "oos_trades": 80, "data_key": "fx-2016"}
    assert hit_rate_from_evidence(ok)["hit_rate"] == 0.41
    assert hit_rate_from_evidence({**ok, "oos_trades": MIN_TRADES - 1}) is None
    assert hit_rate_from_evidence({k: v for k, v in ok.items() if k != "win_rate_lower"}) is None


def test_write_stats_keeps_every_other_line(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text("# hand written\nid: s-one\nversion: 1\nstatus: proposed\nstats:\n  sharpe: 1.1\nuniverse: [A, B]\n")
    stats = {"hit_rate": 0.41, "hit_rate_trades": 80, "hit_rate_source": "walk_forward fx"}
    assert write_stats(p, stats) is True
    text = p.read_text()
    assert text.startswith("# hand written\nid: s-one") and "universe: [A, B]" in text
    assert yaml.safe_load(text)["stats"] == stats
    assert write_stats(p, stats) is False                      # idempotent


def test_write_stats_adds_the_block_when_absent(tmp_path):
    p = tmp_path / "s.yaml"
    p.write_text("id: s-one\nversion: 1\nstatus: proposed\n")
    write_stats(p, {"hit_rate": 0.4})
    assert yaml.safe_load(p.read_text())["stats"] == {"hit_rate": 0.4}


def test_measure_reads_walk_forward_evidence_into_the_spec_files(tmp_path):
    d = tmp_path / "proposed"
    d.mkdir()
    (d / "s-one.yaml").write_text("id: s-one\nversion: 1\nstatus: proposed\n")
    (d / "s-two.yaml").write_text("id: s-two\nversion: 1\nstatus: proposed\n")
    state = LoopState(tmp_path / "loop.sqlite")
    state.add_evidence("s-one", 1, "walk_forward", "r1", False, "h",
                       {"win_rate_lower": 0.43, "win_rate": 0.5, "oos_trades": 120, "data_key": "k"})
    out = measure(state, YamlSpecStore(d))
    assert "43.0%" in out["s-one"] and out["s-two"] == "no walk-forward evidence"
    assert yaml.safe_load((d / "s-one.yaml").read_text())["stats"]["hit_rate"] == 0.43
    assert "stats" not in yaml.safe_load((d / "s-two.yaml").read_text())


def test_paper_build_holds_back_unmeasured_strategies_and_refuses_when_none_is_measured():
    from test_runtime_build import FakeStream, RecordedHistory, START
    from test_runtime_core import _mixed_specs
    from tradex.core.interfaces import ReplayClock
    from tradex.core.ledger import Ledger
    from tradex.data.oanda import QuoteBook
    from tradex.runtime.build import build_runtime
    from tradex.runtime.config import RuntimeConfig

    def build(specs):
        clock = ReplayClock(START)
        return build_runtime("paper", specs, Ledger(":memory:", git_commit="t"), history={"forex": RecordedHistory()},
                             stream=FakeStream(), quotes=QuoteBook(30, clock=clock.now), clock=clock,
                             config=RuntimeConfig(), warmup_bars=400, dry=True)

    h1, h4 = _mixed_specs()
    h4.stats = {}
    note = next(c for c in build([h1, h4]).checks if c.name == "strategies:unmeasured")
    assert note.status == "skip" and "h4-mom" in note.detail and "h1-trend" not in note.detail
    h1.stats = {}
    with pytest.raises(ValueError, match="no measured hit rate for h1-trend, h4-mom"):
        build([h1, h4])
