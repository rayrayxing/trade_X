import pandas as pd
import pytest

from tradex import cli
from tradex.backtest.holdout import HoldoutLocked, HoldoutPolicy, holdout_look, looks, research_view
from conftest import flat_bars


@pytest.fixture
def policy(tmp_path):
    cfg = tmp_path / "holdout.yaml"
    cfg.write_text(f'start: "2024-03-01"\nlooks_ledger: {tmp_path / "looks.jsonl"}\n')
    return HoldoutPolicy.from_config(cfg)


def test_repo_config_loads_and_is_protected():
    p = HoldoutPolicy.from_config()
    assert p.start.tzinfo is not None
    protected = open("config/protected_paths.txt").read().split()
    assert any("config/gates/holdout.yaml".startswith(x) for x in protected)


def test_research_view_hides_bars_that_close_after_start(policy):
    h1 = pd.DataFrame({"close": 1.0}, index=pd.date_range("2024-02-29 20:00", periods=8, freq="h", tz="UTC"))
    v = research_view({"X": h1}, "H1", policy)["X"]
    assert v.index[-1] == pd.Timestamp("2024-02-29 23:00", tz="UTC")    # closes exactly at start
    d1 = flat_bars(60, start="2024-01-02")
    v = research_view({"X": d1}, "D1", policy)["X"]
    assert v.index.max() + pd.Timedelta(days=1) <= policy.start


def test_one_look_per_strategy_version(policy):
    d1 = flat_bars(60, start="2024-01-02")
    got = holdout_look({"X": d1}, "stk-a", 1, "gate 3 check", policy)["X"]
    assert got.index.min() >= policy.start and len(got)
    with pytest.raises(HoldoutLocked):
        holdout_look({"X": d1}, "stk-a", 1, "again", policy)
    holdout_look({"X": d1}, "stk-a", 2, "new version", policy)       # a new version gets its own look
    assert [(lk["strategy_id"], lk["version"]) for lk in looks(policy)] == [("stk-a", "1"), ("stk-a", "2")]


def test_cli_loader_applies_the_holdout(tmp_path, monkeypatch):
    from tradex.data.providers import CsvProvider
    from conftest import simple_spec
    start = HoldoutPolicy.from_config().start
    bars = flat_bars(30, start=(start - pd.Timedelta(days=20)).strftime("%Y-%m-%d"))
    CsvProvider(tmp_path).save("X", "D1", bars)
    out = cli._load_data(simple_spec(), str(tmp_path), None)["X"]
    assert len(out) and out.index.max() + pd.Timedelta(days=1) <= start
