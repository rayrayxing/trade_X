import numpy as np
import pandas as pd
import pytest
import yaml

from tradex.research.loop import health
from tradex.research.loop.specstore import SpecExists, YamlSpecStore, content_hash
from tradex.strategy.spec import StrategySpec

from loopkit import oos_returns, spec_dict, write_spec


# --- spec store -------------------------------------------------------------------------------------

def test_hash_ignores_status_stats_and_provenance_but_not_the_strategy():
    a = spec_dict()
    b = spec_dict(status="paper", stats={"hit_rate": 0.9}, provenance={"source": "elsewhere"})
    assert content_hash(a) == content_hash(b)
    assert content_hash(a) != content_hash(spec_dict(entry={"long": "close > 0"}))
    assert content_hash(a) != content_hash(spec_dict(version=2))


def test_write_new_creates_a_file_that_tradex_check_accepts(tmp_path):
    store = YamlSpecStore(tmp_path / "proposed")
    path = store.write_new(spec_dict("stk-new-one"))
    assert path.endswith("stk-new-one.yaml")
    assert StrategySpec.load(path).validate() == []
    assert store.get("stk-new-one").status == "proposed"


def test_write_new_refuses_overwrites_and_unsafe_ids(tmp_path):
    store = YamlSpecStore(tmp_path / "proposed", tmp_path / "seeds")
    write_spec(tmp_path / "seeds", spec_dict("stk-old"))
    with pytest.raises(SpecExists):
        store.write_new(spec_dict("stk-old"))                      # exists in another folder
    store.write_new(spec_dict("stk-fresh"))
    with pytest.raises(SpecExists):
        store.write_new(spec_dict("stk-fresh"))
    for bad in ("../evil", "Has Caps", "a", "x/y", ""):
        with pytest.raises(ValueError):
            store.write_new(spec_dict(bad))


def test_set_status_edits_only_the_status_line_and_keeps_comments(tmp_path):
    p = tmp_path / "proposed" / "stk-c.yaml"
    p.parent.mkdir()
    p.write_text("# hand written, keep this comment\nid: stk-c\nversion: 1\nasset_class: stocks\nuniverse: [A]\n"
                 "timeframes: {signal: D1}\nstatus: proposed   # trailing note\nentry: {long: 'close > 0'}\n")
    store = YamlSpecStore(p.parent)
    store.set_status("stk-c", "screened")
    text = p.read_text()
    assert text.startswith("# hand written, keep this comment\n") and "status: screened" in text
    assert "version: 1" in text and "entry: {long: 'close > 0'}" in text
    assert store.get("stk-c").status == "screened"


def test_set_status_appends_when_there_is_no_status_line_and_rejects_unknown_ids(tmp_path):
    p = tmp_path / "s" / "stk-d.yaml"
    p.parent.mkdir()
    p.write_text("id: stk-d\nversion: 1\n")
    store = YamlSpecStore(p.parent)
    store.set_status("stk-d", "rejected")
    assert yaml.safe_load(p.read_text()) == {"id": "stk-d", "version": 1, "status": "rejected"}
    with pytest.raises(KeyError):
        store.set_status("nope", "paper")


def test_set_status_does_not_change_the_content_hash(tmp_path):
    store = YamlSpecStore(tmp_path / "p")
    store.write_new(spec_dict("stk-e"))
    h = store.content_hash("stk-e")
    store.set_status("stk-e", "paper")
    assert store.content_hash("stk-e") == h


def test_list_specs_walks_every_folder_and_skips_non_specs(tmp_path):
    store = YamlSpecStore(tmp_path / "a", tmp_path / "b")
    write_spec(tmp_path / "a", spec_dict("stk-one"))
    write_spec(tmp_path / "b", spec_dict("stk-two", version=3))
    (tmp_path / "b" / "notes.yaml").write_text("just: notes\n")
    got = {(r.spec_id, r.version) for r in store.list_specs()}
    assert got == {("stk-one", 1), ("stk-two", 3)}


# --- health -----------------------------------------------------------------------------------------

def test_cusum_is_zero_on_returns_at_the_reference_and_falls_on_a_run_below_it():
    flat = health.lower_cusum([0.001] * 10, mu0=0.001, sigma0=0.01, k=0.5)
    assert flat.max() == 0.0 and (flat == 0).all()                      # z=0, +k keeps it clipped at 0
    down = health.lower_cusum([-0.019] * 6, mu0=0.001, sigma0=0.01, k=0.5)   # z = -2 each day: S falls 1.5 per day
    assert np.allclose(down, [-1.5, -3.0, -4.5, -6.0, -7.5, -9.0])
    rec = health.lower_cusum([-0.019, 0.05, 0.05], mu0=0.001, sigma0=0.01, k=0.5)
    assert rec[0] == -1.5 and rec[1] == 0.0                              # it never goes above zero


def test_cusum_needs_a_positive_sigma():
    with pytest.raises(ValueError):
        health.lower_cusum([0.0], 0.0, 0.0, 0.5)


def test_baseline_from_returns_has_mean_std_and_drawdown():
    r = pd.Series([0.1, -0.2, 0.05, 0.05])
    b = health.baseline_from_returns(r)
    assert b.n == 4 and b.mean == pytest.approx(0.0) and b.std == pytest.approx(r.std(ddof=1))
    assert b.max_drawdown == pytest.approx(-0.2)


def assess(r, base, **kw):
    args = dict(k=0.5, h=8.0, warn_frac=0.5, mean_shrink=0.5, dd_multiple=1.5) | kw
    return health.assess(r, base, **args)


def test_a_healthy_strategy_is_left_alone_over_many_seeds():
    base = health.baseline_from_returns(oos_returns(500, mean=0.0008, std=0.01, seed=1))
    alarms = 0
    for seed in range(40):
        live = pd.Series(np.random.default_rng(100 + seed).normal(0.0008, 0.01, 120))
        alarms += assess(live, base).cusum_alarm
    assert alarms <= 2                                                   # false alarm rate stays small


def test_a_strategy_that_stopped_working_alarms():
    base = health.baseline_from_returns(oos_returns(500, mean=0.0015, std=0.01, seed=2))
    live = pd.Series(np.random.default_rng(7).normal(-0.004, 0.01, 60))
    h = assess(live, base)
    assert h.cusum_alarm and h.alarm and h.cusum_min <= -8 and h.n == 60


def test_warning_comes_before_the_alarm():
    base = health.Baseline(mean=0.0, std=0.01, n=250, max_drawdown=-0.5)
    live = pd.Series([-0.0125] * 30)                                      # z=-1.25 + k=.5 -> -0.75/day
    h = assess(live, base, h=40.0)
    assert h.cusum_warn and not h.cusum_alarm
    h = assess(live, base, h=1000.0)
    assert not h.cusum_warn


def test_drawdown_alarm_uses_the_baselines_multiple():
    base = health.Baseline(mean=0.0, std=0.05, n=250, max_drawdown=-0.10)
    ok = assess(pd.Series([-0.01] * 12), base, h=1e9)                     # about -11%: inside 1.5 x 10%
    bad = assess(pd.Series([-0.02] * 10), base, h=1e9)                    # about -18%: beyond -15%
    assert not ok.drawdown_alarm and bad.drawdown_alarm and bad.alarm


def test_no_drawdown_alarm_when_the_baseline_had_none():
    base = health.Baseline(mean=0.0, std=0.01, n=10, max_drawdown=0.0)
    assert not assess(pd.Series([-0.05] * 5), base, h=1e9).drawdown_alarm
