import json

import pytest

from test_spine import _frames, _two_family_specs
from tradex.core.ledger import Ledger
from tradex.core.replay import run_replay
from tradex.data.providers import CsvProvider
from tradex.profit.__main__ import _series_rate_fn, main


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    d = tmp_path_factory.mktemp("profit_cli")
    frames = _frames()
    prov = CsvProvider(d / "bars")
    for sym, df in frames.items():
        prov.save(sym, "D1", df)
    path = d / "replay.sqlite"
    run_replay(_two_family_specs(), frames, Ledger(path, git_commit="t"), frames["AAA"].index[260])
    return d, path


def test_filters_calibrate_costs_and_exits_reports_run_read_only(world, capsys):
    d, path = world
    before = Ledger(path, read_only=True).digest()
    assert main(["filters", "--ledger", str(path)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["gates"] and "selection" in out
    assert main(["calibrate", "--ledger", str(path)]) == 0
    cal = json.loads(capsys.readouterr().out)
    assert cal["method"] in ("base_rate", "platt", "isotonic") and "reliability" in cal
    assert main(["costs", "--ledger", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)[0]["asset_class"] == "stocks"
    assert main(["exits", "--ledger", str(path), "--data", str(d / "bars")]) == 0
    ex = json.loads(capsys.readouterr().out)
    assert ex["policies"]["plain"]["n"] > 0 and "delta_mean_r" in ex["policies"]["engine"]
    assert Ledger(path, read_only=True).digest() == before and Ledger(path, read_only=True).verify()[0]


def test_radar_runs_on_the_latest_snapshot(world, capsys, tmp_path):
    d, path = world
    win = tmp_path / "w.yaml"
    win.write_text("windows:\n  - {name: synthetic_dip, start: 2019-06-03, end: 2019-06-28}\n"
                   "  - {name: nothing_here, start: 2001-01-02, end: 2001-01-09}\n")
    rc = main(["radar", "--ledger", str(path), "--data", str(d / "bars"), "--windows", str(win), "--json"])
    rep = json.loads(capsys.readouterr().out)
    assert rc in (0, 2) and {r["name"] for r in rep["scenarios"]} >= {"synthetic_dip", "nothing_here", "single_stock_earnings_gap"}
    assert next(r for r in rep["scenarios"] if r["name"] == "nothing_here")["status"] == "no_data"


def test_rate_function_has_no_fallback(world):
    d, _ = world
    rate = _series_rate_fn(_frames())
    assert rate("USD", None) == 1.0
    with pytest.raises(LookupError):
        rate("JPY", None)
