import copy
from pathlib import Path

import pytest
import yaml

from opt_specs import make_spec
from tradex.options.contract import Right
from tradex.options.spec import STRUCTURES, OptionStrategySpec, load_dir
from tradex.strategy.spec import StrategySpec

SPEC_DIR = Path(__file__).resolve().parents[2] / "tradex" / "options" / "specs"
ROOT = Path(__file__).resolve().parents[2]


def raw(name):
    return yaml.safe_load((SPEC_DIR / name).read_text())


def test_bundled_specs_validate():
    specs = load_dir(SPEC_DIR)
    assert {s.structure for s in specs} == {"long_call", "long_put", "naked_call"}
    for s in specs:
        assert s.validate() == [], s.id


def test_structure_properties():
    by = {s.structure: s for s in load_dir(SPEC_DIR)}
    c, p, n = by["long_call"], by["long_put"], by["naked_call"]
    assert (c.entry_key, c.right, c.side, c.is_naked) == ("long", Right.CALL, 1, False)
    assert (p.entry_key, p.right, p.side, p.is_naked) == ("short", Right.PUT, 1, False)
    assert (n.entry_key, n.right, n.side, n.is_naked) == ("short", Right.CALL, -1, True)
    assert c.asset_class == "options" and c.universe[0] == "$watchlist" and c.signal_tf == "D1"
    assert c.status == "proposed" and c.family == "breakout" and c.cap_risk_pct == 1.0


def test_underlying_spec_is_a_normal_stock_spec():
    s = load_dir(SPEC_DIR)[0]
    assert isinstance(s.base, StrategySpec) and s.base.asset_class == "stocks"
    assert s.base.validate() == []
    assert s.base.exit.stop_atr == 2.0 and s.base.entry["long"].startswith("close >")


def test_option_blocks_parse_into_typed_rules():
    c = next(s for s in load_dir(SPEC_DIR) if s.structure == "long_call")
    o = c.option
    assert (o.dte.min, o.dte.max) == (30, 60) and o.strike.by == "delta" and o.strike.target == 0.40
    assert o.liquidity.min_open_interest == 200 and o.exit.close_dte == 10 and o.exit.premium_stop_pct == 0.5
    n = next(s for s in load_dir(SPEC_DIR) if s.is_naked).option.naked
    assert n.stop_premium_mult == 2.0 and n.gap_pct == 0.20 and n.max_days_to_cover == 5


def test_stock_check_command_is_unaffected_by_option_specs():
    from tradex.strategy.spec import load_dir as stock_load
    ids = {s.id for s in stock_load(ROOT / "strategies")}
    assert ids and not any(i.startswith("opt-") for i in ids)
    assert all(s.validate() == [] for s in stock_load(ROOT / "strategies"))


def test_cli_check_runs(capsys):
    from tradex.options.__main__ import main
    assert main(["check"]) == 0
    out = capsys.readouterr().out
    assert "ok   opt-naked-call-overbought-fade" in out
    assert main(["nonsense"]) == 2


def test_cli_check_fails_on_a_bad_spec(tmp_path, capsys):
    from tradex.options.__main__ import main
    d = raw("naked-call-overbought-fade.yaml")
    d["option"]["naked"]["stop_premium_mult"] = None
    (tmp_path / "bad.yaml").write_text(yaml.safe_dump(d))
    assert main(["check", str(tmp_path)]) == 1
    assert "FAIL" in capsys.readouterr().out


@pytest.mark.parametrize("mutate,msg", [
    (lambda d: d.update(asset_class="stocks"), "asset_class must be 'options'"),
    (lambda d: d.update(structure="straddle"), "structure must be one of"),
    (lambda d: d["underlying"].update(asset_class="forex"), "underlying.asset_class"),
    (lambda d: d["entry"].update(short="close < 1"), "entry.short is not used by long_call"),
    (lambda d: d["entry"].pop("long"), "reads entry.long"),
    (lambda d: d["option"]["dte"].update(min=0), "option.dte"),
    (lambda d: d["option"]["dte"].update(min=10, max=5), "option.dte"),
    (lambda d: d["option"]["exit"].update(close_dte=45), "close_dte must be below"),
    (lambda d: d["option"]["strike"].update(by="atm"), "option.strike.by"),
    (lambda d: d["option"]["strike"].update(target=1.4), "absolute delta"),
    (lambda d: d["option"]["strike"].update(tolerance=0), "tolerance"),
    (lambda d: d["option"]["liquidity"].update(max_spread_pct=0), "max_spread_pct"),
    (lambda d: d["option"]["liquidity"].update(min_bid=-1), "liquidity values"),
    (lambda d: d["option"].update(iv={"min": 2.0, "max": 1.0}), "iv.min"),
    (lambda d: d["option"]["exit"].update(premium_stop_pct=1.5), "premium_stop_pct"),
    (lambda d: d["option"]["exit"].update(premium_target_mult=0), "premium_target_mult"),
    (lambda d: d["option"].update(bogus={}), "unknown key 'option.bogus'"),
    (lambda d: d["option"]["dte"].update(weeks=2), "unknown key 'option.dte.weeks'"),
    (lambda d: d["option"].update(naked={"stop_premium_mult": 2.0}), "option.naked applies to naked_call only"),
    (lambda d: d["exit"].update(stop_atr=0), "stop"),
    (lambda d: d["features"].update(x={"fn": "nope.fn"}), "not registered"),
    (lambda d: d.update(status="shipped"), "status must be one of"),
])
def test_long_call_validation_errors(mutate, msg):
    d = raw("long-call-on-breakout.yaml")
    mutate(d)
    errs = OptionStrategySpec.from_dict(d).validate()
    assert any(msg in e for e in errs), errs


@pytest.mark.parametrize("mutate,msg", [
    (lambda d: d["option"]["naked"].pop("stop_premium_mult"), "stop is mandatory"),
    (lambda d: d["option"]["naked"].update(stop_premium_mult=1.0), "stop is mandatory"),
    (lambda d: d["option"]["naked"].update(stop_premium_mult=0.5), "stop is mandatory"),
    (lambda d: d["option"]["naked"].update(gap_pct=0.15), "cannot be below 20%"),
    (lambda d: d["option"]["naked"].update(max_short_interest_pct_float=35), "cannot exceed 20"),
    (lambda d: d["option"]["naked"].update(max_days_to_cover=9), "cannot exceed 5"),
    (lambda d: d["option"]["naked"].update(earnings_buffer_days=-1), "earnings_buffer_days"),
    (lambda d: d.update(filters=[]), "no_earnings_3d"),
    (lambda d: d["option"]["exit"].update(premium_stop_pct=0.5), "long options"),
])
def test_naked_call_cannot_relax_the_limits(mutate, msg):
    d = raw("naked-call-overbought-fade.yaml")
    mutate(d)
    errs = OptionStrategySpec.from_dict(d).validate()
    assert any(msg in e for e in errs), errs


def test_naked_may_tighten_limits():
    d = raw("naked-call-overbought-fade.yaml")
    d["option"]["naked"].update(gap_pct=0.30, max_short_interest_pct_float=10, max_days_to_cover=3, earnings_buffer_days=5)
    assert OptionStrategySpec.from_dict(d).validate() == []


def test_long_put_reads_the_short_entry():
    s = make_spec("long_put")
    assert s.entry_key == "short" and s.validate() == []
    d = copy.deepcopy(s.raw)
    d["entry"] = {"long": "close > 0"}
    assert any("reads entry.short" in e for e in OptionStrategySpec.from_dict(d).validate())


def test_with_params_and_param_grid():
    s = load_dir(SPEC_DIR)[0]
    t = s.with_params({"stop_atr": 3.0, "option.strike.target": 0.55, "features.hi50.period": 30})
    assert t.base.exit.stop_atr == 3.0 and t.option.strike.target == 0.55 and t.base.features["hi50"]["period"] == 30
    assert s.base.exit.stop_atr == 2.0 and s.option.strike.target == 0.40          # original untouched
    grid = s.param_grid(points=3)
    assert grid and all({"stop_atr", "option.strike.target"} <= set(g) for g in grid)
    assert len({tuple(sorted(g.items())) for g in grid}) == len(grid)


def test_structures_table():
    assert set(STRUCTURES) == {"long_call", "long_put", "naked_call"}


def test_make_spec_helper_builds_valid_specs():
    for st in STRUCTURES:
        assert make_spec(st).validate() == []
