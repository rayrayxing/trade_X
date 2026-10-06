import json
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from tradex.core.ledger import Ledger
from tradex.core.records import EquitySnapshot
from tradex.data.providers import CsvProvider
from tradex.research import builders
from tradex.research.loop import adapters
from tradex.research.loop.adapters import (CollectingAlerter, GateResearchData, LedgerLiveReturns, LockedHoldout,
                                           UnavailableHoldout, UnavailableLive, require_real, urllib_get)
from tradex.research.loop.ports import (DataUnavailable, HoldoutAlreadyLooked, HoldoutUnavailable, LiveDataMissing)
from tradex.strategy.spec import StrategySpec
from tradex.timeframes import duration

from loopkit import make_frames, spec_dict


# --- the holdout lock, as tradex.backtest.holdout exposes it ----------------------------------------

class HoldoutLocked(PermissionError):
    pass


def lock_module(tmp_path, start="2022-01-03"):
    ledger = tmp_path / "looks.jsonl"

    class Policy:
        def __init__(self):
            self.start = pd.Timestamp(start, tz="UTC")

        @classmethod
        def from_config(cls, path=None):
            return cls()

    def research_view(frames, tf, policy):
        return {s: df[df.index <= policy.start - duration(tf)] for s, df in frames.items()}

    def looks(policy):
        return [json.loads(x) for x in ledger.read_text().splitlines()] if ledger.exists() else []

    def holdout_look(frames, strategy_id, version, reason, policy):
        if any((lk["strategy_id"], lk["version"]) == (strategy_id, str(version)) for lk in looks(policy)):
            raise HoldoutLocked("already looked")
        with ledger.open("a") as fh:
            fh.write(json.dumps({"strategy_id": strategy_id, "version": str(version), "reason": reason}) + "\n")
        return {s: df[df.index >= policy.start] for s, df in frames.items()}

    return SimpleNamespace(HoldoutPolicy=Policy, HoldoutLocked=HoldoutLocked, research_view=research_view, looks=looks,
                           holdout_look=holdout_look)


def test_locked_holdout_delegates_to_the_lock_and_translates_its_refusal(tmp_path):
    h = LockedHoldout(module=lock_module(tmp_path))
    frames = make_frames(("AAA",))
    cut = h.research_view(frames, "D1")["AAA"]
    assert cut.index.max() + duration("D1") <= h.start
    assert h.has_looked("stk-a", 1) is False
    held = h.look(frames, "stk-a", 1, "why")["AAA"]
    assert held.index.min() >= h.start and h.has_looked("stk-a", 1) and not h.has_looked("stk-a", 2)
    with pytest.raises(HoldoutAlreadyLooked):
        h.look(frames, "stk-a", 1, "again")


def test_locked_holdout_is_unavailable_when_the_lock_is_not_installed(monkeypatch):
    monkeypatch.setitem(sys.modules, "tradex.backtest.holdout", None)       # import raises ImportError
    with pytest.raises(HoldoutUnavailable, match="not in this checkout"):
        LockedHoldout()


def test_locked_holdout_is_unavailable_when_the_policy_cannot_be_read(tmp_path):
    mod = lock_module(tmp_path)

    def broken(path=None):
        raise FileNotFoundError("config/gates/holdout.yaml")
    mod.HoldoutPolicy.from_config = staticmethod(broken)
    with pytest.raises(HoldoutUnavailable, match="unreadable"):
        LockedHoldout(module=mod)


def test_the_unavailable_stand_ins_raise_on_every_use():
    h = UnavailableHoldout("lock missing")
    for call in (lambda: h.start, lambda: h.research_view({}, "D1"), lambda: h.has_looked("a", 1),
                 lambda: h.look({}, "a", 1, "r")):
        with pytest.raises(HoldoutUnavailable, match="lock missing"):
            call()
    with pytest.raises(LiveDataMissing, match="no ledger"):
        UnavailableLive("no ledger").daily_returns("a", 1, None)


# --- paper results from the trading ledger -----------------------------------------------------------

def snap(led, book, day, equity):
    led.append(EquitySnapshot(time=f"{day}T21:00:00+00:00", book=book, equity_usd=equity, cash_usd=equity,
                              open_risk_usd=0.0, positions=[], exposure_by_currency={}, limits={}, config_hash="x"))


def test_live_returns_come_from_the_strategys_own_virtual_book():
    led = Ledger(":memory:", run_id="t")
    for d, e in (("2026-09-01", 10000.0), ("2026-09-02", 10100.0), ("2026-09-03", 10000.0), ("2026-09-04", 10200.0)):
        snap(led, "virtual:stk-a", d, e)
    snap(led, "ensemble", "2026-09-02", 99999.0)                        # another book is ignored
    r = LedgerLiveReturns(led).daily_returns("stk-a", 1, None)
    assert list(r.round(4)) == [0.01, -0.0099, 0.02]
    assert r.index.tz is not None and str(r.index[0].date()) == "2026-09-02"
    since = LedgerLiveReturns(led).daily_returns("stk-a", 1, pd.Timestamp("2026-09-03", tz="UTC"))
    assert len(since) == 2


def test_live_returns_raise_when_the_book_has_no_snapshots():
    led = Ledger(":memory:", run_id="t")
    snap(led, "ensemble", "2026-09-02", 1.0)
    with pytest.raises(LiveDataMissing, match="virtual:stk-a"):
        LedgerLiveReturns(led).daily_returns("stk-a", 1, None)


# --- research data -----------------------------------------------------------------------------------

def test_builder_comes_from_the_gate_plan_and_unknown_column_readers_are_refused():
    g = GateResearchData()
    assert g.builder_for(StrategySpec.from_dict(spec_dict("stk-pead-ear", features={"x": {"fn": "data.column", "name": "ed_z"}},
                                                          entry={"long": "x > 1"}))) == "earnings"
    plain = StrategySpec.from_dict(spec_dict("stk-unknown-plain"))
    assert g.builder_for(plain) == "us_d1"
    fx = StrategySpec.from_dict(spec_dict("fx-unknown-plain", asset_class="forex", holding={"expected_hours": 24, "crosses_rollover": True}))
    assert g.builder_for(fx) == "fx"
    cols = StrategySpec.from_dict(spec_dict("stk-unknown-cols", features={"x": {"fn": "data.column", "name": "my_col"}},
                                            entry={"long": "x > 1"}))
    with pytest.raises(DataUnavailable, match="no data builder is registered"):
        g.builder_for(cols)


def test_frames_are_the_cached_real_bars_and_missing_ones_raise(tmp_path):
    cache = tmp_path / "opend"
    for s, df in make_frames(("AAA", "BBB"), n=300).items():
        CsvProvider(cache).save(s, "D1", df)
    spec = StrategySpec.from_dict(spec_dict("stk-unknown-plain", universe=("AAA", "BBB", "CCC")))
    got = GateResearchData(cache=cache).frames(spec)
    assert sorted(got) == ["AAA", "BBB"] and len(got["AAA"]) == 300           # CCC is simply absent, nothing made up for it
    with pytest.raises(DataUnavailable, match="no cached bars"):
        GateResearchData(cache=tmp_path / "empty").frames(spec)


def test_fx_without_cached_history_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(builders, "OANDA_CACHE", tmp_path / "no-oanda")
    spec = StrategySpec.from_dict(spec_dict("fx-unknown-plain", asset_class="forex", universe=["EUR_USD"],
                                            holding={"expected_hours": 24, "crosses_rollover": True}))
    with pytest.raises(DataUnavailable):
        GateResearchData().frames(spec)


def test_watchlist_strategies_cannot_be_researched_on_a_fixed_universe():
    spec = StrategySpec.from_dict(spec_dict("stk-watch-test", universe=["$watchlist"]))
    with pytest.raises(DataUnavailable, match="watchlist"):
        GateResearchData().frames(spec)


def test_a_data_port_must_say_it_is_real():
    class Good:
        real, source = True, "opend"

    class NotReal:
        real, source = False, "opend"

    class Named:
        real, source = True, "synthetic-feed"
    require_real(Good())
    for bad in (NotReal(), Named(), object()):
        with pytest.raises(DataUnavailable):
            require_real(bad)


def test_the_only_network_getter_is_https_only():
    with pytest.raises(ValueError, match="https"):
        urllib_get()("http://export.arxiv.org/api/query")
    with pytest.raises(ValueError, match="https"):
        urllib_get()("file:///etc/passwd")


def test_collecting_alerter_passes_through():
    seen = []
    inner = SimpleNamespace(alert=lambda *a: seen.append(a))
    c = CollectingAlerter(inner)
    c.alert("warning", "t", "d")
    assert c.items == [{"level": "warning", "title": "t", "detail": "d"}] and seen == [("warning", "t", "d")]
    adapters.LogAlerter().alert("critical", "logged")                     # does not raise
