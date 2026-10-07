"""Paper/live wiring with fakes only: real-data guard, live costs and rates, stream -> bars -> core."""
import pandas as pd
import pytest

import tradex.costs.models as cm
from test_runtime_core import _fx_frames, _mixed_specs
from tradex.cli import main
from tradex.core.interfaces import ReplayClock
from tradex.core.ledger import Ledger
from tradex.costs.models import OandaFxCosts, RateMissing
from tradex.data.guard import RealDataMissing, SyntheticDataRefused
from tradex.data.oanda import QuoteBook, Tick
from tradex.data.providers import CachedProvider, CsvProvider
from tradex.execution.sim import SimBroker
from tradex.runtime.build import VenuesMissing, build_runtime
from tradex.runtime.config import RuntimeConfig
from tradex.runtime.fx import LiveQuoteRates, MissingRate

START = pd.Timestamp("2025-02-05 10:00", tz="UTC")          # a Wednesday: forex is open


@pytest.fixture(autouse=True)
def _reset_run_mode():
    yield
    cm.configure_run_mode("backtest")


class RecordedHistory:
    """Stands in for the Oanda candles endpoint: recorded bars, sliced like the real one."""

    def __init__(self):
        self.frames = _fx_frames()

    def get_bars(self, symbol, tf, start, end):
        df = self.frames[symbol]
        return df[(df.index >= pd.Timestamp(start)) & (df.index < pd.Timestamp(end))]


class FakeStream:
    heartbeats = 0

    def ticks(self):
        return iter(())


class SyntheticFeed:
    def get_bars(self, *a):
        raise AssertionError("never called")


def _build(mode="paper", dry=False, **kw):
    clock = kw.pop("clock", ReplayClock(START))
    quotes = QuoteBook(30, clock=clock.now)
    args = dict(history={"forex": RecordedHistory()}, stream=FakeStream(), quotes=quotes, clock=clock,
                config=RuntimeConfig(), dry=dry, warmup_bars=400)
    args.update(kw)
    return build_runtime(mode, _mixed_specs(), Ledger(":memory:", git_commit="t"), **args), clock, quotes


@pytest.mark.parametrize("provider", [lambda p: CsvProvider(p), lambda p: CachedProvider(RecordedHistory(), p, "x"),
                                      lambda p: SyntheticFeed()])
def test_paper_refuses_csv_cached_and_synthetic_history(tmp_path, provider):
    with pytest.raises(SyntheticDataRefused):
        _build(history={"forex": provider(tmp_path)}, dry=True)


def test_paper_refuses_missing_history_or_stream():
    with pytest.raises(RealDataMissing):
        _build(history={}, dry=True)
    with pytest.raises(RealDataMissing):
        _build(stream=None, dry=True)


def test_runtime_is_for_paper_and_live_only():
    with pytest.raises(ValueError):
        _build(mode="replay", dry=True)


def test_real_run_needs_venue_adapters():
    with pytest.raises(VenuesMissing):
        _build()


def test_dry_run_wires_one_live_quote_book_everywhere_and_connects_nothing():
    rt, _, quotes = _build(dry=True)
    fx = rt.core.costs["forex"]
    assert isinstance(fx, OandaFxCosts) and fx.mode == "paper" and fx.spread_source.quotes is quotes
    assert rt.book.rates.quotes is quotes and rt.core.rates.quotes is quotes and rt.book.venues == {}
    assert cm._RUN_MODE == "paper" and cm._RATE_SOURCE.quotes is quotes
    assert rt.core.cfg.agents_mode == "shadow" and rt.core.cfg.live
    assert not rt.store.frames                                       # nothing fetched
    text = rt.readiness()
    assert "not connected (dry run)" in text and text.splitlines()[-1].startswith("READY")
    with pytest.raises(MissingRate):                                 # no quote yet: no default spread
        fx.fill("EUR_USD", 1, 1.1, START)


def test_missing_rate_is_both_a_lookup_error_and_rate_missing():
    assert issubclass(MissingRate, LookupError) and issubclass(MissingRate, RateMissing)
    with pytest.raises(MissingRate):
        LiveQuoteRates(QuoteBook()).usd_per_unit("JPY")


def test_stream_ticks_become_bars_the_core_runs_on():
    clock = ReplayClock(START)
    rt, _, quotes = _build(clock=clock, venues={"forex": SimBroker(10_000, account_id="sim")})
    have = rt.store.frames["EUR_USD"].index[-1]
    assert have == START - pd.Timedelta(hours=1)                     # warm-up history up to now
    for m in range(0, 60, 5):                                        # ticks through the 10:00 hour
        t = START + pd.Timedelta(minutes=m)
        clock.set(t)
        for sym, px in (("EUR_USD", 1.1 + m * 1e-4), ("USD_JPY", 150.0 + m * 0.01)):
            rt.feed.on_tick(Tick(sym, t, px - 5e-5, px + 5e-5))
    assert quotes.usd_per_unit("JPY") == pytest.approx(1 / 150.55, rel=1e-6)
    t = START + pd.Timedelta(hours=1, seconds=5)                     # first tick of the next hour closes the bar
    clock.set(t)
    for sym, px in (("EUR_USD", 1.1056), ("USD_JPY", 150.56)):
        rt.feed.on_tick(Tick(sym, t, px - 5e-5, px + 5e-5))
    clock.set(START + pd.Timedelta(hours=1, seconds=10))             # past the close and the grace
    ran = rt.runner.tick()
    assert [(e.tf, e.ts) for e in ran] == [("H1", START + pd.Timedelta(hours=1))]
    bar = rt.store.frames["EUR_USD"].iloc[-1]
    assert rt.store.frames["EUR_USD"].index[-1] == START and bar["open"] == pytest.approx(1.1)
    assert bar["close"] == pytest.approx(1.1055)
    assert not [h for h in rt.core.ledger.rows(kind="health") if not h["ok"]]


def test_silent_stream_while_forex_trades_is_one_fault_then_recovery():
    clock = ReplayClock(START)
    rt, _, _ = _build(clock=clock, venues={"forex": SimBroker(10_000, account_id="sim")})
    for minutes in (1, 3, 4):                                        # silent past 120 s, still silent
        clock.set(START + pd.Timedelta(minutes=minutes))
        rt.feed.before_close("H1", clock.now())
    clock.set(START + pd.Timedelta(minutes=5))
    rt.feed.on_tick(Tick("EUR_USD", clock.now(), 1.1, 1.1001))
    rt.feed.before_close("H1", clock.now())
    h = rt.core.ledger.rows(kind="health")
    assert [(r["check"], r["ok"]) for r in h] == [("feed", False), ("feed", True)]
    assert "silent" in h[0]["detail"]


def test_silent_stream_on_the_weekend_is_not_a_fault():
    sat = pd.Timestamp("2025-02-08 12:00", tz="UTC")
    clock = ReplayClock(sat)
    rt, _, _ = _build(clock=clock, venues={"forex": SimBroker(10_000, account_id="sim")})
    clock.set(sat + pd.Timedelta(hours=3))
    rt.feed.before_close("H1", clock.now())
    assert not rt.core.ledger.rows(kind="health")


def _write_fx_strategy(d):
    d.mkdir()
    (d / "fx.yaml").write_text(
        "id: fx-t\nversion: 1\nfamily: trend\nasset_class: forex\nuniverse: [EUR_USD]\nstatus: proposed\n"
        "timeframes: {signal: H1}\nfeatures: {ema: {fn: talib.EMA, period: 20}}\n"
        "entry: {long: close > ema, short: close < ema}\nexit: {stop_atr: 1.5, target_r: 2.0, max_bars: 10}\n"
        "holding: {expected_hours: 24, crosses_rollover: true}\n")


def test_cli_dry_run_prints_readiness_without_a_broker(tmp_path, capsys, monkeypatch):
    import tradex.secrets
    _write_fx_strategy(tmp_path / "s")
    monkeypatch.setattr(tradex.secrets, "has", lambda name: True)
    assert main(["run", "--mode", "paper", "--dry", "--strategies", str(tmp_path / "s")]) == 0
    out = capsys.readouterr().out
    assert "not connected (dry run)" in out and "READY (paper" in out
    monkeypatch.setattr(tradex.secrets, "has", lambda name: name != "oanda_token")
    assert main(["run", "--mode", "paper", "--dry", "--strategies", str(tmp_path / "s")]) == 1
    assert "missing oanda_token" in capsys.readouterr().out


def test_cli_real_run_refuses_without_venue_adapters(tmp_path, capsys, monkeypatch):
    import sys

    import tradex.secrets
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tradex.secrets, "has", lambda name: True)
    monkeypatch.setattr(tradex.secrets, "get", lambda name: "TEST-" + name)   # CI has no Keychain account IDs
    import tradex.execution
    for name in ("oanda", "moomoo"):                                     # adapters unimportable: never connects
        monkeypatch.setitem(sys.modules, f"tradex.execution.{name}", None)
        monkeypatch.delattr(tradex.execution, name, raising=False)
    _write_fx_strategy(tmp_path / "s")
    assert main(["run", "--mode", "paper", "--strategies", "s"]) == 1
    assert "tradex/execution/oanda.py is not there" in capsys.readouterr().err


def test_a_strategy_that_reads_unbuilt_research_columns_is_held_back_once_not_faulted_every_bar():
    from tradex.runtime.build import held_text, split_unfed
    from tradex.strategy.spec import StrategySpec
    carry = StrategySpec.from_dict({
        "id": "fx-carry", "version": 1, "asset_class": "forex", "universe": ["EUR_USD"], "status": "paper",
        "family": "carry", "timeframes": {"signal": "D1"},
        "features": {"cry": {"fn": "data.column", "name": "carry"}, "xs": {"fn": "data.column", "name": "carry_xs"}},
        "entry": {"long": "xs >= 0.7 and cry > 0", "short": "xs <= 0.3 and cry < 0"},
        "holding": {"expected_hours": 960, "crosses_rollover": True},
        "exit": {"stop_atr": 3.0, "target_r": 4.0, "max_bars": 40}})
    run, held = split_unfed(_mixed_specs() + [carry])
    assert [s.id for s in run] == ["h1-trend", "h4-mom"]
    assert held == {"fx-carry": ["carry", "carry_xs"]}
    assert "fx-carry needs carry, carry_xs" in held_text(held)

    rt, _, _ = _build(dry=True)                                      # plain specs: nothing held back
    assert not any(c.name == "strategies:held_back" for c in rt.checks)
    clock = ReplayClock(START)
    rt = build_runtime("paper", _mixed_specs() + [carry], Ledger(":memory:", git_commit="t"),
                       history={"forex": RecordedHistory()}, stream=FakeStream(), quotes=QuoteBook(30, clock=clock.now),
                       clock=clock, config=RuntimeConfig(), warmup_bars=400, dry=True)
    note = next(c for c in rt.checks if c.name == "strategies:held_back")
    assert note.status == "skip" and "fx-carry" in note.detail
    assert all(s.id != "fx-carry" for s in rt.core.strategies)
