"""Live research columns: built at each close from data known then, equal to research bar for bar,
blocked (with one fault) when a pair or an official rate is missing. Recorded/fixture data only, no network."""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tradex.data import macro_history as mh
from tradex.research import builders
from tradex.research.sources import CsvRateSource
from tradex.runtime.columns import LiveColumns, builder_for, daily_close
from tradex.runtime.policy_rates import LivePolicyRates
from tradex.strategy.spec import StrategySpec, load_dir

ROOT = Path(__file__).resolve().parents[1]
FIX = Path(__file__).parent / "fixtures" / "macro"
PROPOSED = {s.id: s for s in load_dir(ROOT / "strategies" / "proposed")}
FX_IDS = ["fx-carry-trend", "fx-carry-vol-filter", "fx-currency-strength-momentum", "fx-rate-diff-trend"]
PAIRS = PROPOSED["fx-carry-trend"].universe
NY = "America/New_York"


def daily_frames(start="2024-01-02", end="2026-10-02", drop=None):
    """Random-walk daily candles stamped like Oanda's (open 17:00 New York the day before a weekday close)."""
    closes = pd.bdate_range(start, end).map(lambda d: pd.Timestamp(d.date()).replace(hour=17).tz_localize(NY))
    idx = pd.DatetimeIndex(closes).tz_convert("UTC") - pd.Timedelta(days=1)
    out = {}
    for i, p in enumerate(PAIRS):
        rng = np.random.default_rng(i)
        px = (150.0 if "JPY" in p else 1.0) * np.exp(np.cumsum(rng.normal(0.0002 * (i - 4), 0.006, len(idx))))
        out[p] = pd.DataFrame({"open": px, "high": px * 1.003, "low": px * 0.997, "close": px, "volume": 1000.0},
                              index=idx)
    return out


class Candles:
    """Stands in for the Oanda candles endpoint: only candles complete by ``end`` come back."""

    def __init__(self, frames, fail=()):
        self.frames, self.fail, self.calls = frames, set(fail), []

    def get_bars(self, symbol, tf, start, end):
        self.calls.append((symbol, tf))
        if symbol in self.fail:
            raise ConnectionError("candles endpoint down")
        df = self.frames[symbol]
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        return df[(df.index >= s) & (df.index + pd.Timedelta(days=1) <= e)]


def official(tmp_path, fetched="2026-10-05T07:00:00+00:00"):
    """The recorded official raw files as the live cache, all fetched at ``fetched``."""
    cache = tmp_path / "macro_live"
    shutil.copytree(FIX, cache)
    for p in cache.glob("*.source.json"):
        p.write_text(f'{{"url": "x", "fetched_at": "{fetched}"}}')
    return cache


def research(spec_id, frames, tmp_path, monkeypatch):
    monkeypatch.setattr(builders, "fx_bars", lambda syms, tf: {s: frames[s] for s in syms})
    monkeypatch.setattr(builders, "RATES_FILE", mh.write_rates(FIX, tmp_path / "policy_rates.csv"))
    name = "fx_strength" if spec_id == "fx-currency-strength-momentum" else "fx_carry"
    data, missing = builders.BUILDERS[name](PROPOSED[spec_id])
    assert not missing
    return data


def test_each_proposed_fx_strategy_has_the_live_builder_research_uses():
    from tradex.research import gate
    plans = {p.spec.stem: p.builder for p in gate.PLANS if p.spec is not None}
    for sid in FX_IDS:
        assert builder_for(PROPOSED[sid]) == plans[sid], sid
    ambiguous = StrategySpec.from_dict({
        "id": "x", "version": 1, "asset_class": "forex", "universe": ["EUR_USD"], "family": "trend",
        "timeframes": {"signal": "D1"}, "features": {"t": {"fn": "data.column", "name": "ts_mom"}},
        "entry": {"long": "t > 0", "short": "t < 0"}, "holding": {"expected_hours": 24, "crosses_rollover": True},
        "exit": {"stop_atr": 2.0, "target_r": 2.0, "max_bars": 10}})
    assert builder_for(ambiguous) is None                            # 63- or 100-bar trend? held back


def test_live_official_rates_read_like_the_research_table(tmp_path):
    live = LivePolicyRates(mh.RATE_FILES, cache=official(tmp_path))
    live.load_cached()
    csv = CsvRateSource(mh.write_rates(FIX, tmp_path / "policy_rates.csv"))
    idx = pd.date_range("2008-01-01", "2026-10-05", freq="D", tz="UTC")
    for ccy in mh.RATE_FILES:
        pd.testing.assert_series_equal(live.rates(ccy, idx), csv.rates(ccy, idx), check_names=False)


def test_live_columns_on_a_replayed_series_equal_research_bar_for_bar(tmp_path, monkeypatch):
    """Replay daily candles one close at a time: the live value at each close is the research value of that bar."""
    frames = daily_frames("2024-02-01", "2025-06-30")
    expect = {sid: research(sid, frames, tmp_path, monkeypatch) for sid in FX_IDS}
    rates = LivePolicyRates(mh.RATE_FILES, cache=official(tmp_path, fetched="2024-01-01T00:00:00+00:00"),
                            max_age=pd.Timedelta(days=10_000))
    rates.load_cached()
    specs = [PROPOSED[sid] for sid in FX_IDS]
    first = frames[PAIRS[0]].index[0]
    lc = LiveColumns(specs, Candles(frames), rates)
    lc.warm(first + pd.Timedelta(hours=1))                           # nothing has closed yet
    assert not lc.frames
    got = {sid: {p: [] for p in PAIRS} for sid in FX_IDS}
    for t in frames[PAIRS[0]].index:
        ts = t + pd.Timedelta(days=1, hours=3)                       # a store close three hours after this candle's
        lc.before_close("H1", ts)
        for sid in FX_IDS:
            g = lc.by_spec[sid]
            assert g.fault is None and g.asof == t + pd.Timedelta(days=1)
            for p in PAIRS:
                got[sid][p].append(lc.latest(sid, p).rename(t))
    for sid in FX_IDS:
        cols = sorted(c for c in expect[sid][PAIRS[0]].columns if c not in ("open", "high", "low", "close", "volume"))
        for p in PAIRS:
            live = pd.DataFrame(got[sid][p])[cols]
            live.index = live.index.astype(expect[sid][p].index.dtype)
            pd.testing.assert_frame_equal(live, expect[sid][p][cols], check_freq=False, check_names=False)
        assert expect[sid]["EUR_USD"][cols].iloc[-1].notna().all(), sid    # warm by the end: real values compared
    assert expect["fx-rate-diff-trend"]["EUR_USD"]["rate_chg"].abs().max() > 0      # a rate change inside the window


def store_d1(frames, end):
    """UTC-midnight daily bars like the bar store resamples from H1 (index = open)."""
    idx = pd.date_range("2026-09-01", end, freq="D", tz="UTC")
    return pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}, index=idx)


def test_store_bars_read_the_latest_daily_candle_closed_by_their_close(tmp_path):
    frames = daily_frames()
    rates = LivePolicyRates(mh.RATE_FILES, cache=official(tmp_path))
    rates.load_cached()
    spec = PROPOSED["fx-carry-trend"]
    lc = LiveColumns([spec], Candles(frames), rates)
    close = pd.Timestamp("2026-10-02 00:00", tz="UTC")              # store D1 close: Thursday 20:00 New York
    lc.warm(close)
    lc.update(close)
    bars = store_d1(frames, "2026-10-01")
    out = lc.attach(spec, "EUR_USD", bars, close)
    src = lc.by_spec[spec.id].cols["EUR_USD"]
    thu = pd.Timestamp("2026-10-01 17:00", tz=NY).tz_convert("UTC")   # Thursday's candle closed 21:00Z
    assert lc.by_spec[spec.id].asof == thu
    assert out["carry"].iloc[-1] == src["carry"].iloc[-1]
    assert out["ts_mom"].iloc[-2] == src.loc[thu - pd.Timedelta(days=2), "ts_mom"]   # Wednesday's candle
    assert daily_close(pd.Timestamp("2026-10-05 00:00", tz="UTC")) == pd.Timestamp("2026-10-02 17:00", tz=NY)


def test_a_missing_pair_blocks_the_strategy_with_one_fault_and_never_fills(tmp_path):
    frames = daily_frames()
    rates = LivePolicyRates(mh.RATE_FILES, cache=official(tmp_path))
    rates.load_cached()
    specs = [PROPOSED["fx-carry-trend"], PROPOSED["fx-currency-strength-momentum"]]
    candles = Candles(frames)
    lc = LiveColumns(specs, candles, rates)
    faults = []
    store = type("S", (), {})()
    lc.bind(store, lambda check, ok, detail, ts: faults.append((check, ok, detail)))
    assert store.research is lc
    t0 = pd.Timestamp("2026-09-29 00:00", tz="UTC")
    held = {p: f[f.index + pd.Timedelta(days=1) <= t0] for p, f in frames.items()}
    lc.provider = Candles(held)
    lc.warm(t0)
    lc.update(t0)
    assert faults == [] and lc.by_spec["fx-carry-trend"].fault is None
    lc.provider = Candles(frames, fail={"GBP_JPY"})                  # one pair's poll fails at the next close
    t1 = t0 + pd.Timedelta(days=1)
    for h in range(3):
        lc.before_close("H1", t1 + pd.Timedelta(hours=h))
    assert len(faults) == 2 and all(not ok for _, ok, _ in faults)    # one per strategy group, not per bar
    assert "GBP_JPY" in faults[0][2] and "candles endpoint down" in faults[0][2]
    assert lc.attach(specs[0], "EUR_USD", store_d1(frames, "2026-09-29"), t1) is None
    assert lc.latest("fx-carry-trend", "EUR_USD") is None
    lc.provider = Candles(frames)
    lc.before_close("H1", t1 + pd.Timedelta(hours=4))
    assert [ok for _, ok, _ in faults[2:]] == [True, True]
    assert lc.attach(specs[0], "EUR_USD", store_d1(frames, "2026-09-29"), t1) is not None


def test_missing_or_stale_rates_block_carry_only(tmp_path):
    frames = daily_frames()
    now = pd.Timestamp("2026-10-02 00:00", tz="UTC")                 # Friday 00:00Z
    rates = LivePolicyRates(mh.RATE_FILES, cache=official(tmp_path, fetched="2026-09-28T06:00:00+00:00"))
    rates.load_cached()
    lc = LiveColumns([PROPOSED["fx-carry-trend"], PROPOSED["fx-currency-strength-momentum"]], Candles(frames), rates)
    lc.warm(now)
    lc.update(now)
    carry, strength = lc.by_spec["fx-carry-trend"], lc.by_spec["fx-currency-strength-momentum"]
    assert "stale" in carry.fault and "last fetched 2026-09-28" in carry.fault and strength.fault is None
    sat = pd.Timestamp("2026-10-03 12:00", tz="UTC")                 # not a business day: not stale
    assert rates.problem("USD", sat) is None
    empty = LivePolicyRates(mh.RATE_FILES, cache=tmp_path / "nothing")
    empty.load_cached()
    assert empty.problem("EUR", now).startswith("no official EUR policy rate")
    unknown = LivePolicyRates(["SGD"], cache=tmp_path / "nothing")
    assert "no official series" in unknown.problem("SGD", now)
    lc.rates = None
    lc.update(now)
    assert carry.fault == "no official policy-rate source"


def test_daily_refresh_keeps_the_last_good_series_and_its_age(tmp_path):
    cache = tmp_path / "live"
    clock = [pd.Timestamp("2026-10-05 08:00", tz="UTC")]
    down = {"on": False}

    def get(url):
        if down["on"]:
            raise TimeoutError("publisher down")
        name = next(n for n, u in mh.RAW.items() if u == url)
        return (FIX / name).read_bytes()

    r = LivePolicyRates(["EUR", "USD"], cache=cache, get=get, clock=lambda: clock[0])
    assert r.due(clock[0]) and r.refresh() == [] and not r.due(clock[0])
    assert r.fetched_at("EUR") == clock[0].floor("s") and r.problem("EUR", clock[0]) is None
    down["on"] = True
    clock[0] += pd.Timedelta(days=2)
    errs = r.refresh()
    assert len(errs) == 2 and "publisher down" in errs[0]
    assert r.problem("EUR", clock[0]) is None                        # two days old: still usable
    clock[0] += pd.Timedelta(days=2)                                 # Friday, four days after the last good fetch
    assert "stale" in r.problem("EUR", clock[0]) and "publisher down" in r.problem("EUR", clock[0])
    idx = pd.DatetimeIndex(["2024-06-13"], tz="UTC")
    assert r.rates("EUR", idx).iloc[0] == pytest.approx(0.0375)       # the last good series is still what it reads


def test_oanda_financing_disagreement_blocks_carry(tmp_path):
    frames = daily_frames()
    now = pd.Timestamp("2026-10-03 00:00", tz="UTC")
    seen = {}

    def financing(pairs):
        seen["pairs"] = pairs
        return {"EUR_USD": (0.05 - 0.025, -0.05 - 0.025)}           # Oanda says +5.00% for EUR_USD

    rates = LivePolicyRates(mh.RATE_FILES, pairs=PAIRS, cache=official(tmp_path), financing=financing)
    rates.load_cached()
    lc = LiveColumns([PROPOSED["fx-carry-vol-filter"]], Candles(frames), rates)
    lc.warm(now)
    lc.update(now)
    assert lc.by_spec["fx-carry-vol-filter"].fault is None
    gaps = rates.cross_check(now)
    assert seen["pairs"] == sorted(PAIRS) and len(gaps) == 1 and "EUR_USD" in gaps[0]
    lc.update(now)
    assert "disagrees with Oanda financing" in lc.by_spec["fx-carry-vol-filter"].fault
    rates.financing = lambda pairs: (_ for _ in ()).throw(ConnectionError("no route"))
    assert "cross-check not run" in rates.cross_check(now)[0]        # optional: a failed check is reported only


def test_fetch_financing_reads_the_instruments_financing_field():
    from tradex.data.oanda import fetch_financing
    calls = {}

    def http(url, headers):
        calls["url"] = url
        return {"instruments": [{"name": "EUR_USD", "financing": {"longRate": "-0.0386", "shortRate": "0.0124"}},
                                {"name": "USD_JPY"}]}

    got = fetch_financing(["USD_JPY", "EUR_USD"], account_id="acct", token="tok", http=http)
    assert got == {"EUR_USD": (-0.0386, 0.0124)}
    assert calls["url"].startswith("https://api-fxpractice.oanda.com/v3/accounts/acct/instruments?instruments=EUR_USD")
