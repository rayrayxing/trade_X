"""Data builders for the proposed strategies: real bars plus the research columns each spec reads.

Each builder takes the spec and returns ``(data, missing)`` like ``gate.build``. Inputs come
from files on the machine that runs the gate (OpenD and Oanda caches, calendars, rate history);
when a required file is absent the builder raises ``DataUnavailable`` and the gate reports
"needs data". Nothing is generated or assumed. The paths are module attributes so tests can
point them at fixtures.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from tradex.data.opend import DEFAULT_CACHE as OPEND_CACHE, load_cached
from tradex.research import panels
from tradex.research.carry import carry_columns
from tradex.research.events import earnings_columns, event_window_columns
from tradex.research.regime import regime_columns
from tradex.research.rel_strength import currency_strength, pair_strength_diff, rs_momentum
from tradex.research.sources import CsvEarningsCalendar, CsvRateSource, DataUnavailable, FileEventCalendar
from tradex.strategy.spec import StrategySpec
from tradex.timeframes import OHLCV, duration

ROOT = Path(__file__).resolve().parents[2]
EARNINGS_DIR = ROOT / "data" / "calendar" / "earnings"
FOMC_FILE = ROOT / "data" / "calendar" / "fomc_history.csv"
RATES_FILE = ROOT / "data" / "rates" / "policy_rates.csv"
OANDA_CACHE = ROOT / "data" / "cache" / "oanda"

MARKET, BOND, GOLD = "SPY", "TLT", "GLD"
SECTOR_BASKET = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB"]


def _symbols(spec: StrategySpec) -> list[str]:
    return [s for s in spec.universe if not s.startswith("$")]


def us_bars(symbols, tf: str, cache=None) -> dict[str, pd.DataFrame]:
    cache = OPEND_CACHE if cache is None else cache
    out = {}
    for s in symbols:
        b = load_cached(s, tf, cache)
        if b is not None and len(b):
            out[s] = b
    return out


def fx_bars(symbols, tf: str) -> dict[str, pd.DataFrame]:
    from tradex.data.oanda_history import OandaHistory
    h = OandaHistory(cache_dir=OANDA_CACHE)
    out = {}
    for s in symbols:
        b = h.load(s, tf)
        if b is not None and len(b):
            out[s] = b[OHLCV]
    if not out:
        raise DataUnavailable(f"no Oanda {tf} history in {OANDA_CACHE} (run `python -m tradex.research.universe oanda`)")
    return out


def _regime_panel(cache) -> pd.DataFrame:
    """The market-wide regime columns from the SPY / TLT / GLD / sector-ETF daily bars in the cache."""
    need = [MARKET, BOND, GOLD] + SECTOR_BASKET
    bars = us_bars(need, "D1", cache)
    absent = [s for s in need if s not in bars]
    if absent:
        raise DataUnavailable(f"regime columns need daily bars for {absent}")
    return regime_columns({s: b["close"] for s, b in bars.items()}, MARKET, BOND, GOLD, SECTOR_BASKET)


def _attach(data: dict[str, pd.DataFrame], cols: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {s: panels.with_columns(b, cols) for s, b in data.items()}


def earnings(spec: StrategySpec, cache=None):
    cal = CsvEarningsCalendar(EARNINGS_DIR)
    syms = _symbols(spec)
    data = us_bars(syms, spec.signal_tf, cache)
    have = [s for s in data if cal.has(s)]
    if not have:
        raise DataUnavailable(f"no earnings-date files for the universe in {EARNINGS_DIR} (one <SYMBOL>.csv with date[,when])")
    bench = us_bars(["SPY"], spec.signal_tf, cache).get("SPY")
    out = {s: panels.with_columns(data[s], earnings_columns(data[s], cal.announcements(s), bench)) for s in have}
    return out, [s for s in syms if s not in out]


def fomc_window(spec: StrategySpec, cache=None):
    syms = _symbols(spec)
    data = us_bars(syms, spec.signal_tf, cache)
    times = FileEventCalendar(FOMC_FILE, kind="fomc").times()
    cols = {s: event_window_columns(b, times, duration(spec.signal_tf)) for s, b in data.items()}
    return {s: panels.with_columns(b, cols[s]) for s, b in data.items()}, [s for s in syms if s not in data]


def regime_etf(spec: StrategySpec, cache=None):
    syms = _symbols(spec)
    data = us_bars(syms, spec.signal_tf, cache)
    return _attach(data, _regime_panel(cache)), [s for s in syms if s not in data]


def rs_crash(spec: StrategySpec, cache=None):
    syms = _symbols(spec)
    data = us_bars(syms, spec.signal_tf, cache)
    spy = us_bars([MARKET], "D1", cache).get(MARKET)
    if spy is None:
        raise DataUnavailable("relative strength needs SPY daily bars")
    rg = _regime_panel(cache)
    rsm = pd.DataFrame({s: rs_momentum(b["close"], spy["close"]) for s, b in data.items()})
    xs = panels.xs_percentile(rsm)
    out = {s: panels.with_columns(b, {"rsm": rsm[s], "rsm_xs": xs[s], "rg_crash": rg["rg_crash"]}) for s, b in data.items()}
    return out, [s for s in syms if s not in data]


def fx_strength_columns(data: dict[str, pd.DataFrame], n: int = 63) -> dict[str, dict[str, pd.Series]]:
    """cs_diff, cs_xs, ts_mom per pair. The live builder (tradex.runtime.columns) calls this same function."""
    strength = currency_strength({s: b["close"] for s, b in data.items()}, n)
    diff = pd.DataFrame({s: pair_strength_diff(s, strength) for s in data})
    xs = panels.xs_percentile(diff)
    return {s: {"cs_diff": diff[s], "cs_xs": xs[s], "ts_mom": b["close"].pct_change(n)} for s, b in data.items()}


def fx_carry_columns(data: dict[str, pd.DataFrame], rates) -> dict[str, pd.DataFrame]:
    """The carry columns plus carry_xs per pair. The live builder (tradex.runtime.columns) calls this same function."""
    cols = {s: carry_columns(b, s, rates) for s, b in data.items()}
    xs = panels.xs_percentile(pd.DataFrame({s: c["carry"] for s, c in cols.items()}))
    return {s: cols[s].assign(carry_xs=xs[s]) for s in cols}


def fx_strength(spec: StrategySpec, cache=None, n: int = 63):
    syms = _symbols(spec)
    data = fx_bars(syms, spec.signal_tf)
    cols = fx_strength_columns(data, n)
    return {s: panels.with_columns(b, cols[s]) for s, b in data.items()}, [s for s in syms if s not in data]


def fx_carry(spec: StrategySpec, cache=None):
    syms = _symbols(spec)
    data = fx_bars(syms, spec.signal_tf)
    cols = fx_carry_columns(data, CsvRateSource(RATES_FILE))
    return {s: panels.with_columns(b, cols[s]) for s, b in data.items()}, [s for s in syms if s not in data]


BUILDERS = {"earnings": earnings, "fomc_window": fomc_window, "regime_etf": regime_etf, "rs_crash": rs_crash,
            "fx_strength": fx_strength, "fx_carry": fx_carry}
