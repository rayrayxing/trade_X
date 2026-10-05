"""Data builders for the proposed strategies: real bars plus the research columns each spec reads.

Each builder takes the spec (and, for screened universes, ``members``: date x symbol
membership that limits cross-sectional ranks to the universe of the day) and returns
``(data, missing)`` like ``gate.build``. Inputs come
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
from tradex.research.events import daily_event_columns, earnings_columns, event_window_columns
from tradex.research.regime import regime_columns
from tradex.research.rel_strength import currency_strength, pair_strength_diff, rs_momentum
from tradex.research.sources import CsvEarningsCalendar, CsvRateSource, DataUnavailable, FileEventCalendar
from tradex.strategy.spec import StrategySpec
from tradex.timeframes import duration

ROOT = Path(__file__).resolve().parents[2]
EARNINGS_DIR = ROOT / "data" / "cache" / "earnings"     # tradex.data.earnings writes it
MACRO = ROOT / "data" / "cache" / "macro"                  # tradex.data.macro_history writes these three
FOMC_FILE = MACRO / "fomc_history.csv"
RATES_FILE = MACRO / "policy_rates.csv"
VIX_FILE = MACRO / "vix_daily.csv"
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
            out[s] = b      # OHLCV plus the bid/ask columns the gate measures spreads from
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


def earnings(spec: StrategySpec, cache=None, members=None):
    cal = CsvEarningsCalendar(EARNINGS_DIR)
    syms = _symbols(spec)
    data = us_bars(syms, spec.signal_tf, cache)
    have = [s for s in data if cal.has(s)]
    if not have:
        raise DataUnavailable(f"no earnings-date files for the universe in {EARNINGS_DIR} (one <SYMBOL>.csv with date[,when])")
    bench = us_bars(["SPY"], spec.signal_tf, cache).get("SPY")
    out = {s: panels.with_columns(data[s], earnings_columns(data[s], cal.announcements(s), bench)) for s in have}
    return out, [s for s in syms if s not in out]


def fomc_window(spec: StrategySpec, cache=None, members=None):
    syms = _symbols(spec)
    data = us_bars(syms, spec.signal_tf, cache)
    times = FileEventCalendar(FOMC_FILE, kind="fomc").times()
    cols = {s: event_window_columns(b, times, duration(spec.signal_tf)) for s, b in data.items()}
    return {s: panels.with_columns(b, cols[s]) for s, b in data.items()}, [s for s in syms if s not in data]


def fomc_daily(spec: StrategySpec, cache=None, members=None):
    """Daily bars with close-to-close flags around each scheduled FOMC statement (``fill: next_close`` specs)."""
    syms = _symbols(spec)
    data = us_bars(syms, spec.signal_tf, cache)
    times = FileEventCalendar(FOMC_FILE, kind="fomc").times()
    return {s: panels.with_columns(b, daily_event_columns(b, times)) for s, b in data.items()}, \
        [s for s in syms if s not in data]


def regime_etf(spec: StrategySpec, cache=None, members=None):
    syms = _symbols(spec)
    data = us_bars(syms, spec.signal_tf, cache)
    return _attach(data, _regime_panel(cache)), [s for s in syms if s not in data]


def vix_column(index: pd.DatetimeIndex) -> pd.Series:
    """Cboe VIX close of each daily bar's own session (bars stamped 00:00 New York). The close prints
    at 16:15, after the stock close; the engine acts on a bar's signal at the next open, so it is known in time."""
    if not VIX_FILE.exists():
        raise DataUnavailable(f"no Cboe VIX history at {VIX_FILE} (python -m tradex.data.macro_history vix)")
    from tradex.data.macro_history import load_vix
    vix = load_vix(VIX_FILE)
    days = index.tz_convert("America/New_York").tz_localize(None).normalize()
    return pd.Series(vix.reindex(days).to_numpy(), index=index)


def regime_vix(spec: StrategySpec, cache=None, members=None):
    """The regime columns plus ``vix`` (Cboe VIX close)."""
    data, missing = regime_etf(spec, cache)
    return {s: panels.with_columns(b, {"vix": vix_column(b.index)}) for s, b in data.items()}, missing


def rs_crash(spec: StrategySpec, cache=None, members=None):
    syms = _symbols(spec)
    data = us_bars(syms, spec.signal_tf, cache)
    spy = us_bars([MARKET], "D1", cache).get(MARKET)
    if spy is None:
        raise DataUnavailable("relative strength needs SPY daily bars")
    rg = _regime_panel(cache)
    rsm = pd.DataFrame({s: rs_momentum(b["close"], spy["close"]) for s, b in data.items()})
    ranked = rsm if members is None else rsm.where(
        members.reindex(index=rsm.index, columns=rsm.columns).fillna(False).astype(bool))
    xs = panels.xs_percentile(ranked)
    out = {s: panels.with_columns(b, {"rsm": rsm[s], "rsm_xs": xs[s], "rg_crash": rg["rg_crash"]}) for s, b in data.items()}
    return out, [s for s in syms if s not in data]


def fx_strength(spec: StrategySpec, cache=None, n: int = 63, members=None):
    syms = _symbols(spec)
    data = fx_bars(syms, spec.signal_tf)
    strength = currency_strength({s: b["close"] for s, b in data.items()}, n)
    diff = pd.DataFrame({s: pair_strength_diff(s, strength) for s in data})
    xs = panels.xs_percentile(diff)
    out = {s: panels.with_columns(b, {"cs_diff": diff[s], "cs_xs": xs[s], "ts_mom": b["close"].pct_change(n)})
           for s, b in data.items()}
    return out, [s for s in syms if s not in data]


def fx_carry(spec: StrategySpec, cache=None, members=None):
    syms = _symbols(spec)
    data = fx_bars(syms, spec.signal_tf)
    rates = CsvRateSource(RATES_FILE)
    cols = {s: carry_columns(b, s, rates) for s, b in data.items()}
    xs = panels.xs_percentile(pd.DataFrame({s: c["carry"] for s, c in cols.items()}))
    out = {s: panels.with_columns(b, cols[s].assign(carry_xs=xs[s])) for s, b in data.items()}
    return out, [s for s in syms if s not in data]


def preflight(builder: str) -> None:
    """Raise ``DataUnavailable`` when a calendar or rate file the builder reads is absent, before any bars
    are loaded or screened, so the gate says which input is missing rather than "no bars"."""
    if builder == "earnings" and not (EARNINGS_DIR.is_dir() and any(EARNINGS_DIR.glob("*.csv"))):
        raise DataUnavailable(f"no earnings-date files in {EARNINGS_DIR} (python -m tradex.data.earnings SYMBOL ...)")
    if builder in ("fomc_window", "fomc_daily") and not FOMC_FILE.exists():
        raise DataUnavailable(f"no FOMC event calendar at {FOMC_FILE}")
    if builder in ("fx_strength", "fx_carry") and not (OANDA_CACHE.is_dir() and any(OANDA_CACHE.glob("*.csv"))):
        raise DataUnavailable(f"no Oanda history in {OANDA_CACHE} (run `python -m tradex.research.universe oanda`)")
    if builder == "regime_vix" and not VIX_FILE.exists():
        raise DataUnavailable(f"no Cboe VIX history at {VIX_FILE}")
    if builder == "fx_carry" and not RATES_FILE.exists():
        raise DataUnavailable(f"no rate history at {RATES_FILE}")


BUILDERS = {"earnings": earnings, "fomc_window": fomc_window, "fomc_daily": fomc_daily, "regime_etf": regime_etf, "regime_vix": regime_vix,
            "rs_crash": rs_crash,
            "fx_strength": fx_strength, "fx_carry": fx_carry}
