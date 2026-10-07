"""Live research columns: the ``data.column`` inputs of the FX research strategies, built at each close.

Research (tradex.research.builders) attaches columns to Oanda daily candles (17:00 New York
alignment) across a strategy's universe; the live bar store only carries OHLCV. This builder keeps the
same daily candles per pair (a ``HISTORY_YEARS`` warm-up from the history provider, then polled for
each new 17:00 New York close) and calls the research functions themselves on them, so a live column
at a close is what research computes for that bar from the data known then (tests/test_live_columns.py
replays a series and compares bar for bar).

  fx_strength  cs_diff, cs_xs, ts_mom (63 bars)       from all pairs of the universe
  fx_carry     carry, carry_vol, rate_chg, ts_mom (100 bars), carry_trend, fxvol_pct, carry_xs
               from all pairs plus official policy rates (tradex.runtime.policy_rates)

``ts_mom`` means a different horizon in each, so a strategy is fed by the one builder whose
columns cover everything it reads (a spec reading only ``ts_mom`` is ambiguous and held back).

Nothing is filled. A strategy's columns are blocked, with one Health fault when it starts and one
when it clears, while any pair of its universe lacks the daily bar of the latest close (cross-sectional
ranks need every pair at the same close), or, for carry, while a currency's official rate is missing
or stale or disagrees with Oanda's financing. A blocked strategy simply has no signal at that close
(``SignalCache`` asks ``attach``, which returns None).

The strategy itself runs on the bar store's own signal bars; each of those bars reads the columns of
the latest daily candle that had closed by its close (an as-of join, never ahead).
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from tradex.research.builders import fx_carry_columns, fx_strength_columns
from tradex.research.carry import CARRY_COLUMNS
from tradex.strategy.spec import StrategySpec
from tradex.timeframes import OHLCV, duration

NY = "America/New_York"
STRENGTH = frozenset({"cs_diff", "cs_xs", "ts_mom"})
CARRY = frozenset(CARRY_COLUMNS) | {"carry_xs"}
BUILDERS: dict[str, frozenset[str]] = {"fx_strength": STRENGTH, "fx_carry": CARRY}
NEEDS_RATES = {"fx_carry"}
LIVE_COLUMNS: frozenset[str] = STRENGTH | CARRY
LIVE_TFS = ("D1",)                        # research builds these columns on Oanda daily candles
HISTORY_YEARS = 10                        # the research cache holds ten years (OandaHistory.fetch)
GRACE = pd.Timedelta(hours=1)             # a daily candle is required one hour after its 17:00 New York close
HealthFn = Callable[[str, bool, str, pd.Timestamp], None]


def research_columns(spec: StrategySpec) -> set[str]:
    return {f["name"] for f in spec.features.values() if f.get("fn") == "data.column" and f.get("name")}


def builder_for(spec: StrategySpec) -> str | None:
    """The one live builder whose columns cover everything ``spec`` reads, or None."""
    need = research_columns(spec)
    if not need or spec.asset_class != "forex" or spec.signal_tf not in LIVE_TFS:
        return None
    fits = [b for b, cols in BUILDERS.items() if need <= cols]
    return fits[0] if len(fits) == 1 else None


def daily_close(ts: pd.Timestamp) -> pd.Timestamp:
    """The latest Oanda daily-candle close (17:00 New York, Monday to Friday) at or before ``ts``."""
    ny = ts.tz_convert(NY)
    for k in range(8):
        d = ny.date() - dt.timedelta(days=k)
        c = pd.Timestamp(dt.datetime.combine(d, dt.time(17, 0))).tz_localize(NY)
        if c <= ny and c.weekday() < 5:
            return c.tz_convert("UTC")
    raise RuntimeError("no daily close found")


def _pairs(spec: StrategySpec) -> tuple[str, ...]:
    return tuple(sorted(u for u in spec.universe if not u.startswith("$")))


@dataclass
class _Group:
    builder: str
    tf: str
    pairs: tuple[str, ...]
    specs: list[str] = field(default_factory=list)
    cols: dict[str, pd.DataFrame] = field(default_factory=dict)
    asof: pd.Timestamp | None = None      # close of the latest daily candle the columns include
    fault: str | None = None
    built: tuple = ()

    @property
    def currencies(self) -> list[str]:
        return sorted({c for p in self.pairs for c in p.split("_")})


class LiveColumns:
    def __init__(self, specs: list[StrategySpec], provider: Any, rates: Any = None,
                 years: int = HISTORY_YEARS, grace: pd.Timedelta = GRACE):
        self.provider, self.rates, self.years, self.grace = provider, rates, years, grace
        self.groups: dict[tuple, _Group] = {}
        self.by_spec: dict[str, _Group] = {}
        for s in specs:
            b = builder_for(s)
            if b is None:
                continue
            g = self.groups.setdefault((b, s.signal_tf, _pairs(s)), _Group(b, s.signal_tf, _pairs(s)))
            g.specs.append(s.id)
            self.by_spec[s.id] = g
        self.frames: dict[tuple[str, str], pd.DataFrame] = {}
        self.poll_errors: dict[tuple[str, str], str] = {}
        self.health: HealthFn | None = None
        self._said: dict[tuple, str | None] = {}
        self._last_poll: pd.Timestamp | None = None

    # --- what it needs ---------------------------------------------------------------------------

    def serves(self, spec: StrategySpec) -> bool:
        return spec.id in self.by_spec

    @property
    def series(self) -> list[tuple[str, str]]:
        return sorted({(p, g.tf) for g in self.groups.values() for p in g.pairs})

    @property
    def currencies(self) -> list[str]:
        return sorted({c for g in self.groups.values() if g.builder in NEEDS_RATES for c in g.currencies})

    @property
    def rate_pairs(self) -> list[str]:
        return sorted({p for g in self.groups.values() if g.builder in NEEDS_RATES for p in g.pairs})

    def describe(self) -> str:
        parts = [f"{g.builder} ({len(g.pairs)} pairs {g.tf}) for {', '.join(g.specs)}" for g in self.groups.values()]
        rates = " and official policy rates (refreshed daily; missing or stale: blocked)" if self.currencies else ""
        return "; ".join(parts) + f" from Oanda daily candles ({self.years}y warm-up, polled at each 17:00 New York close){rates}"

    # --- data in -----------------------------------------------------------------------------------

    def warm(self, now: pd.Timestamp) -> None:
        start = now - pd.Timedelta(days=365 * self.years)
        for pair, tf in self.series:
            self._fetch(pair, tf, start, now)
        self._last_poll = now

    def bind(self, store, health: HealthFn) -> None:
        self.health = health
        store.research = self

    def _fetch(self, pair: str, tf: str, start: pd.Timestamp, end: pd.Timestamp) -> None:
        try:
            df = self.provider.get_bars(pair, tf, start.isoformat(), end.isoformat())
        except Exception as exc:  # noqa: BLE001 - one pair's poll failing blocks its strategies, nothing else
            self.poll_errors[(pair, tf)] = f"{type(exc).__name__}: {exc}"
            return
        self.poll_errors.pop((pair, tf), None)
        if df is None or not len(df):
            return
        df = df[df.index + duration(tf) <= end][OHLCV]
        old = self.frames.get((pair, tf))
        new = df if old is None else pd.concat([old, df])
        self.frames[(pair, tf)] = new[~new.index.duplicated(keep="last")].sort_index()

    def before_close(self, tf: str, ts: pd.Timestamp) -> None:
        if self._last_poll is None or ts > self._last_poll:   # several timeframes close at once: poll once
            self._last_poll = ts
            want = daily_close(ts)
            for pair, ctf in self.series:
                have = self.frames.get((pair, ctf))
                if have is None or not len(have):
                    self._fetch(pair, ctf, ts - pd.Timedelta(days=365 * self.years), ts)
                elif have.index[-1] + duration(ctf) < want:
                    self._fetch(pair, ctf, have.index[-1], ts)
        self.update(ts)

    # --- columns -----------------------------------------------------------------------------------

    def _fault(self, g: _Group, required: pd.Timestamp, now: pd.Timestamp) -> str | None:
        d = duration(g.tf)
        missing = [p for p in g.pairs if (f := self.frames.get((p, g.tf))) is None or (required - d) not in f.index]
        if missing:
            why = "; ".join(f"{p}: {self.poll_errors[(p, g.tf)]}" for p in missing if (p, g.tf) in self.poll_errors)
            return (f"no {g.tf} candle closing {required:%Y-%m-%d %H:%M}Z for {', '.join(missing)}"
                    + (f" ({why})" if why else ""))
        if g.builder in NEEDS_RATES and self.rates is None:
            return "no official policy-rate source"
        if g.builder in NEEDS_RATES:
            probs = [p for p in (self.rates.problem(c, now) for c in g.currencies) if p]
            probs += [x for x in (self.rates.disagreement(p) for p in g.pairs) if x]
            if probs:
                return "; ".join(probs)
        return None

    def _compute(self, g: _Group, required: pd.Timestamp) -> dict[str, pd.DataFrame]:
        d = duration(g.tf)
        data = {p: (f := self.frames[(p, g.tf)])[f.index + d <= required] for p in g.pairs}
        if g.builder == "fx_strength":
            return {p: pd.DataFrame(c) for p, c in fx_strength_columns(data).items()}
        return fx_carry_columns(data, self.rates)

    def update(self, ts: pd.Timestamp) -> None:
        """Rebuild each group's columns for the latest required daily close, or block it."""
        required = daily_close(ts - self.grace)
        for key, g in self.groups.items():
            g.fault = self._fault(g, required, ts)
            if g.fault is None:
                stamp = (required, self.rates.version if g.builder in NEEDS_RATES else None)
                if g.built != stamp:
                    g.cols, g.asof, g.built = self._compute(g, required), required, stamp
            if self.health is not None and self._said.get(key) != g.fault:
                who = ", ".join(g.specs)
                if g.fault:
                    self.health("columns", False, f"{who} blocked: {g.fault}", ts)
                else:
                    self.health("columns", True, f"{who} columns built again ({g.builder}, {g.tf} close "
                                                 f"{required:%Y-%m-%d %H:%M}Z)", ts)
                self._said[key] = g.fault

    def latest(self, spec_id: str, pair: str) -> pd.Series | None:
        """The columns of ``pair`` at the group's latest close (None when blocked or not built)."""
        g = self.by_spec[spec_id]
        if g.fault or g.asof is None or pair not in g.cols:
            return None
        return g.cols[pair].iloc[-1]

    def attach(self, spec: StrategySpec, symbol: str, bars: pd.DataFrame, close_t: pd.Timestamp) -> pd.DataFrame | None:
        """``bars`` (the strategy's signal bars up to ``close_t``) with its columns, as of the latest daily candle
        closed by each bar's close; None when the strategy is blocked at ``close_t``."""
        g = self.by_spec[spec.id]
        if g.fault or g.asof is None or g.asof != daily_close(close_t - self.grace) or symbol not in g.cols:
            return None
        src = g.cols[symbol]
        right = src.set_axis(src.index + duration(g.tf)).rename_axis("_t").reset_index()
        left = pd.DataFrame({"_t": bars.index + duration(spec.signal_tf)})
        if left.empty:
            return bars.assign(**{c: pd.Series(dtype=float) for c in src.columns})
        right["_t"] = right["_t"].astype(left["_t"].dtype)
        merged = pd.merge_asof(left, right, on="_t", direction="backward")
        out = bars.copy()
        for c in src.columns:
            out[c] = merged[c].to_numpy()
        return out

    def checks(self) -> list[tuple[str, str, str]]:
        out = []
        for g in self.groups.values():
            name = f"columns:{g.builder}"
            if g.fault:
                out.append((name, "skip", f"{', '.join(g.specs)} blocked until fixed: {g.fault}"))
            elif g.asof is not None:
                out.append((name, "ok", f"built to the {g.tf} close {g.asof:%Y-%m-%d %H:%M}Z for {', '.join(g.specs)}"))
        return out
