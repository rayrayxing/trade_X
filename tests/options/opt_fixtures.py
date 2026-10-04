"""Test-only fixtures for the options lane: a Black-Scholes chain provider over given bars and
hand-built signal frames. Synthetic by design; nothing outside tests/ may import this."""
from __future__ import annotations

import datetime as dt
import math

import pandas as pd

from tradex.options.backtest import bar_date
from tradex.options.contract import Greeks, OptionContract, OptionQuote, Right
from tradex.options.pricing import bs_greeks, bs_price
from tradex.options.providers import Dividend, EarningsInfo
from tradex.execution.checks import ShortInfo
from tradex.strategy.spec import SignalFrame

UTC = "UTC"


def bars_from_closes(closes, start="2024-01-02", gap_open=None, rng=0.5, volume=1e6):
    """Business-day bars (00:00 UTC stamps). Open = previous close unless ``gap_open`` maps i -> open."""
    idx = pd.bdate_range(start, periods=len(closes), tz=UTC)
    closes = [float(c) for c in closes]
    opens = [closes[0]] + closes[:-1]
    for i, v in (gap_open or {}).items():
        opens[i] = float(v)
    hi = [max(o, c) + rng for o, c in zip(opens, closes)]
    lo = [min(o, c) - rng for o, c in zip(opens, closes)]
    df = pd.DataFrame({"open": opens, "high": hi, "low": lo, "close": closes, "volume": volume}, index=idx)
    df.index = df.index.astype("datetime64[ns, UTC]")
    return df


def bars_with_extremes(closes, highs=None, lows=None, opens=None, start="2024-01-02"):
    df = bars_from_closes(closes, start)
    for col, m in (("high", highs), ("low", lows), ("open", opens)):
        for i, v in (m or {}).items():
            df.iloc[i, df.columns.get_loc(col)] = float(v)
    return df


class BSChain:
    """Chain provider that prices every contract with Black-Scholes at the bar's open/close underlying.

    Weekly Friday expiries up to ``horizon_days`` out, strikes on a grid around the underlying.
    Spread is ``spread_pct`` of mid (full width); open interest and volume are large.
    """

    recorded = False

    def __init__(self, data: dict[str, pd.DataFrame], iv=0.30, spread_pct=0.04, horizon_days=75, rate=0.0,
                 oi=5000, volume=500, quote_filter=None):
        self.data, self.iv, self.spread_pct, self.horizon = data, iv, spread_pct, horizon_days
        self.rate, self.oi, self.volume = rate, oi, volume
        self.quote_filter = quote_filter          # callable(contract, ts, at) -> bool, to simulate gaps in data
        self.calls = {"chain": 0, "quote": 0}

    def _spot(self, underlying, ts, at):
        df = self.data[underlying]
        if ts not in df.index:
            return None
        row = df.loc[ts]
        return float(row["open"] if at == "open" else row["close"])

    def _moment(self, ts, at):
        d = bar_date(ts)
        t = dt.time(9, 30) if at == "open" else dt.time(16, 0)
        return pd.Timestamp(dt.datetime.combine(d, t)).tz_localize("America/New_York").tz_convert("UTC")

    def _iv(self, ts):
        return self.iv(ts) if callable(self.iv) else self.iv

    def _quote(self, c: OptionContract, ts, at, spot) -> OptionQuote | None:
        when = self._moment(ts, at)
        t = c.years_to_expiry(when)
        iv = self._iv(ts)
        px = bs_price(spot, c.strike, t, iv, c.right, self.rate)
        g = bs_greeks(spot, c.strike, t, iv, c.right, self.rate)
        half = 0.5 * self.spread_pct * px
        bid, ask = max(px - half, 0.0), px + half
        if self.quote_filter and not self.quote_filter(c, ts, at):
            return None
        return OptionQuote(c, ts, round(bid, 4), round(ask, 4), round(px, 4), self.volume, self.oi,
                           Greeks(g["delta"], g["gamma"], g["theta"], g["vega"], g["rho"], iv), spot)

    def chain(self, underlying, ts, at="close"):
        self.calls["chain"] += 1
        spot = self._spot(underlying, ts, at)
        if spot is None:
            return []
        step = 1.0 if spot < 50 else 2.5 if spot < 150 else 5.0
        base = round(spot / step) * step
        strikes = [base + k * step for k in range(-12, 13) if base + k * step > 0]
        d0 = bar_date(ts)
        expiries = [d0 + dt.timedelta(days=k) for k in range(0, self.horizon + 1)
                    if (d0 + dt.timedelta(days=k)).weekday() == 4 and k >= 1]
        out = []
        for e in expiries:
            for k in strikes:
                for r in (Right.CALL, Right.PUT):
                    q = self._quote(OptionContract(underlying, e, k, r), ts, at, spot)
                    if q is not None:
                        out.append(q)
        return out

    def quote(self, contract, ts, at="close"):
        self.calls["quote"] += 1
        spot = self._spot(contract.underlying, ts, at)
        if spot is None:
            return None
        return self._quote(contract, ts, at, spot)


def signals(n, entries=(), exits=(), atr=1.0, side="long"):
    """Hand-built SignalFrame over ``n`` business-day bars: entry/exit flags at the given positions."""
    idx = pd.bdate_range("2024-01-02", periods=n, tz=UTC)
    idx = idx.astype("datetime64[ns, UTC]")
    flag = lambda pos: pd.Series([i in set(pos) for i in range(n)], index=idx)   # noqa: E731
    false = pd.Series(False, index=idx)
    le, se = (flag(entries), false) if side == "long" else (false, flag(entries))
    lx, sx = (flag(exits), false) if side == "long" else (false, flag(exits))
    return SignalFrame(le, se, lx, sx, pd.Series(float(atr), index=idx), {}, [])


class FakeRisk:
    """UnderlyingRiskProvider: fixed answers, or a function of the query date."""

    def __init__(self, earnings=EarningsInfo(None), short=ShortInfo(short_interest_pct_float=3.0, days_to_cover=1.0)):
        self.earnings, self.short = earnings, short

    def next_earnings(self, underlying, asof):
        return self.earnings(asof) if callable(self.earnings) else self.earnings

    def short_info(self, underlying, asof):
        return self.short(asof) if callable(self.short) else self.short


class FakeDividends:
    def __init__(self, dividend: Dividend | None):
        self.dividend = dividend

    def next_ex_dividend(self, underlying, asof):
        return self.dividend


def approx(a, b, tol=1e-6):
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)
