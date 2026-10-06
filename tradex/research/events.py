"""Event features from an injected calendar: earnings reaction and drift, and announcement windows.

Both take calendars as arguments (``tradex.research.sources``), so no dates live in this
file. Every column at bar t uses only bars at or before t. The one forward-looking input is
the calendar itself: ``ed_days_to`` assumes the announcement date was known in advance,
which holds for scheduled earnings and central-bank meetings but not for a date that was
later moved.

Earnings columns (daily bars; the "reaction session" is the first session that trades on the news):

  ed_ret    reaction-session close-to-close return, minus the benchmark's if one is given
  ed_gap    reaction-session open over the previous close
  ed_z      ed_ret over the standard deviation of daily returns in the ``vol_n`` sessions before it
  ed_volx   reaction-session volume over its prior ``volume_n``-session average
  ed_age    sessions since the reaction session (0 on it); NaN before the first event
  ed_post   return since the reaction session's close
  ed_days_to sessions until the next announcement session (0 on the announcement day)

The first five hold from the bar where the reaction is known until the next event. When the
calendar says nothing about before-open or after-close (``when`` = unknown) the reaction is
the larger-moving of the announcement session and the next one, which is only known at the
close of the later session, so the columns start there.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NY = "America/New_York"
EARNINGS_COLUMNS = ["ed_ret", "ed_gap", "ed_z", "ed_volx", "ed_age", "ed_post", "ed_days_to"]
WINDOW_COLUMNS = ["evt_enter", "evt_leave"]


def session_dates(index: pd.DatetimeIndex, tz: str = NY) -> pd.DatetimeIndex:
    """Local calendar date of each bar's open time (tz-naive). Daily bars stamped at New York midnight map to their own day."""
    return index.tz_convert(tz).tz_localize(None).normalize()


def reaction_sessions(bars: pd.DataFrame, announcements: pd.DataFrame, tz: str = NY) -> pd.DataFrame:
    """For each announcement: ``announce`` (bar position of the announcement session), ``reaction`` and ``known`` positions."""
    dates = session_dates(bars.index, tz).to_numpy()
    close = bars["close"].to_numpy(float)
    n = len(bars)
    rows = []
    for d, when in zip(pd.DatetimeIndex(announcements["date"]).to_numpy(), announcements["when"]):
        a = int(dates.searchsorted(d, side="left"))
        if a >= n:
            continue
        on_session = dates[a] == d
        if not on_session or when == "bmo":
            r = k = a
        elif when == "amc":
            r = k = a + 1
        else:  # unknown timing: the bigger mover of the two sessions, known once both have closed
            b = a + 1
            if a < 1 or b >= n:
                continue
            ra, rb = abs(close[a] / close[a - 1] - 1), abs(close[b] / close[a] - 1)
            r, k = (a if ra >= rb else b), b
        if r < 1 or k >= n:
            continue
        rows.append((a, r, k))
    out = pd.DataFrame(rows, columns=["announce", "reaction", "known"])
    out = out.sort_values("known").drop_duplicates("reaction")
    keep, last = [], -1
    for row in out.itertuples():
        if row.known > last:
            keep.append(row.Index)
            last = row.known
    return out.loc[keep].reset_index(drop=True)


def earnings_columns(bars: pd.DataFrame, announcements: pd.DataFrame, benchmark: pd.DataFrame | None = None,
                     vol_n: int = 60, volume_n: int = 20, tz: str = NY) -> pd.DataFrame:
    """Post-earnings reaction and drift columns for one symbol's daily bars. See the module docstring."""
    n = len(bars)
    cols = {c: np.full(n, np.nan) for c in EARNINGS_COLUMNS}
    close, open_, volume = (bars[c].to_numpy(float) for c in ("close", "open", "volume"))
    ret = pd.Series(close, index=bars.index).pct_change()
    bench = (benchmark["close"].reindex(bars.index, method="ffill").pct_change().to_numpy(float)
             if benchmark is not None else np.zeros(n))
    ev = reaction_sessions(bars, announcements, tz)
    ends = list(ev["known"].iloc[1:]) + [n]
    for (a, r, k), end in zip(ev[["announce", "reaction", "known"]].itertuples(index=False), ends):
        sd = ret.iloc[max(1, r - vol_n):r].std() if r - 1 >= 2 else np.nan
        adj = ret.iloc[r] - (0.0 if np.isnan(bench[r]) else bench[r])
        prior_vol = volume[max(0, r - volume_n):r]
        seg = slice(k, end)
        cols["ed_ret"][seg] = adj
        cols["ed_gap"][seg] = open_[r] / close[r - 1] - 1
        cols["ed_z"][seg] = adj / sd if sd and sd > 0 else np.nan
        cols["ed_volx"][seg] = volume[r] / prior_vol.mean() if len(prior_vol) and prior_vol.mean() > 0 else np.nan
        cols["ed_age"][seg] = np.arange(k, end) - r
        cols["ed_post"][seg] = close[k:end] / close[r] - 1
    if len(ev):
        pos = ev["announce"].to_numpy()
        t = np.arange(n)
        nxt = np.searchsorted(pos, t, side="left")
        ok = nxt < len(pos)
        cols["ed_days_to"][ok] = pos[nxt[ok]] - t[ok]
    return pd.DataFrame(cols, index=bars.index)


def event_window_columns(bars: pd.DataFrame, times: pd.DatetimeIndex, bar_td: pd.Timedelta,
                         lead: pd.Timedelta = pd.Timedelta(hours=24), max_gap: pd.Timedelta = pd.Timedelta(hours=18)
                         ) -> pd.DataFrame:
    """Entry and exit flags for holding through the ``lead`` before each announcement (pre-FOMC drift).

    ``evt_enter`` is 1 on the last bar that closes at or before ``time - lead``; the engine fills the
    entry at the next bar's open. ``evt_leave`` is 1 on the last bar that closes at or before ``time``;
    the exit fills at the next open, just before the announcement. Each flag is skipped when the bars
    before it are missing by more than ``max_gap`` or no later bar exists, so a data hole cannot move the window.
    """
    n = len(bars)
    enter, leave = np.zeros(n), np.zeros(n)
    close_t = bars.index + bar_td
    for e in pd.DatetimeIndex(times):
        # a flag needs a later bar in the data, otherwise the bar that closes after the target may simply not have arrived
        for target, out in ((e - lead, enter), (e, leave)):
            i = int(close_t.searchsorted(target, side="right")) - 1
            if 0 <= i < n - 1 and target - close_t[i] <= max_gap:
                out[i] = 1.0
    return pd.DataFrame({"evt_enter": enter, "evt_leave": leave}, index=bars.index)


def daily_event_columns(bars: pd.DataFrame, times: pd.DatetimeIndex, tz: str = NY) -> pd.DataFrame:
    """Close-to-close flags on daily bars for holding from the close of the session before each
    announcement day to the close of the announcement day (pre-FOMC drift on D1).

    For a strategy with ``fill: next_close`` (orders fill at the next bar's close): ``evt_enter``
    is 1 two sessions before the announcement session, so the entry fills at the previous
    session's close; ``evt_leave`` is 1 on the session before, so the exit fills at the
    announcement session's close. The flags read only the published schedule and the
    trading calendar (sessions are known in advance; the bars stand in for the exchange
    calendar), never a price. An announcement on a day with no bar is skipped, and so is one
    whose sessions are not all in the data yet.
    """
    n = len(bars)
    enter, leave = np.zeros(n), np.zeros(n)
    dates = session_dates(bars.index, tz)
    for e in pd.DatetimeIndex(times):
        d = pd.Timestamp(e).tz_convert(tz).tz_localize(None).normalize()
        t = int(dates.searchsorted(d, side="left"))
        if t >= n or dates[t] != d or t < 2:
            continue
        enter[t - 2], leave[t - 1] = 1.0, 1.0
    return pd.DataFrame({"evt_enter": enter, "evt_leave": leave}, index=bars.index)
