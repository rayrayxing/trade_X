"""Injected data interfaces for event and rate features, plus file-backed readers.

The feature builders (``events``, ``carry``) never contain dates or rates. They take an
object that satisfies one of the protocols below, so research reads real files today and a
broker or vendor feed later, and tests pass small fixtures. A reader whose file is absent
raises ``DataUnavailable``; the research gate reports that as "needs data" and runs nothing
on a stand-in.

File layouts (put the files under ``data/``; see research/proposals.md):

  earnings   ``<root>/<SYMBOL>.csv``      columns: date[, when]   (when = bmo | amc | unknown)
  events     ``<path>.csv``               column: time (UTC ISO)  or the YAML in data/calendar
  rates      ``<path>.csv``               columns: date, currency, rate (percent, effective date)
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd
import yaml

WHEN = ("bmo", "amc", "unknown")


class DataUnavailable(RuntimeError):
    """A required input (calendar, rate history) is not on disk. Maps to "needs data"."""


@runtime_checkable
class EarningsCalendar(Protocol):
    def announcements(self, symbol: str) -> pd.DataFrame:
        """One row per announcement: ``date`` (session date, tz-naive) and ``when`` (bmo | amc | unknown)."""


@runtime_checkable
class EventCalendar(Protocol):
    def times(self) -> pd.DatetimeIndex:
        """Announcement instants, UTC, sorted."""


@runtime_checkable
class RateSource(Protocol):
    def rates(self, ccy: str, index: pd.DatetimeIndex) -> pd.Series:
        """Annual policy/financing rate of ``ccy`` as a decimal, as known at each timestamp of ``index``
        (a step function: the latest rate whose effective date is at or before the timestamp, NaN before the first)."""


class CsvEarningsCalendar:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def path(self, symbol: str) -> Path:
        return self.root / f"{symbol}.csv"

    def has(self, symbol: str) -> bool:
        return self.path(symbol).exists()

    def announcements(self, symbol: str) -> pd.DataFrame:
        p = self.path(symbol)
        if not p.exists():
            raise DataUnavailable(f"no earnings dates for {symbol}: expected {p}")
        df = pd.read_csv(p)
        df.columns = [str(c).strip().lower() for c in df.columns]
        if "date" not in df.columns:
            raise ValueError(f"{p}: needs a 'date' column")
        out = pd.DataFrame({"date": pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()})
        when = df["when"].fillna("unknown").astype(str).str.strip().str.lower() if "when" in df.columns else "unknown"
        out["when"] = when
        bad = sorted(set(out["when"]) - set(WHEN)) if not isinstance(when, str) else []
        if bad:
            raise ValueError(f"{p}: 'when' must be one of {WHEN}, got {bad}")
        return out.drop_duplicates("date").sort_values("date").reset_index(drop=True)


class FileEventCalendar:
    """Event instants from a CSV (column ``time``) or from the YAML in data/calendar (``events:``, filter by ``kind``)."""

    def __init__(self, path: str | Path, kind: str | None = None):
        self.path, self.kind = Path(path), kind

    def times(self) -> pd.DatetimeIndex:
        if not self.path.exists():
            raise DataUnavailable(f"no event calendar at {self.path}")
        if self.path.suffix in (".yaml", ".yml"):
            rows = (yaml.safe_load(self.path.read_text()) or {}).get("events", [])
            ts = [r["time"] for r in rows if self.kind is None or r.get("kind") == self.kind]
        else:
            df = pd.read_csv(self.path)
            df.columns = [str(c).strip().lower() for c in df.columns]
            if "time" not in df.columns:
                raise ValueError(f"{self.path}: needs a 'time' column")
            if self.kind is not None and "kind" in df.columns:
                df = df[df["kind"] == self.kind]
            ts = list(df["time"])
        if not ts:
            raise DataUnavailable(f"{self.path} has no {self.kind or ''} events")
        return pd.DatetimeIndex(sorted(pd.to_datetime(ts, utc=True))).astype("datetime64[ns, UTC]")


class CsvRateSource:
    """Policy or financing rates from ``date,currency,rate`` (percent), same layout as costs/policy_rates.csv."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._by_ccy: dict[str, pd.Series] | None = None

    def _load(self) -> dict[str, pd.Series]:
        if self._by_ccy is None:
            if not self.path.exists():
                raise DataUnavailable(f"no rate history at {self.path}")
            t = pd.read_csv(self.path, comment="#")
            t.columns = [str(c).strip().lower() for c in t.columns]
            t["date"] = pd.to_datetime(t["date"], utc=True)
            self._by_ccy = {c: g.sort_values("date").drop_duplicates("date", keep="last").set_index("date")["rate"] / 100.0
                            for c, g in t.groupby("currency")}
        return self._by_ccy

    def currencies(self) -> list[str]:
        return sorted(self._load())

    def rates(self, ccy: str, index: pd.DatetimeIndex) -> pd.Series:
        table = self._load()
        if ccy not in table:
            raise DataUnavailable(f"{self.path} has no rates for {ccy}")
        s = table[ccy]
        idx = pd.DatetimeIndex(index)
        merged = s.reindex(s.index.union(idx)).ffill().reindex(idx)
        return merged.astype(float)
