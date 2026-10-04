"""Event calendar and blackout rules (recommendation 4).

Earnings, central bank decisions and major data releases move prices in ways no chart
signal predicts. Each event kind has a window around it; a plan touching the window is
vetoed or has its size cut. Calendar rules can only subtract: they never create a trade.

Events load from YAML or CSV files with columns ``time`` (UTC ISO), ``kind``, ``scope``
(a currency like ``USD``, a symbol like ``AAPL``, or ``*``) and optional ``note``.
Weekend gaps for forex are generated, not loaded.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

from tradex.costs.models import split_pair


@dataclass(frozen=True)
class Event:
    time: pd.Timestamp
    kind: str                          # fomc | boj | ecb | boe | cpi | nfp | earnings | ...
    scope: str                         # currency, symbol or "*"
    note: str = ""


@dataclass
class Window:
    before: pd.Timedelta
    after: pd.Timedelta
    action: str = "veto"               # veto | halve


DEFAULT_RULES: dict[str, Window] = {
    "fomc": Window(pd.Timedelta(hours=12), pd.Timedelta(hours=2)),
    "boj": Window(pd.Timedelta(hours=12), pd.Timedelta(hours=2)),
    "ecb": Window(pd.Timedelta(hours=12), pd.Timedelta(hours=2)),
    "boe": Window(pd.Timedelta(hours=12), pd.Timedelta(hours=2)),
    "cpi": Window(pd.Timedelta(hours=2), pd.Timedelta(minutes=30)),
    "nfp": Window(pd.Timedelta(hours=2), pd.Timedelta(minutes=30)),
    "earnings": Window(pd.Timedelta(days=3), pd.Timedelta(hours=12)),
    "weekend": Window(pd.Timedelta(hours=4), pd.Timedelta(0), action="halve"),
}


@dataclass
class EventCalendar:
    events: list[Event] = field(default_factory=list)
    rules: dict[str, Window] = field(default_factory=lambda: dict(DEFAULT_RULES))
    sources: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, paths: list[str | Path]) -> "EventCalendar":
        cal = cls()
        for p in paths:
            cal.add_file(p)
        return cal

    def add_file(self, path: str | Path) -> None:
        path = Path(path)
        if path.suffix in (".yaml", ".yml"):
            doc = yaml.safe_load(path.read_text()) or {}
            rows = doc.get("events", doc if isinstance(doc, list) else [])
        else:
            rows = pd.read_csv(path).to_dict("records")
        for r in rows:
            self.events.append(Event(pd.Timestamp(r["time"]).tz_convert("UTC") if pd.Timestamp(r["time"]).tzinfo
                                     else pd.Timestamp(r["time"], tz="UTC"),
                                     str(r["kind"]).lower(), str(r.get("scope", "*")), str(r.get("note", "") or "")))
        self.events.sort(key=lambda e: e.time)
        self.sources.append(str(path))

    def scopes_for(self, symbol: str, asset_class: str) -> set[str]:
        if asset_class == "forex":
            return set(split_pair(symbol)) | {"*"}
        return {symbol, "USD", "*"}

    def check(self, symbol: str, asset_class: str, start: pd.Timestamp, end: pd.Timestamp
              ) -> tuple[str | None, float, list[Event]]:
        """For a trade expected to be open from ``start`` to ``end``: a veto reason (or None),
        a size factor (1.0 or 0.5), and the events that matter. Earnings only block entries
        whose holding period reaches the event; macro events block entries inside their window."""
        scopes = self.scopes_for(symbol, asset_class)
        hits: list[Event] = []
        factor = 1.0
        veto = None
        for e in self.events:
            if e.scope not in scopes:
                continue
            w = self.rules.get(e.kind)
            if w is None:
                continue
            if e.kind == "earnings":
                inside = start - w.after <= e.time <= end + w.before
            else:
                inside = e.time - w.before <= start <= e.time + w.after
            if not inside:
                continue
            hits.append(e)
            if w.action == "veto" and veto is None:
                veto = f"{e.kind} {e.scope} at {e.time:%Y-%m-%d %H:%M} UTC inside its window"
            elif w.action == "halve":
                factor = min(factor, 0.5)
        if asset_class == "forex" and _crosses_weekend(start, end):
            factor = min(factor, 0.5)
            hits.append(Event(start, "weekend", "*", "held over the weekend: gap risk"))
        return veto, factor, hits

    def upcoming(self, now: pd.Timestamp, days: int = 7) -> list[Event]:
        return [e for e in self.events if now <= e.time <= now + pd.Timedelta(days=days)]


def _crosses_weekend(start: pd.Timestamp, end: pd.Timestamp) -> bool:
    """True when a forex trade open over [start, end] spans the Friday 17:00 New York close."""
    ny_s, ny_e = start.tz_convert("America/New_York"), end.tz_convert("America/New_York")
    d = ny_s.normalize()
    while d <= ny_e:
        if d.weekday() == 4:
            close = d + pd.Timedelta(hours=17)
            if ny_s <= close <= ny_e:
                return True
        d += pd.Timedelta(days=1)
    return False
