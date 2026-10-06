"""Stop guardian for venues without resting stop orders (moomoo SIMULATE; protected path).

SIMULATE takes only market and limit orders, so a position's stop and target live in the
adapter (``stop_levels``) and this guardian enforces them: at least once a minute it reads
marks for the symbols with open positions, asks ``venue.check_stops(marks)`` which levels
were crossed, and sends a market exit for each through ``venue.place`` - the order guard,
like every other order. Each position is exited at most once (the adapter records the
fired client ID; only a rejected or cancelled exit re-arms). A missing mark is a fault,
never a guess: that position is simply not checked this minute, and the fault is loud.

Every tick is a ping (``last_ping``, ``on_ping``) so the runtime can alert if the guardian
itself stops; ``alive(now)`` is false after two missed intervals.
"""
from __future__ import annotations

import threading
from typing import Callable, Protocol

import pandas as pd

from tradex.core.interfaces import OrderRequest

MAX_INTERVAL_S = 60.0


class StopVenue(Protocol):
    def stop_levels(self) -> dict[str, dict]: ...
    def check_stops(self, marks: dict[str, float]) -> list: ...
    def mark_fired(self, decision_id: str, client_order_id: str) -> None: ...
    def place(self, req: OrderRequest) -> str: ...


class StopGuardian:
    def __init__(self, venue: StopVenue, marks: Callable[[list[str]], dict[str, float]],
                 on_fault: Callable[[str, str], None], *, interval_s: float = MAX_INTERVAL_S,
                 asset_class: str = "stocks", on_ping: Callable[[pd.Timestamp], None] | None = None,
                 clock: Callable[[], pd.Timestamp] = lambda: pd.Timestamp.now(tz="UTC")):
        self.venue, self.marks, self.on_fault, self.on_ping = venue, marks, on_fault, on_ping
        self.interval_s = min(float(interval_s), MAX_INTERVAL_S)
        self.asset_class, self.clock = asset_class, clock
        self.last_ping: pd.Timestamp | None = None
        self.sent: list[str] = []

    def tick(self) -> list[str]:
        """One pass; returns the client IDs of exits sent."""
        now = self.clock()
        self.last_ping = now
        if self.on_ping:
            self.on_ping(now)
        levels = self.venue.stop_levels()
        syms = sorted({lv["symbol"] for lv in levels.values() if lv.get("stop") is not None or lv.get("target") is not None})
        if not syms:
            return []
        try:
            marks = {s: float(m) for s, m in (self.marks(syms) or {}).items() if m is not None and m == m and m > 0}
        except Exception as exc:  # noqa: BLE001 - no marks this minute is a loud fault, not a crash
            self.on_fault("stop_guardian", f"marks unavailable: {type(exc).__name__}: {exc}")
            return []
        for s in syms:
            if s not in marks:
                self.on_fault("stop_guardian", f"{s}: no live mark; its stop is unwatched this minute")
        out = []
        for trig in self.venue.check_stops(marks):
            req = OrderRequest(trig.client_order_id, trig.decision_id, trig.symbol, self.asset_class, -trig.direction,
                               trig.qty, "market", purpose="exit", book=trig.book)
            try:
                self.venue.place(req)
            except Exception as exc:  # noqa: BLE001 - try again next minute; the fault is loud
                self.on_fault("stop_guardian", f"{trig.decision_id} {trig.reason} exit not sent: "
                                               f"{type(exc).__name__}: {exc}")
                continue
            self.venue.mark_fired(trig.decision_id, trig.client_order_id)
            self.sent.append(trig.client_order_id)
            out.append(trig.client_order_id)
        return out

    def alive(self, now: pd.Timestamp | None = None) -> bool:
        now = now or self.clock()
        return self.last_ping is not None and (now - self.last_ping).total_seconds() <= 2 * self.interval_s

    def run(self, stop: threading.Event) -> None:
        """Tick until ``stop`` is set (the runtime runs this in a daemon thread)."""
        while not stop.is_set():
            try:
                self.tick()
            except Exception as exc:  # noqa: BLE001 - the guardian must outlive any one bad pass
                self.on_fault("stop_guardian", f"tick failed: {type(exc).__name__}: {exc}")
            stop.wait(self.interval_s)
