"""Live marks for the stop guardian of a venue without resting stops (moomoo SIMULATE).

``OpenDMarks(symbols)`` reads OpenD market snapshots (a quote context, never a trade
context) and returns the last price of each symbol whose snapshot is fresh: updated no
more than ``max_age`` ago, inside the US regular session. Anything else is left out, and
the guardian reports a left-out symbol as a fault: a stop is never checked against a
stale or estimated price. Outside the regular session SIMULATE cannot fill a market exit
anyway, so ``us_regular_session`` lets the runtime keep those minutes quiet.
"""
from __future__ import annotations

from typing import Callable

import pandas as pd

from tradex.data.opend import HOST, NY, PORT, us_code

RET_OK = 0


def us_regular_session(ts: pd.Timestamp) -> bool:
    """Mon-Fri 09:30-16:00 New York (exchange holidays are not known here)."""
    ny = ts.tz_convert(NY)
    return ny.weekday() < 5 and (9, 30) <= (ny.hour, ny.minute) < (16, 0)


class OpenDMarks:
    def __init__(self, ctx=None, max_age_s: float = 120.0, host: str = HOST, port: int = PORT,
                 clock: Callable[[], pd.Timestamp] = lambda: pd.Timestamp.now(tz="UTC")):
        self._ctx, self.host, self.port = ctx, host, port
        self.max_age = pd.Timedelta(seconds=max_age_s)
        self.clock = clock

    @property
    def ctx(self):
        if self._ctx is None:
            from moomoo import OpenQuoteContext  # quote context only, never a trade context
            self._ctx = OpenQuoteContext(host=self.host, port=self.port)
        return self._ctx

    def close(self) -> None:
        if self._ctx is not None:
            self._ctx.close()
            self._ctx = None

    def __call__(self, symbols: list[str]) -> dict[str, float]:
        now = self.clock()
        if not symbols or not us_regular_session(now):
            return {}
        ret, df = self.ctx.get_market_snapshot([us_code(s) for s in symbols])
        if ret != RET_OK:
            raise LookupError(f"OpenD get_market_snapshot failed: {df}")
        out = {}
        for r in df.to_dict("records"):
            px, upd = r.get("last_price"), r.get("update_time")
            if px is None or not upd or not float(px) > 0:
                continue
            t = pd.Timestamp(upd)
            t = (t.tz_localize(NY) if t.tzinfo is None else t).tz_convert("UTC")
            if now - t <= self.max_age:
                out[str(r["code"]).removeprefix("US.")] = float(px)
        return out
