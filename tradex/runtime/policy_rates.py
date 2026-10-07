"""Official policy rates for the live carry columns, refreshed daily from the central banks.

The research gate builds ``carry`` and ``rate_chg`` from tradex.data.macro_history's official series
(Fed, ECB, BoE, RBA, BoC, SNB; BoJ and RBNZ via the BIS). Live reads the same series through the
same parsers, fetched again at most every ``refresh_every`` into its own cache, so a live column at a
close equals what research computes from the table as known then. Nothing is typed in: a currency
whose series has never been fetched, or whose last successful fetch is more than ``max_age`` old on a
business day, is a ``problem`` and the strategies that need it are blocked.

Optionally each differential is cross-checked against Oanda's current per-instrument financing
(read-only): Oanda charges ``diff - fee`` long and ``-diff - fee`` short, so ``(long - short) / 2`` is its
differential; a gap over ``tolerance`` is a problem for that pair.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from tradex.data import macro_history as mh
from tradex.research.sources import rate_steps, rates_at

LIVE_CACHE = mh.ROOT / "data" / "cache" / "macro_live"
SINCE = "2000-01-01"                      # the research table starts here too (macro_history.rates_table)


def business_day(ts: pd.Timestamp) -> bool:
    return ts.tz_convert("UTC").weekday() < 5


class LivePolicyRates:
    """A ``RateSource`` (``rates(ccy, index)``) over the official series, with freshness per currency.
    ``refresh`` runs on a background thread (``run``); the core's thread only reads."""

    def __init__(self, currencies: Iterable[str], pairs: Iterable[str] = (), cache: str | Path = LIVE_CACHE,
                 get: Callable[[str], bytes] = mh.http_get,
                 clock: Callable[[], pd.Timestamp] = lambda: pd.Timestamp.now(tz="UTC"),
                 refresh_every: pd.Timedelta = pd.Timedelta(days=1), max_age: pd.Timedelta = pd.Timedelta(days=3),
                 financing: Callable[[list[str]], dict[str, tuple[float, float]]] | None = None,
                 tolerance: float = 0.015, retry_every: pd.Timedelta = pd.Timedelta(hours=1)):
        self.currencies = sorted(set(currencies))
        self.pairs = sorted(set(pairs))
        self.cache, self.get, self.clock = Path(cache), get, clock
        self.refresh_every, self.max_age, self.retry_every = refresh_every, max_age, retry_every
        self.financing, self.tolerance = financing, tolerance
        self._lock = threading.Lock()
        self._steps: dict[str, pd.Series] = {}
        self._fetched: dict[str, pd.Timestamp] = {}
        self._errors: dict[str, str] = {c: "no official series is known for it" for c in self.currencies
                                        if c not in mh.RATE_FILES}
        self._gaps: dict[str, str] = {}
        self._last_try: pd.Timestamp | None = None
        self.last_errors: list[str] = []
        self.version = 0                  # bumps whenever the rates or the cross-check change: rebuild columns

    # --- reading -------------------------------------------------------------------------------

    def rates(self, ccy: str, index: pd.DatetimeIndex) -> pd.Series:
        with self._lock:
            s = self._steps.get(ccy)
        if s is None:
            return pd.Series(float("nan"), index=pd.DatetimeIndex(index))
        return rates_at(s, index)

    def problem(self, ccy: str, now: pd.Timestamp) -> str | None:
        """Why ``ccy`` cannot be used at ``now`` (missing or stale), or None."""
        with self._lock:
            got, err = self._fetched.get(ccy), self._errors.get(ccy)
        if got is None:
            return f"no official {ccy} policy rate" + (f" ({err})" if err else "")
        if business_day(now) and now - got > self.max_age:
            return f"official {ccy} policy rate is stale: last fetched {got:%Y-%m-%d %H:%M}Z" + (f" ({err})" if err else "")
        return None

    def disagreement(self, pair: str) -> str | None:
        with self._lock:
            return self._gaps.get(pair)

    def fetched_at(self, ccy: str) -> pd.Timestamp | None:
        with self._lock:
            return self._fetched.get(ccy)

    # --- refreshing ----------------------------------------------------------------------------

    def load_cached(self) -> None:
        """Use raw files already in the cache, dated by their recorded fetch time (no network)."""
        for ccy in self.currencies:
            if ccy in mh.RATE_FILES:
                self._read(ccy)

    def _read(self, ccy: str) -> bool:
        files = mh.RATE_FILES[ccy]
        if not all((self.cache / n).exists() for n in files):
            return False
        try:
            fetched = min(pd.Timestamp(json.loads((self.cache / f"{n}.source.json").read_text())["fetched_at"])
                          for n in files)
            s = mh.currency_series(ccy, self.cache)
        except Exception as exc:  # noqa: BLE001 - a broken file is a missing rate, not a crash
            with self._lock:
                self._errors[ccy] = f"cached series unreadable: {type(exc).__name__}: {exc}"
            return False
        s = s[s.index >= SINCE]
        table = pd.DataFrame({"date": s.index.strftime("%Y-%m-%d"), "currency": ccy, "rate": s.to_numpy()})
        steps = rate_steps(table).get(ccy) if len(table) else None
        if steps is None or steps.empty:
            with self._lock:
                self._errors[ccy] = "official series is empty"
            return False
        with self._lock:
            old = self._steps.get(ccy)
            self._steps[ccy], self._fetched[ccy] = steps, fetched
            if old is None or not old.equals(steps):
                self.version += 1
        return True

    def _fetch(self, name: str, now: pd.Timestamp) -> None:
        """One raw file from its publisher, with its URL and fetch time beside it (macro_history's layout)."""
        body = self.get(mh.RAW[name])
        self.cache.mkdir(parents=True, exist_ok=True)
        (self.cache / name).write_bytes(body)
        (self.cache / f"{name}.source.json").write_text(json.dumps(
            {"url": mh.RAW[name], "fetched_at": now.isoformat(timespec="seconds"), "bytes": len(body)}))

    def due(self, now: pd.Timestamp) -> bool:
        if self._last_try is not None and now - self._last_try < self.retry_every:
            return False
        with self._lock:
            got = [self._fetched.get(c) for c in self.currencies if c in mh.RATE_FILES]
        return any(t is None or now - t >= self.refresh_every for t in got)

    def refresh(self, now: pd.Timestamp | None = None) -> list[str]:
        """Fetch every currency's official series again; a failure keeps the last good one (and its age)."""
        now = now if now is not None else self.clock()
        self._last_try = now
        errors = []
        for ccy in self.currencies:
            if ccy not in mh.RATE_FILES:
                errors.append(f"{ccy}: {self._errors[ccy]}")
                continue
            try:
                for n in mh.RATE_FILES[ccy]:
                    self._fetch(n, now)
            except Exception as exc:  # noqa: BLE001 - one publisher down must not stop the others
                msg = f"fetch failed: {type(exc).__name__}: {exc}"
                with self._lock:
                    self._errors[ccy] = msg
                errors.append(f"{ccy}: {msg}")
                continue
            if self._read(ccy):
                with self._lock:
                    self._errors.pop(ccy, None)
            else:
                errors.append(f"{ccy}: {self._errors.get(ccy, 'unreadable')}")
        self.last_errors = errors + self.cross_check(now)
        return self.last_errors

    def cross_check(self, now: pd.Timestamp) -> list[str]:
        """Compare each pair's official differential now with Oanda's financing differential."""
        if self.financing is None or not self.pairs:
            return []
        try:
            fin = self.financing(self.pairs)
        except Exception as exc:  # noqa: BLE001 - the check is optional; the official rates stand alone
            return [f"Oanda financing cross-check not run: {type(exc).__name__}: {exc}"]
        gaps, idx = {}, pd.DatetimeIndex([now])
        for pair in self.pairs:
            if pair not in fin:
                continue
            base, quote = pair.split("_")
            mine = float(self.rates(base, idx).iloc[0] - self.rates(quote, idx).iloc[0])
            theirs = (fin[pair][0] - fin[pair][1]) / 2
            if pd.notna(mine) and abs(mine - theirs) > self.tolerance:
                gaps[pair] = (f"{pair} official rate differential {mine:+.2%} disagrees with Oanda financing "
                              f"{theirs:+.2%} by more than {self.tolerance:.2%}")
        with self._lock:
            if gaps != self._gaps:
                self._gaps = gaps
                self.version += 1
        return list(gaps.values())

    def run(self, stop: threading.Event, poll_s: float = 600.0) -> None:
        """Background refresher (``Runtime.start``): refresh when due, look again every ``poll_s``. Faults are
        raised on the core's thread by the column builder, from ``problem`` (a failed fetch only matters once
        the last good one is stale), so this thread never writes the ledger."""
        while not stop.is_set():
            now = self.clock()
            if self.due(now):
                self.last_errors = self.refresh(now)
            stop.wait(poll_s)
