"""Shared helpers for the adversarial gap tests (ids refer to the 4 Oct 2026 gap review).

``known_gap`` marks a test that reproduces a gap still present on the Phase 1 tip. It is a
STRICT xfail naming the gap id: the day a fix lands the test XPASSes, which fails the run
and tells whoever fixed it to delete the marker. ``raises=AssertionError`` keeps a broken
test (a typo, an import error) from hiding behind the marker: only a failed assertion about
the gap counts as "gap still open".
"""
from __future__ import annotations

import pandas as pd
import pytest

from tradex.core.records import TradePlan, Verdict

T0 = pd.Timestamp("2026-03-02 21:00", tz="UTC")


def known_gap(gap_id: str, why: str):
    return pytest.mark.xfail(strict=True, raises=AssertionError, reason=f"{gap_id}: {why}")


def raised(exc_types, fn, *a, **kw) -> bool:
    """True when ``fn`` raises one of ``exc_types``. Lets a gap test assert plainly instead of
    using ``pytest.raises`` (whose failure is not an AssertionError and would not xfail)."""
    try:
        fn(*a, **kw)
    except exc_types:
        return True
    return False


def plan(did="2026-03-02-0001", symbol="EUR_USD", direction=1, entry=1.10, stop=1.09, book="ensemble") -> TradePlan:
    return TradePlan(did, T0.isoformat(), symbol, "forex", direction, "market", entry, stop,
                     [entry + direction * 2 * abs(entry - stop)], 20, "stop", ["trend"], ["s1"], 0.6, 0.45, "base_rate",
                     2.0, 0.2, 0.05, book, "H1")


def verdict(did="2026-03-02-0001", qty=1000.0, outcome="accepted") -> Verdict:
    return Verdict(did, T0.isoformat(), outcome, qty, 10.0, 0.1, [], {}, f"{did}-v")


def recorded(df: pd.DataFrame) -> pd.DataFrame:
    """Test bars standing in for what a real feed returned: the synthetic provenance tag is dropped, the way a
    recorded response would arrive. Only for tests of the live wiring; paper/live refuse tagged frames."""
    out = df.copy()
    out.attrs.pop("origin", None)
    return out
