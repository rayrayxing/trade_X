"""Real-data guard: paper and live never run on synthetic, CSV or cached stand-in data.

``require_real_data(mode, provider)`` is called when a data provider is wired into a run.
``require_present`` is called per decision: a missing price/spread raises RealDataMissing,
which callers must treat as "block this trade and alert" (never estimate).
"""
from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger("tradex.data.guard")

MODES = ("backtest", "replay", "paper", "live")
STRICT = ("paper", "live")
_STAND_INS = ("synthetic", "csv", "cached")     # matched against the provider's class and module names


class SyntheticDataRefused(RuntimeError):
    """Paper/live was wired to a synthetic, CSV or cached provider."""


class RealDataMissing(RuntimeError):
    """Real data needed for a decision is absent. Block the trade and alert; do not fall back."""


def _label(provider) -> str:
    if isinstance(provider, str):
        return provider.lower()
    t = type(provider)
    return f"{t.__module__}.{t.__qualname__}".lower()


def require_real_data(mode: str, provider) -> None:
    if mode not in MODES:
        raise ValueError(f"unknown run mode {mode!r}; known: {MODES}")
    if mode not in STRICT:
        return
    if provider is None:
        _alert("no data provider configured", mode)
        raise RealDataMissing(f"{mode}: no data provider configured")
    label = _label(provider)
    if any(tag in label for tag in _STAND_INS):
        _alert(f"refused provider {label}", mode)
        raise SyntheticDataRefused(f"{mode} mode refuses synthetic/CSV/cached data ({label})")


def require_present(mode: str, value, what: str):
    """Return ``value`` if it is real data; in paper/live raise RealDataMissing when it is None/empty/NaN."""
    missing = value is None
    if not missing and isinstance(value, (pd.DataFrame, pd.Series)):
        missing = value.empty or bool(value.isna().all().all() if isinstance(value, pd.DataFrame) else value.isna().all())
    elif not missing and isinstance(value, float):
        missing = value != value
    if missing and mode in STRICT:
        _alert(f"missing {what}", mode)
        raise RealDataMissing(f"{mode}: no real data for {what}")
    return value


def _alert(msg: str, mode: str) -> None:
    log.error(msg, extra={"event": "real_data_missing", "mode": mode})
