"""Production wiring for the ports: real bar caches, the holdout lock, the paper ledger, alerts.

Nothing here fabricates or substitutes data. A missing cache, an uninstalled holdout lock or a book
without snapshots raises (``DataUnavailable``, ``HoldoutUnavailable``, ``LiveDataMissing``) and the stage
that asked records the item as blocked and alerts.
"""
from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from tradex.research.loop.ports import (DataUnavailable, HoldoutAlreadyLooked, HoldoutUnavailable, LiveDataMissing)

log = logging.getLogger("tradex.research.loop")


# --- data -----------------------------------------------------------------------------------------

class GateResearchData:
    """Real bars from the OpenD and Oanda caches, with the research columns the phase-1 gate builds.

    The builder for a spec is the one ``tradex.research.gate.PLANS`` names for it. A spec the gate does not know
    and that reads no research column gets plain bars; one that reads a research column nobody builds raises
    ``DataUnavailable`` ("needs a data builder") instead of running on bars without it.
    """
    real = True
    source = "opend+oanda caches"

    def __init__(self, cache=None):
        self.cache = cache

    @staticmethod
    def column_names(spec) -> set[str]:
        return {f["name"] for f in spec.features.values() if f.get("fn") == "data.column" and f.get("name")}

    def builder_for(self, spec) -> str:
        from tradex.research.gate import PLANS
        plan = next((p for p in PLANS if p.spec is not None and p.spec.stem == spec.id), None)
        if plan is not None and plan.builder:
            return plan.builder
        cols = self.column_names(spec)
        if cols:
            raise DataUnavailable(f"{spec.id} reads research columns {sorted(cols)} but no data builder is registered "
                                  f"for it (add a Plan in tradex/research/gate.py)")
        return "fx" if spec.asset_class == "forex" else "us_d1"

    def frames(self, spec) -> dict[str, pd.DataFrame]:
        from tradex.research import builders, gate
        builder = self.builder_for(spec)
        syms = [s for s in spec.universe if not s.startswith("$")]
        if spec.uses_watchlist:
            raise DataUnavailable(f"{spec.id} uses the daily watchlist; it cannot be researched on a fixed universe")
        if builder == "fx":
            data = builders.fx_bars(syms, spec.signal_tf)               # raises DataUnavailable when nothing is cached
            missing = [s for s in syms if s not in data]
        else:
            kw = {} if self.cache is None else {"cache": self.cache}
            data, missing = gate.build(builder, spec, **kw)
        if not data:
            raise DataUnavailable(f"no cached bars for {missing or syms}")
        return data


# --- holdout --------------------------------------------------------------------------------------

class LockedHoldout:
    """The holdout lock (``tradex.backtest.holdout`` + the protected ``config/gates/holdout.yaml``).

    If the lock is not installed in this checkout, construction raises ``HoldoutUnavailable`` and the holdout
    stage blocks: there is no substitute holdout, because a made-up split is not a locked one.
    """

    def __init__(self, module: Any = None, policy: Any = None):
        if module is None:
            try:
                from tradex.backtest import holdout as module
            except ImportError as exc:
                raise HoldoutUnavailable("tradex.backtest.holdout is not in this checkout; merge the PR that "
                                         "carries the locked holdout (spec C4) first") from exc
        self.mod = module
        try:
            self.policy = policy or module.HoldoutPolicy.from_config()
        except (OSError, KeyError, ValueError) as exc:
            raise HoldoutUnavailable(f"holdout policy unreadable (config/gates/holdout.yaml): {exc}") from exc
        self.start = self.policy.start

    def research_view(self, frames, tf):
        return self.mod.research_view(frames, tf, self.policy)

    def has_looked(self, strategy_id: str, version: int) -> bool:
        return any((lk["strategy_id"], str(lk["version"])) == (strategy_id, str(version))
                   for lk in self.mod.looks(self.policy))

    def look(self, frames, strategy_id, version, reason):
        try:
            return self.mod.holdout_look(frames, strategy_id, version, reason, self.policy)
        except self.mod.HoldoutLocked as exc:
            raise HoldoutAlreadyLooked(str(exc)) from exc


class UnavailableHoldout:
    """Stands in for the holdout port when the lock is not installed. Every use raises ``HoldoutUnavailable``, so
    the stages that need a research cut (screen, walk-forward) and the look (holdout) block instead of running
    on an uncut history."""

    def __init__(self, reason: str):
        self.reason = reason

    @property
    def start(self):
        raise HoldoutUnavailable(self.reason)

    def research_view(self, frames, tf):
        raise HoldoutUnavailable(self.reason)

    def has_looked(self, strategy_id, version):
        raise HoldoutUnavailable(self.reason)

    def look(self, frames, strategy_id, version, reason):
        raise HoldoutUnavailable(self.reason)


class UnavailableLive:
    """Stands in for the paper-results port when no ledger is configured."""

    def __init__(self, reason: str):
        self.reason = reason

    def daily_returns(self, strategy_id, version, since):
        raise LiveDataMissing(self.reason)


# --- paper results --------------------------------------------------------------------------------

class LedgerLiveReturns:
    """Daily returns of a strategy's virtual book, from the end-of-day equity snapshots in the trading ledger.

    Each strategy trades its own ``virtual:<id>`` book next to the ensemble, so its equity curve is its own.
    """

    def __init__(self, ledger):
        self.ledger = ledger

    def daily_returns(self, strategy_id: str, version: int, since: pd.Timestamp | None) -> pd.Series | None:
        rows = self.ledger.rows(kind="snapshot", book=f"virtual:{strategy_id}")
        if not rows:
            raise LiveDataMissing(f"no equity snapshots for book virtual:{strategy_id} in the ledger")
        eq = pd.Series({pd.Timestamp(r["time"]): float(r["equity_usd"]) for r in rows}).sort_index()
        eq.index = eq.index.tz_convert("UTC") if eq.index.tz is not None else eq.index.tz_localize("UTC")
        eq = eq.groupby(eq.index.normalize()).last()
        rets = eq.pct_change().dropna()
        if since is not None:
            rets = rets[rets.index >= pd.Timestamp(since).tz_convert("UTC").normalize()]
        return rets


# --- alerts ---------------------------------------------------------------------------------------

class LogAlerter:
    """Alerts to the structured log (the Telegram service and the dashboard read the log/ledger health rows)."""
    LEVELS = {"info": logging.INFO, "warning": logging.WARNING, "critical": logging.ERROR}

    def alert(self, level: str, title: str, detail: str = "") -> None:
        log.log(self.LEVELS.get(level, logging.WARNING), "research-loop alert: %s", title,
                extra={"event": "research_loop_alert", "level_name": level, "detail": detail})


class CollectingAlerter:
    """Keeps alerts in memory (tests, and the weekly report's alert list)."""

    def __init__(self, inner=None):
        self.items: list[dict[str, str]] = []
        self.inner = inner

    def alert(self, level: str, title: str, detail: str = "") -> None:
        self.items.append({"level": level, "title": title, "detail": detail})
        if self.inner is not None:
            self.inner.alert(level, title, detail)


def urllib_get(timeout: float = 30.0) -> Callable[[str], str]:
    """A plain HTTPS GET (honours HTTPS_PROXY). The only network call in the loop, and only for idea sources."""
    def get(url: str) -> str:
        if not url.startswith("https://"):
            raise ValueError("idea sources are fetched over https only")
        req = urllib.request.Request(url, headers={"User-Agent": "tradex-research-loop"})
        with urllib.request.urlopen(req, timeout=timeout) as r:   # noqa: S310 - https only, checked above
            return r.read().decode("utf-8", "replace")
    return get


def require_real(data) -> None:
    """Refuse a data port that does not say it serves real data, or whose name says synthetic."""
    label = f"{type(data).__module__}.{type(data).__qualname__} {getattr(data, 'source', '')}".lower()
    if getattr(data, "real", False) is not True or "synthetic" in label:
        raise DataUnavailable(f"research data port {label.strip()!r} is not a real-data provider")


def default_ideas_dir() -> Path:
    from tradex.research.loop.config import DEFAULT_IDEAS_DIR
    return DEFAULT_IDEAS_DIR
