"""Interfaces the research loop is written against.

Every stage in ``tradex.research.loop.stages`` is a function over these protocols, so the
loop never reaches for a data file, a network, a clock or a broker by itself. Production
wiring lives in ``adapters``/``sources``/``specstore``; tests wire fakes and fixtures.

Two rules shape the interfaces:

* Real data only. ``ResearchData`` either returns real bars or raises ``DataUnavailable``;
  there is no fallback anywhere in the loop. A stage that cannot get its data records the
  item as ``blocked``, raises an alert and tries again on the next run.
* The holdout is a port of its own. Earlier stages never hold holdout bars: the loop cuts
  every frame through ``HoldoutPort.research_view`` before screening or walk-forward, and
  only the holdout stage calls ``HoldoutPort.look``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from tradex.research.sources import DataUnavailable   # re-exported: the one "missing real data" exception

__all__ = ["DataUnavailable", "HoldoutUnavailable", "HoldoutAlreadyLooked", "LiveDataMissing", "Idea", "SpecRecord",
           "CatalogPort", "IdeaSource", "SpecStore", "SpecDrafter", "ResearchData", "HoldoutPort", "LiveReturns",
           "Alerter", "Evaluator", "Ports"]


class HoldoutUnavailable(RuntimeError):
    """The holdout lock (tradex.backtest.holdout and config/gates/holdout.yaml) is not installed or not readable."""


class HoldoutAlreadyLooked(PermissionError):
    """This strategy version has used its one look at the holdout."""


class LiveDataMissing(RuntimeError):
    """Paper results for a strategy cannot be read (no ledger rows, no equity snapshot)."""


@dataclass(frozen=True)
class Idea:
    """A candidate strategy idea. Text fields come from outside (papers, agent files) and are data only."""
    id: str                                  # stable and unique across sources, e.g. "arxiv:2501.01234", "file:my-idea"
    source: str                              # "catalog" | "arxiv" | "agent-file" | ...
    title: str
    summary: str = ""
    url: str = ""
    arxiv: tuple[str, ...] = ()
    assets: tuple[str, ...] = ()
    family: str = "other"
    horizon: str = ""
    catalog_id: str = ""                     # set when the idea is a catalog entry
    published: str = ""
    extra: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)


@dataclass(frozen=True)
class SpecRecord:
    """One strategy spec file as the store sees it."""
    spec_id: str
    version: int
    status: str
    path: str
    raw: dict[str, Any] = field(compare=False, hash=False, repr=False)
    writable: bool = False                   # whether the loop may change this file's status line


@runtime_checkable
class CatalogPort(Protocol):
    def entries(self) -> list[Any]:
        """``tradex.research.catalog.CatalogEntry`` rows."""

    def handled(self) -> dict[str, str]:
        """catalog id -> why the loop must not propose it again (already has a spec, not buildable, ...)."""


@runtime_checkable
class IdeaSource(Protocol):
    name: str

    def fetch(self) -> list[Idea]:
        """New or recent ideas. May raise; the loop records the failure and goes on with the other sources."""


@runtime_checkable
class SpecStore(Protocol):
    def list_specs(self) -> list[SpecRecord]: ...

    def get(self, spec_id: str) -> SpecRecord | None: ...

    def write_new(self, raw: dict[str, Any]) -> str:
        """Write a new proposed spec file and return its path. Refuses to overwrite an existing id."""

    def set_status(self, spec_id: str, status: str) -> None:
        """Change only the ``status:`` line of an existing, writable spec file."""

    def content_hash(self, spec_id: str) -> str:
        """Hash of everything that defines the strategy (not status, stats or provenance)."""


@runtime_checkable
class SpecDrafter(Protocol):
    def draft(self, idea: Idea) -> dict[str, Any] | None:
        """A strategy spec (a dict in the spec YAML format) for the idea, or None when none is available yet."""


@runtime_checkable
class ResearchData(Protocol):
    source: str                              # e.g. "opend", "oanda"; recorded in the audit trail
    real: bool                               # must be True; the loop refuses a provider that says otherwise

    def frames(self, spec: Any) -> dict[str, pd.DataFrame]:
        """Real bars (and research columns) for the spec's universe, whole history including any holdout period.

        Raises ``DataUnavailable`` when the bars or an input file are missing. Never fabricates or fills in."""


@runtime_checkable
class HoldoutPort(Protocol):
    start: pd.Timestamp

    def research_view(self, frames: dict[str, pd.DataFrame], tf: str) -> dict[str, pd.DataFrame]:
        """The frames cut to bars that closed at or before the holdout start."""

    def has_looked(self, strategy_id: str, version: int) -> bool: ...

    def look(self, frames: dict[str, pd.DataFrame], strategy_id: str, version: int, reason: str
             ) -> dict[str, pd.DataFrame]:
        """The holdout bars for one look, recorded in the lock's ledger. A second look raises ``HoldoutAlreadyLooked``."""


@runtime_checkable
class LiveReturns(Protocol):
    def daily_returns(self, strategy_id: str, version: int, since: pd.Timestamp | None) -> pd.Series | None:
        """Paper-account daily returns attributed to the strategy (fraction of equity). None or empty when there are none."""


@runtime_checkable
class Alerter(Protocol):
    def alert(self, level: str, title: str, detail: str = "") -> None:
        """level is "info", "warning" or "critical"."""


@runtime_checkable
class Evaluator(Protocol):
    """The three measurements the loop takes, plus the gate thresholds it judges them against."""

    thresholds: Any                          # tradex.backtest.validation.Thresholds (from the protected config)

    def screen(self, spec: Any, frames: dict[str, pd.DataFrame], trials: Any, run_id: str) -> dict[str, Any]: ...

    def walk_forward(self, spec: Any, frames: dict[str, pd.DataFrame], trials: Any, data_key: str) -> Any: ...

    def holdout(self, spec: Any, params: dict[str, Any], frames: dict[str, pd.DataFrame], start: pd.Timestamp
                ) -> dict[str, Any]: ...


@dataclass
class Ports:
    """Everything the loop is allowed to touch. Built once per run, by the CLI or by a test."""
    catalog: CatalogPort
    specs: SpecStore
    idea_sources: list[IdeaSource]
    drafter: SpecDrafter
    data: ResearchData
    holdout: HoldoutPort
    live: LiveReturns
    alerts: Alerter
    evaluator: Evaluator
    trials: Any                              # tradex.research.trials.TrialLedger
    now: Any = None                          # callable returning a UTC pd.Timestamp; None = wall clock
