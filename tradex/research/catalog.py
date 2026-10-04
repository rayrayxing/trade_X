"""The strategy catalog (research/strategy_catalog.yaml): load, schema-check, query.

The catalog is the research backlog: every idea with its source, data needs and status.
It is not a strategy spec; an entry becomes tradable only once someone writes a spec
for it (strategies/) and that spec passes the gates. ``status`` is free text whose first
word is the category (``Build: filter for all trend strategies`` is a Build entry).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

CATALOG_PATH = Path(__file__).resolve().parents[2] / "research" / "strategy_catalog.yaml"
STATUS_KINDS = ("Build", "Seed", "Watch", "Needs data", "Screened")
ASSETS = {"fx": "forex", "stocks": "stocks", "etfs": "etfs", "options": "options"}
REQUIRED = ("family", "name", "asset", "horizon", "data", "source", "data_today", "status")


class CatalogError(ValueError):
    pass


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


@dataclass(frozen=True)
class CatalogEntry:
    id: str
    family: str
    name: str
    assets: tuple[str, ...]        # normalised: forex, stocks, etfs, options
    horizon: str
    data: str
    source: str
    data_today: str
    status: str                    # full text as written
    status_kind: str               # one of STATUS_KINDS
    arxiv: tuple[str, ...] = field(default_factory=tuple)

    @property
    def status_note(self) -> str:
        return self.status[len(self.status_kind):].lstrip(" :;").strip()

    @property
    def is_build(self) -> bool:
        return self.status_kind == "Build"

    @property
    def is_seed(self) -> bool:
        """Seed entries map to a spec in strategies/seeds/ (some are also screened)."""
        return self.status_kind == "Seed"


def _status_kind(status: str) -> str:
    for k in STATUS_KINDS:
        if status == k or status.startswith(k + ":") or status.startswith(k + ";") or status.startswith(k + " "):
            return k
    raise CatalogError(f"status {status!r} does not start with one of {STATUS_KINDS}")


def _assets(text: str) -> tuple[str, ...]:
    out = []
    for part in re.split(r"[,/]", text):
        key = part.strip().lower()
        if key not in ASSETS:
            raise CatalogError(f"asset {part.strip()!r} unknown; known: {sorted(ASSETS)}")
        out.append(ASSETS[key])
    return tuple(out)


def parse(rows: list[dict]) -> list[CatalogEntry]:
    if not isinstance(rows, list):
        raise CatalogError("catalog must be a YAML list of entries")
    out, seen, errs = [], set(), []
    for i, r in enumerate(rows):
        try:
            if not isinstance(r, dict):
                raise CatalogError("entry is not a mapping")
            missing = [k for k in REQUIRED if not str(r.get(k, "")).strip()]
            if missing:
                raise CatalogError(f"missing {missing}")
            arxiv = r.get("arxiv") or []
            if not isinstance(arxiv, list) or not all(re.fullmatch(r"\d{4}\.\d{4,5}", str(a)) for a in arxiv):
                raise CatalogError(f"arxiv must be a list of arXiv ids, got {arxiv!r}")
            e = CatalogEntry(
                id=slug(r["name"]), family=str(r["family"]), name=str(r["name"]), assets=_assets(str(r["asset"])),
                horizon=str(r["horizon"]), data=str(r["data"]), source=str(r["source"]),
                data_today=str(r["data_today"]), status=str(r["status"]), status_kind=_status_kind(str(r["status"])),
                arxiv=tuple(str(a) for a in arxiv),
            )
            if e.id in seen:
                raise CatalogError(f"duplicate name {e.name!r}")
            seen.add(e.id)
            out.append(e)
        except CatalogError as exc:
            errs.append(f"entry {i} ({r.get('name') if isinstance(r, dict) else r!r}): {exc}")
    if errs:
        raise CatalogError("; ".join(errs))
    return out


def load(path: str | Path | None = None) -> list[CatalogEntry]:
    return parse(yaml.safe_load(Path(path or CATALOG_PATH).read_text()))


def by_status(entries: list[CatalogEntry], kind: str) -> list[CatalogEntry]:
    return [e for e in entries if e.status_kind == kind]
