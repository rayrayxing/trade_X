"""Where ideas and spec drafts come from.

Agents (the researcher and strategy designer) propose through files only: an idea is a small YAML
or JSON file in ``research/ideas/``, and a draft strategy for it is a spec file in
``research/ideas/specs/<idea id>.yaml``. Neither can set a status or touch the gates; the loop
re-validates every draft, forces ``status: proposed`` and records where it came from.

``ArxivAtomSource`` reads arXiv search results through an injected ``http_get`` callable, so the
loop itself never opens a connection and tests feed it a saved response.

Everything read here is untrusted text. It is length-capped, stripped of control characters and only
ever stored or printed as data.
"""
from __future__ import annotations

import json
import re
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Callable, Iterable

import yaml

from tradex.research import catalog as catalog_mod
from tradex.research.loop.ports import Idea

ARXIV_ID = re.compile(r"^\d{4}\.\d{4,5}$")
ASSETS = {"forex", "stocks", "etfs", "options"}
FAMILY_FALLBACK = "other"
MAX_TITLE, MAX_SUMMARY = 200, 4000
_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
ATOM = "{http://www.w3.org/2005/Atom}"


class DraftError(ValueError):
    """A draft file exists but cannot be read as a spec."""


def clean(text: Any, limit: int) -> str:
    s = _CTRL.sub("", str(text or ""))
    s = re.sub(r"\s+", " ", s).strip()
    return s[:limit]


def safe_name(idea_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", idea_id).strip("._") or "idea"


# --- catalog --------------------------------------------------------------------------------------

class FileCatalog:
    """The research catalog (research/strategy_catalog.yaml) plus what the phase-1 gate plans already cover."""

    def __init__(self, path: str | Path | None = None, plans: Iterable[Any] | None = None):
        self.path = path
        self._plans = plans

    def entries(self) -> list[catalog_mod.CatalogEntry]:
        return catalog_mod.load(self.path)

    def handled(self) -> dict[str, str]:
        plans = self._plans
        if plans is None:
            from tradex.research.gate import PLANS
            plans = PLANS
        out: dict[str, str] = {}
        for p in plans:
            if getattr(p, "spec", None) is not None:
                out.setdefault(p.catalog_id, f"has a spec ({Path(p.spec).name})")
            elif getattr(p, "verdict", None):
                out.setdefault(p.catalog_id, f"{p.verdict}: {p.why}")
        return out


# --- ideas from agent files -----------------------------------------------------------------------

class FileIdeaSource:
    """Idea files dropped in a folder by agents or by hand."""
    name = "agent-files"
    local = True                  # reads files only; `--dry-plan` may call it

    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        self.errors: list[str] = []

    def fetch(self) -> list[Idea]:
        self.errors = []
        if not self.dir.exists():
            return []
        out = []
        for p in sorted(self.dir.iterdir()):
            if p.is_dir() or p.suffix not in (".yaml", ".yml", ".json"):
                continue
            try:
                doc = json.loads(p.read_text()) if p.suffix == ".json" else yaml.safe_load(p.read_text())
                out.append(self._idea(p, doc))
            except (ValueError, yaml.YAMLError) as exc:
                self.errors.append(f"{p.name}: {exc}")
        return out

    @staticmethod
    def _idea(p: Path, doc: Any) -> Idea:
        if not isinstance(doc, dict):
            raise ValueError("not a mapping")
        title = clean(doc.get("title"), MAX_TITLE)
        if not title:
            raise ValueError("title is required")
        arxiv = tuple(str(a) for a in (doc.get("arxiv") or []))
        if not all(ARXIV_ID.match(a) for a in arxiv):
            raise ValueError(f"arxiv must be a list of ids like 2501.01234, got {list(arxiv)}")
        assets = tuple(str(a).lower() for a in (doc.get("assets") or []))
        if not set(assets) <= ASSETS:
            raise ValueError(f"assets must be among {sorted(ASSETS)}, got {list(assets)}")
        return Idea(id=f"file:{safe_name(str(doc.get('id') or p.stem))}", source="agent-file", title=title,
                    summary=clean(doc.get("summary"), MAX_SUMMARY), url=clean(doc.get("url"), 300), arxiv=arxiv,
                    assets=assets, family=clean(doc.get("family") or FAMILY_FALLBACK, 40),
                    horizon=clean(doc.get("horizon"), 60), published=clean(doc.get("published"), 40))


# --- ideas from arXiv -----------------------------------------------------------------------------

class ArxivAtomSource:
    """arXiv API search results (Atom). ``http_get(url) -> str`` is injected."""

    def __init__(self, query: str, http_get: Callable[[str], str], max_results: int = 25, name: str = "arxiv"):
        self.query, self.http_get, self.max_results, self.name = query, http_get, max_results, name

    def url(self) -> str:
        q = urllib.parse.urlencode({"search_query": self.query, "sortBy": "submittedDate", "sortOrder": "descending",
                                    "start": 0, "max_results": self.max_results})
        return f"https://export.arxiv.org/api/query?{q}"

    def fetch(self) -> list[Idea]:
        return parse_atom(self.http_get(self.url()))


def parse_atom(xml_text: str) -> list[Idea]:
    if re.search(r"<!(DOCTYPE|ENTITY)", xml_text, re.IGNORECASE):
        raise ValueError("arXiv response carries a DTD/entity declaration; refusing to parse it")
    root = ET.fromstring(xml_text)
    out = []
    for e in root.findall(f"{ATOM}entry"):
        link = (e.findtext(f"{ATOM}id") or "").strip()
        m = re.search(r"arxiv\.org/abs/(?:[a-z\-.]+/)?(\d{4}\.\d{4,5})(?:v\d+)?$", link)
        if not m:
            continue
        aid = m.group(1)
        cats = [c.get("term", "") for c in e.findall(f"{ATOM}category")]
        out.append(Idea(id=f"arxiv:{aid}", source="arxiv", title=clean(e.findtext(f"{ATOM}title"), MAX_TITLE),
                        summary=clean(e.findtext(f"{ATOM}summary"), MAX_SUMMARY), url=f"https://arxiv.org/abs/{aid}",
                        arxiv=(aid,), published=clean(e.findtext(f"{ATOM}published"), 40),
                        extra={"categories": [clean(c, 30) for c in cats]}))
    return out


# --- drafts from agent files ----------------------------------------------------------------------

class FileSpecDrafter:
    """A draft spec for an idea, if an agent (or Ray) left one in the drafts folder."""
    local = True

    def __init__(self, directory: str | Path):
        self.dir = Path(directory)

    def draft(self, idea: Idea) -> dict[str, Any] | None:
        names = [safe_name(idea.id)] + ([safe_name(idea.catalog_id)] if idea.catalog_id else [])
        for n in names:
            for ext in (".yaml", ".yml"):
                p = self.dir / f"{n}{ext}"
                if p.exists():
                    try:
                        doc = yaml.safe_load(p.read_text())
                    except yaml.YAMLError as exc:
                        raise DraftError(f"{p.name}: {exc}") from exc
                    if not isinstance(doc, dict):
                        raise DraftError(f"{p.name}: not a mapping")
                    return doc
        return None
