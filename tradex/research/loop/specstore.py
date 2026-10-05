"""Strategy spec files on disk: list, create, change the status line, hash the content.

New specs are written to the first directory (``strategies/proposed``). The loop edits only the
``status:`` line of a file it did not create, so comments and layout in hand-written specs survive.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import yaml

from tradex.research.loop.ports import SpecRecord

SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9-]{2,60}$")
NOT_PART_OF_THE_STRATEGY = ("status", "stats", "provenance")     # change without making it a different strategy
_STATUS_LINE = re.compile(r"^status:[^\n]*$", re.MULTILINE)


class SpecExists(FileExistsError):
    pass


def content_hash(raw: dict[str, Any]) -> str:
    body = {k: v for k, v in raw.items() if k not in NOT_PART_OF_THE_STRATEGY}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()[:16]


class YamlSpecStore:
    def __init__(self, new_dir: str | Path, *other_dirs: str | Path):
        self.new_dir = Path(new_dir)
        self.dirs = [self.new_dir] + [Path(d) for d in other_dirs]

    def _files(self) -> list[Path]:
        out = []
        for d in self.dirs:
            if d.exists():
                out += sorted(d.glob("**/*.yaml"))
        return out

    def list_specs(self) -> list[SpecRecord]:
        out = []
        for p in self._files():
            raw = yaml.safe_load(p.read_text())
            if not isinstance(raw, dict) or "id" not in raw:
                continue
            out.append(SpecRecord(spec_id=str(raw["id"]), version=int(raw.get("version", 1)),
                                  status=str(raw.get("status", "proposed")), path=str(p), raw=raw, writable=True))
        return out

    def get(self, spec_id: str) -> SpecRecord | None:
        return next((r for r in self.list_specs() if r.spec_id == spec_id), None)

    def write_new(self, raw: dict[str, Any]) -> str:
        sid = str(raw.get("id", ""))
        if not SAFE_ID.match(sid):
            raise ValueError(f"spec id {sid!r} is not a safe file name (lowercase letters, digits, hyphens)")
        if self.get(sid) is not None:
            raise SpecExists(f"a spec with id {sid} already exists")
        self.new_dir.mkdir(parents=True, exist_ok=True)
        path = self.new_dir / f"{sid}.yaml"
        if path.exists():
            raise SpecExists(str(path))
        tmp = path.with_suffix(".yaml.tmp")
        tmp.write_text(yaml.safe_dump(raw, sort_keys=False, default_flow_style=None, width=120))
        os.replace(tmp, path)
        return str(path)

    def set_status(self, spec_id: str, status: str) -> None:
        rec = self.get(spec_id)
        if rec is None:
            raise KeyError(f"no spec file for {spec_id}")
        p = Path(rec.path)
        text = p.read_text()
        line = f"status: {status}"
        new = _STATUS_LINE.sub(line, text, count=1) if _STATUS_LINE.search(text) else text.rstrip("\n") + f"\n{line}\n"
        after = yaml.safe_load(new)
        if not isinstance(after, dict) or after.get("status") != status or content_hash(after) != content_hash(rec.raw):
            raise ValueError(f"{p}: changing the status line would alter something else; edit refused")
        tmp = p.with_suffix(".yaml.tmp")
        tmp.write_text(new)
        os.replace(tmp, p)

    def content_hash(self, spec_id: str) -> str:
        rec = self.get(spec_id)
        if rec is None:
            raise KeyError(f"no spec file for {spec_id}")
        return content_hash(rec.raw)
