"""`tradex research loop --dry-plan`: what the next run would do, worked out without market data.

It reads the catalog, the spec files, the idea and draft folders (local files) and the run state if one exists
(opened read-only; a missing state file is treated as empty and is not created). It does not load bars, open
the holdout, call a network source, write a file or change a status. Sources that would go to the network are
listed by name only.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from tradex.research import catalog as catalog_mod
from tradex.research.loop.config import LoopConfig
from tradex.research.loop.state import LoopState
from tradex.research.loop.stages import CLAIMS_THAT_MATTER, STAGES, _covered_by_specs, idea_from_row, key


def iso_week_id(now: pd.Timestamp) -> str:
    iso = now.isocalendar()
    return f"{int(iso.year)}-W{int(iso.week):02d}"


def _needs(raw: dict) -> dict[str, Any]:
    uni = [u for u in raw.get("universe", []) if not str(u).startswith("$")]
    cols = sorted({f.get("name") for f in (raw.get("features") or {}).values()
                   if isinstance(f, dict) and f.get("fn") == "data.column" and f.get("name")})
    return {"asset_class": raw.get("asset_class"), "timeframe": (raw.get("timeframes") or {}).get("signal"),
            "symbols": len(uni), "research_columns": cols}


def build_plan(cfg: LoopConfig, *, catalog, specs, idea_sources, drafter, state: LoopState | None, run_id: str,
               report_dir: str) -> dict[str, Any]:
    recs = specs.list_specs()
    rec_by = {(r.spec_id, r.version): r for r in recs}
    db_strats = {(r["strategy_id"], r["version"]): r for r in state.strategies()} if state else {}
    db_ideas = {i["idea_id"]: i for i in state.ideas()} if state else {}
    notes: list[str] = []

    # propose
    adopt = [r for k, r in rec_by.items() if k not in db_strats]
    claims = [r for r in adopt if r.status in CLAIMS_THAT_MATTER]
    new_ideas: list[str] = []
    network = []
    for src in idea_sources:
        if getattr(src, "local", False):
            try:
                found = [i.id for i in src.fetch() if i.id not in db_ideas]
            except Exception as exc:   # noqa: BLE001 - a plan reports problems, it does not stop on them
                notes.append(f"idea source {src.name} unreadable: {exc}")
                found = []
            new_ideas += found
            notes += [f"idea file rejected: {m}" for m in getattr(src, "errors", [])]
        else:
            network.append(src.name)
    try:
        entries = catalog.entries()
        handled = catalog.handled()
    except (catalog_mod.CatalogError, OSError) as exc:
        entries, handled = [], {}
        notes.append(f"catalog unreadable: {exc}")
    covered, src_slugs = _covered_by_specs(recs)
    cat = [e for e in sorted(entries, key=lambda e: e.id)
           if e.status_kind in cfg.catalog_kinds and e.id not in handled and e.id not in covered
           and not any(e.id in s for s in src_slugs) and f"catalog:{e.id}" not in db_ideas]
    budget = max(0, cfg.max_new_ideas - len(new_ideas))
    cat_new = [f"catalog:{e.id}" for e in cat[:budget]]
    propose = {"adopt_spec_files": [key(r.spec_id, r.version) for r in adopt], "status_claims_without_evidence": [
        key(r.spec_id, r.version) for r in claims], "new_idea_files": new_ideas, "new_catalog_ideas": cat_new,
        "network_sources_to_query": network, "catalog_ideas_over_budget": max(0, len(cat) - budget)}

    # implement
    pending = [i for i in db_ideas.values() if i["status"] in ("new", "awaiting_spec", "spec_invalid")]
    cand = [i["idea_id"] for i in pending] + new_ideas + cat_new
    drafts, waiting = [], []
    for iid in cand:
        row = db_ideas.get(iid)
        idea = idea_from_row(row) if row else None
        if idea is None:
            from tradex.research.loop.ports import Idea
            idea = Idea(id=iid, source=iid.split(":")[0], title=iid, catalog_id=iid.split(":", 1)[1] if iid.startswith("catalog:") else "")
        try:
            have = bool(getattr(drafter, "local", False)) and drafter.draft(idea) is not None
        except Exception:    # noqa: BLE001
            have = False
        (drafts if have else waiting).append(iid)
    implement = {"draft_available": drafts, "waiting_for_a_draft": waiting}

    # screen / walk-forward / holdout / promote / health
    def proposed_rows():
        return [r for r in db_strats.values() if r["status"] == "proposed" and not r["unverified"]]

    screen_keys = [key(r["strategy_id"], r["version"]) for r in proposed_rows()]
    screen_keys += [key(r.spec_id, r.version) for r in adopt if r.status not in CLAIMS_THAT_MATTER | {"rejected", "retired"}]
    screen_keys = screen_keys[:cfg.max_screens_per_run]
    wf_keys = [key(r["strategy_id"], r["version"]) for r in db_strats.values() if r["status"] == "screened" and not r["unverified"]]
    ho_keys = [key(r["strategy_id"], r["version"]) for r in db_strats.values() if r["status"] == "validated"
               and not r["unverified"] and not (state and state.latest_evidence(r["strategy_id"], r["version"], "holdout"))]
    ready = [r for r in db_strats.values() if r["status"] == "validated" and state
             and state.latest_evidence(r["strategy_id"], r["version"], "holdout")
             and state.latest_evidence(r["strategy_id"], r["version"], "holdout")["passed"]]
    n_paper = sum(1 for r in db_strats.values() if r["status"] == "paper")
    paper_keys = [key(r["strategy_id"], r["version"]) for r in db_strats.values() if r["status"] == "paper"]

    def needs(keys):
        out = {}
        for k in keys:
            sid, _, ver = k.rpartition(":v")
            rec = rec_by.get((sid, int(ver)))
            out[k] = _needs(rec.raw) if rec else {"asset_class": "?"}
        return out

    stages = {
        "propose": propose, "implement": implement,
        "screen": {"strategies": needs(screen_keys), "more_after_implement": len(drafts)},
        "walk_forward": {"strategies": needs(wf_keys[:cfg.max_walk_forwards_per_run]),
                         "also_if_screens_pass": len(screen_keys)},
        "holdout": {"strategies": needs(ho_keys), "also_if_walk_forward_passes": len(wf_keys)},
        "promote": {"holdout_passed": [key(r["strategy_id"], r["version"]) for r in ready],
                    "paper_slots_free": max(0, cfg.max_paper - n_paper),
                    "effect": "status changes applied" if cfg.apply else "recommendations only (apply is off)"},
        "health": {"paper_strategies": paper_keys},
        "report": {"directory": report_dir},
    }
    return {"run_id": run_id, "apply": cfg.apply, "state_file": "present" if state else "absent (first run)",
            "stages": [{"stage": s, **stages[s]} for s in STAGES], "notes": notes,
            "needs_at_run_time": ["real bars for every spec above (OpenD / Oanda caches; missing data blocks that item and alerts)",
                                  "the locked holdout (tradex.backtest.holdout and config/gates/holdout.yaml) for screen, "
                                  "walk-forward and holdout", "paper ledger snapshots for the health stage"]}


def render_plan(plan: dict[str, Any]) -> str:
    L = [f"research loop plan for run {plan['run_id']} (dry: no data loaded, nothing written)",
         f"  state file: {plan['state_file']}; status changes into/out of paper: {'applied' if plan['apply'] else 'not applied'}"]
    for st in plan["stages"]:
        L.append("")
        L.append(f"[{st['stage']}]")
        for k, v in st.items():
            if k == "stage":
                continue
            if isinstance(v, dict):
                L.append(f"  {k}: {len(v)}" + ("" if not v else ""))
                for kk, vv in v.items():
                    L.append(f"    - {kk}: {vv}")
            elif isinstance(v, list):
                L.append(f"  {k}: {len(v)}" + ("" if not v else " -> " + ", ".join(str(x) for x in v[:12]) + (" ..." if len(v) > 12 else "")))
            else:
                L.append(f"  {k}: {v}")
    if plan["notes"]:
        L += ["", "notes:"] + [f"  - {n}" for n in plan["notes"]]
    L += ["", "needs at run time:"] + [f"  - {n}" for n in plan["needs_at_run_time"]]
    return "\n".join(L) + "\n"
