"""`tradex research loop`: the weekly research engine.

    tradex research loop --dry-plan            print what the next run would do; loads no data, writes nothing
    tradex research loop                       run all eight stages (resumes this week's run if one was cut short)
    tradex research loop --stages screen walk_forward      only some stages
    tradex research loop --apply               also let the loop change status into and out of paper

Without ``--apply`` the loop still screens, validates and records evidence, and labels spec files proposed /
screened / validated / rejected, but promotion to paper and demotion out of paper are only recommended.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

from tradex.research.loop.config import (DEFAULT_IDEAS_DIR, DEFAULT_RESULTS, ROOT, LoopConfig, db_path)
from tradex.research.loop.stages import STAGES


def add_parser(sub) -> None:
    rs = sub.add_parser("research", help="research engine commands")
    rsub = rs.add_subparsers(dest="research_cmd", required=True)
    lp = rsub.add_parser("loop", help="the weekly 8-stage research loop (propose, implement, screen, walk-forward, "
                                      "holdout, paper queue, health, report)")
    lp.add_argument("--dry-plan", action="store_true", help="print what would run; loads no data and writes nothing")
    lp.add_argument("--json", action="store_true", help="with --dry-plan: print the plan as JSON")
    lp.add_argument("--apply", action="store_true", help="let the loop change status into and out of paper "
                                                         "(default: recommend only)")
    lp.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    lp.add_argument("--run-id", help="default: the ISO week, e.g. 2026-W41; the same id resumes")
    lp.add_argument("--db", help="run state file (default data/research/loop.sqlite or $TRADEX_LOOP_DB)")
    lp.add_argument("--config", help="default config/research_loop.yaml")
    lp.add_argument("--strategies", default=str(ROOT / "strategies"), help="folder with proposed/ and seeds/")
    lp.add_argument("--extra-specs", nargs="*", default=[str(ROOT / "research" / "specs")],
                    help="other folders of spec files the loop tracks")
    lp.add_argument("--ideas-dir", default=str(DEFAULT_IDEAS_DIR), help="idea files dropped by agents (research/ideas)")
    lp.add_argument("--drafts-dir", help="draft specs for ideas (default <ideas-dir>/specs)")
    lp.add_argument("--arxiv-query", action="append", default=[], help="arXiv search_query to pull ideas from (repeatable)")
    lp.add_argument("--arxiv-max", type=int, default=25)
    lp.add_argument("--ledger", default=str(ROOT / "data" / "ledger" / "live.sqlite"),
                    help="trading ledger the health stage reads paper results from")
    lp.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    ms = rsub.add_parser("measure", help="write each strategy's measured hit rate (from the loop's real-data "
                                         "walk-forward evidence) into its spec file; paper/live vote on nothing else")
    ms.add_argument("--db", help="run state file (default data/research/loop.sqlite or $TRADEX_LOOP_DB)")
    ms.add_argument("--strategies", default=str(ROOT / "strategies"), help="folder with proposed/ and seeds/")
    ms.add_argument("--extra-specs", nargs="*", default=[str(ROOT / "research" / "specs")])


def _sources(a):
    from tradex.research.loop.sources import ArxivAtomSource, FileCatalog, FileIdeaSource, FileSpecDrafter
    from tradex.research.loop.specstore import YamlSpecStore
    root = Path(a.strategies)
    specs = YamlSpecStore(root / "proposed", root / "seeds", *a.extra_specs)
    ideas = [FileIdeaSource(a.ideas_dir)]
    drafter = FileSpecDrafter(a.drafts_dir or Path(a.ideas_dir) / "specs")
    return FileCatalog(), specs, ideas, drafter, ArxivAtomSource


def run(a) -> int:
    if a.research_cmd == "measure":
        from tradex.research.loop.specstore import YamlSpecStore
        from tradex.research.loop.state import LoopState
        from tradex.research.measure import measure
        path = db_path(a.db)
        if not path.exists():
            print(f"no research state at {path}: run `tradex research loop` first", file=sys.stderr)
            return 1
        root = Path(a.strategies)
        res = measure(LoopState(path, read_only=True), YamlSpecStore(root / "proposed", root / "seeds", *a.extra_specs))
        for sid, what in sorted(res.items()):
            print(f"{sid}: {what}")
        return 0
    if a.research_cmd != "loop":
        return 2
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", stream=sys.stderr)
    cfg = LoopConfig.load(a.config, apply=True if a.apply else None)
    catalog, specs, ideas, drafter, arxiv_cls = _sources(a)
    if a.arxiv_query:
        from tradex.research.loop.adapters import urllib_get
        ideas += [arxiv_cls(q, urllib_get(), a.arxiv_max, name=f"arxiv:{q[:40]}") for q in a.arxiv_query]

    if a.dry_plan:
        from tradex.research.loop.plan import build_plan, iso_week_id, render_plan
        from tradex.research.loop.state import LoopState
        path = db_path(a.db)
        state = LoopState(path, read_only=True) if path.exists() else None
        plan = build_plan(cfg, catalog=catalog, specs=specs, idea_sources=ideas, drafter=drafter, state=state,
                          run_id=a.run_id or iso_week_id(pd.Timestamp.now(tz="UTC")), report_dir=str(a.results_dir))
        print(json.dumps(plan, indent=2, default=str) if a.json else render_plan(plan), end="")
        return 0

    from tradex.core.ledger import Ledger
    from tradex.research.loop.adapters import (CollectingAlerter, GateResearchData, LedgerLiveReturns, LockedHoldout,
                                               LogAlerter, UnavailableHoldout, UnavailableLive)
    from tradex.research.loop.engine import run_loop, summarise
    from tradex.research.loop.evaluators import EngineEvaluator
    from tradex.research.loop.ports import HoldoutUnavailable, Ports
    from tradex.research.loop.state import LoopState
    from tradex.research.trials import TrialLedger

    try:
        holdout = LockedHoldout()
    except HoldoutUnavailable as exc:
        print(f"holdout lock unavailable: {exc}\nscreen, walk-forward and holdout will block; the other stages run.",
              file=sys.stderr)
        holdout = UnavailableHoldout(str(exc))
    led_path = Path(a.ledger)
    live = LedgerLiveReturns(Ledger(led_path, read_only=True)) if led_path.exists() else UnavailableLive(
        f"no trading ledger at {led_path}")
    alerts = CollectingAlerter(LogAlerter())
    ports = Ports(catalog=catalog, specs=specs, idea_sources=ideas, drafter=drafter, data=GateResearchData(),
                  holdout=holdout, live=live, alerts=alerts, evaluator=EngineEvaluator(), trials=TrialLedger())
    state = LoopState(db_path(a.db))
    results = run_loop(ports, cfg, state, a.run_id, tuple(a.stages), a.results_dir)
    print(json.dumps(summarise(results), indent=2))
    rep = results.get("report")
    if rep and rep.items.get("report"):
        print(f"report: {rep.items['report']['path']}")
    return 1 if any(r.failed for r in results.values()) else 0
