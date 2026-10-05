"""Run the loop: pick the stages, give them a context, keep the run record, never let one stage stop the rest."""
from __future__ import annotations

import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from tradex.research.loop.adapters import require_real
from tradex.research.loop.config import DEFAULT_RESULTS, LoopConfig
from tradex.research.loop.plan import build_plan, iso_week_id
from tradex.research.loop.ports import DataUnavailable, Ports
from tradex.research.loop.stages import STAGE_FUNCS, STAGES, Ctx, StageResult
from tradex.research.loop.state import LoopState


def run_loop(ports: Ports, cfg: LoopConfig, state: LoopState, run_id: str | None = None,
             stages: tuple[str, ...] = STAGES, out_dir: str | Path | None = None) -> dict[str, StageResult]:
    """Run ``stages`` in order. The same ``run_id`` resumes: items done earlier are not repeated."""
    unknown = [s for s in stages if s not in STAGE_FUNCS]
    if unknown:
        raise ValueError(f"unknown stages {unknown}; known: {list(STAGES)}")
    now = ports.now() if ports.now else pd.Timestamp.now(tz="UTC")
    run_id = run_id or iso_week_id(now)
    ctx = Ctx(ports=ports, cfg=cfg, state=state, run_id=run_id, out_dir=Path(out_dir or DEFAULT_RESULTS))
    plan = build_plan(cfg, catalog=ports.catalog, specs=ports.specs, idea_sources=ports.idea_sources,
                      drafter=ports.drafter, state=state, run_id=run_id, report_dir=str(ctx.out_dir))
    fresh = state.start_run(run_id, plan, asdict(cfg))
    ctx.audit("run", "run_started" if fresh else "run_resumed", apply=cfg.apply, stages=list(stages))
    data_ok = True
    try:
        require_real(ports.data)
    except DataUnavailable as exc:
        data_ok = False
        ctx.alert("critical", "research data port refused", str(exc), "run")
    results: dict[str, StageResult] = {}
    for name in stages:
        if name in ("screen", "walk_forward", "holdout") and not data_ok:
            results[name] = StageResult(name, blocked=1, items={"stage": {"outcome": "blocked", "reason": "no real-data provider"}})
            continue
        try:
            results[name] = STAGE_FUNCS[name](ctx)
        except Exception as exc:    # noqa: BLE001 - the other stages still run
            results[name] = StageResult(name, failed=1, items={"stage": {"outcome": "failed", "reason": str(exc)}})
            ctx.alert("critical", f"stage {name} crashed", f"{type(exc).__name__}: {exc}", name)
            ctx.audit(name, "stage_crashed", error=traceback.format_exc()[-1500:])
    bad = sum(r.failed for r in results.values())
    state.finish_run(run_id, "complete" if not bad else "partial")
    ctx.audit("run", "run_finished", failed=bad, blocked=sum(r.blocked for r in results.values()))
    return results


def summarise(results: dict[str, StageResult]) -> dict[str, Any]:
    return {k: {"done": r.done, "skipped": r.skipped, "blocked": r.blocked, "failed": r.failed}
            for k, r in results.items()}
