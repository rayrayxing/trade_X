"""The eight stages of the weekly research loop.

    1 propose      ideas from the catalog, from injected idea sources (arXiv results, agent files) and spec files
    2 implement    a draft spec for each idea, validated like `tradex check`, written as status: proposed
    3 screen       a cheap backtest of the spec's own parameters on research-window bars (proposed -> screened)
    4 walk_forward the stage-3 gate: walk-forward, costs, global trial ledger, deflated Sharpe (screened -> validated)
    5 holdout      one recorded look at the locked holdout per strategy version
    6 promote      the paper-queue decision (validated -> paper), only with passing evidence on the current spec
    7 health       CUSUM and drawdown against the baseline; demotion of paper strategies, retirement of stale ones
    8 report       the weekly markdown report

Each stage is a function ``stage(ctx) -> StageResult`` over the injected ``ctx.ports`` and the SQLite
``ctx.state``. Work is split into items (one idea, one strategy); an item that is done in this run is not done
again when the run is resumed, and one that was blocked (missing data) or failed is retried. Nothing in a stage
reads the clock, a file, a network or a broker except through a port.

The holdout is never read before stage 5: every frame goes through ``research_frames``, which cuts it with the
holdout port, and refuses to continue if the port cannot (no lock installed means no screening either).
"""
from __future__ import annotations

import copy
import time as _time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from tradex.research import catalog as catalog_mod
from tradex.research.loop import health as health_mod
from tradex.research.loop.config import LoopConfig
from tradex.research.loop.evaluators import num
from tradex.research.loop.ports import (DataUnavailable, HoldoutAlreadyLooked, HoldoutUnavailable, Idea, LiveDataMissing,
                                        Ports)
from tradex.research.loop.sources import DraftError
from tradex.research.loop.specstore import content_hash
from tradex.research.loop.state import IllegalTransition, LoopState, NotEligible
from tradex.strategy.spec import StrategySpec
from tradex.timeframes import duration

STAGES = ("propose", "implement", "screen", "walk_forward", "holdout", "promote", "health", "report")
BLOCKING = (DataUnavailable, HoldoutUnavailable, LiveDataMissing)
CLAIMS_THAT_MATTER = {"validated", "paper", "live"}      # a spec file saying one of these has to be backed by evidence


@dataclass
class StageResult:
    stage: str
    done: int = 0
    skipped: int = 0
    blocked: int = 0
    failed: int = 0
    items: dict[str, dict] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.failed


@dataclass
class Ctx:
    ports: Ports
    cfg: LoopConfig
    state: LoopState
    run_id: str
    out_dir: Any = None                        # where the report goes (pathlib.Path)

    def now(self) -> pd.Timestamp:
        n = self.ports.now() if self.ports.now else pd.Timestamp.now(tz="UTC")
        return n.tz_localize("UTC") if n.tzinfo is None else n.tz_convert("UTC")

    def alert(self, level: str, title: str, detail: str = "", stage: str = "", strategy_id: str = "") -> None:
        self.state.audit(self.run_id, stage, "alert", strategy_id, {"level": level, "title": title, "detail": detail})
        try:
            self.ports.alerts.alert(level, title, detail)
        except Exception:      # noqa: BLE001 - a broken alert channel must not stop research
            pass

    def audit(self, stage: str, event: str, strategy_id: str = "", **detail) -> None:
        self.state.audit(self.run_id, stage, event, strategy_id, detail)


def run_item(ctx: Ctx, res: StageResult, item: str, fn: Callable[[], dict | None]) -> dict | None:
    """Run one unit of work with resume semantics. Returns its result, or None if blocked or failed."""
    prev = ctx.state.step(ctx.run_id, res.stage, item)
    if prev and prev["status"] == "done":
        res.skipped += 1
        res.items[item] = prev["result"]
        return prev["result"]
    ctx.state.begin_step(ctx.run_id, res.stage, item)
    try:
        out = fn() or {}
    except BLOCKING as exc:
        ctx.state.end_step(ctx.run_id, res.stage, item, "blocked", {"reason": str(exc)}, str(exc))
        ctx.alert("warning", f"{res.stage}: {item} blocked", str(exc), res.stage, item.split(":")[0])
        res.blocked += 1
        res.items[item] = {"outcome": "blocked", "reason": str(exc)}
        return None
    except Exception as exc:   # noqa: BLE001 - one bad item must not stop the week
        msg = f"{type(exc).__name__}: {exc}"
        ctx.state.end_step(ctx.run_id, res.stage, item, "failed", {"reason": msg}, traceback.format_exc()[-2000:])
        ctx.alert("critical", f"{res.stage}: {item} failed", msg, res.stage, item.split(":")[0])
        res.failed += 1
        res.items[item] = {"outcome": "failed", "reason": msg}
        return None
    ctx.state.end_step(ctx.run_id, res.stage, item, "done", out)
    res.done += 1
    res.items[item] = out
    return out


def key(sid: str, version: int) -> str:
    return f"{sid}:v{version}"


# --- shared helpers ---------------------------------------------------------------------------------

def research_frames(ctx: Ctx, spec: StrategySpec) -> dict[str, pd.DataFrame]:
    """Real bars for the spec, cut to the research window. The only door stages 3 and 4 get data through."""
    hp = ctx.ports.holdout
    frames = ctx.ports.data.frames(spec)
    start = hp.start                                   # raises HoldoutUnavailable when the lock is missing
    cut = {s: df for s, df in hp.research_view(frames, spec.signal_tf).items() if len(df)}
    if not cut:
        raise DataUnavailable(f"{spec.id}: no bars before the holdout start {start}")
    last = max(df.index[-1] + duration(spec.signal_tf) for df in cut.values())
    if last > start:     # defence in depth: a port that returns more than it should is a bug, not data
        raise RuntimeError(f"{spec.id}: research frames reach {last}, past the holdout start {start}")
    return cut


def data_key(ctx: Ctx, spec: StrategySpec, frames: dict[str, pd.DataFrame]) -> str:
    lo = min(df.index[0] for df in frames.values())
    hi = max(df.index[-1] for df in frames.values())
    return f"{ctx.ports.data.source}:{spec.signal_tf}:{','.join(sorted(frames))}:{lo.date()}..{hi.date()}"


def load_spec(ctx: Ctx, row: dict) -> tuple[StrategySpec, str]:
    """The spec file for a registered strategy, and its content hash. Raises if it is gone or is another version."""
    rec = ctx.ports.specs.get(row["strategy_id"])
    if rec is None:
        raise DataUnavailable(f"spec file for {row['strategy_id']} not found")
    if rec.version != row["version"]:
        raise DataUnavailable(f"spec file for {row['strategy_id']} is v{rec.version}, the loop is at v{row['version']}")
    return StrategySpec.from_dict(rec.raw), content_hash(rec.raw)


def set_file_status(ctx: Ctx, stage: str, sid: str, status: str) -> None:
    try:
        ctx.ports.specs.set_status(sid, status)
    except Exception as exc:   # noqa: BLE001 - the database is the record; the file is a label
        ctx.alert("critical", f"{sid}: could not write status {status} to the spec file", str(exc), stage, sid)


def reject(ctx: Ctx, stage: str, row: dict, reason: str) -> None:
    ctx.state.transition(row["strategy_id"], row["version"], "rejected", reason, run_id=ctx.run_id, stage=stage)
    set_file_status(ctx, stage, row["strategy_id"], "rejected")


def evidence_hash_ok(ctx: Ctx, row: dict, kind: str, spec_hash: str) -> None:
    ev = ctx.state.latest_evidence(row["strategy_id"], row["version"], kind)
    if ev is not None and ev["spec_hash"] != spec_hash:
        raise DataUnavailable(f"{row['strategy_id']}: the spec changed after its {kind} (bump the version to start over)")


# --- 1 propose --------------------------------------------------------------------------------------

def stage_propose(ctx: Ctx) -> StageResult:
    res = StageResult("propose")
    ports, cfg, st = ctx.ports, ctx.cfg, ctx.state

    def used() -> int:
        return sum(int(s["result"].get("new", 0)) for s in st.steps(ctx.run_id, "propose") if s["status"] == "done")

    specs = ports.specs.list_specs()
    for rec in specs:
        run_item(ctx, res, f"spec:{key(rec.spec_id, rec.version)}", lambda rec=rec: _adopt(ctx, rec))

    known_arxiv = {a for e in _safe_entries(ctx) for a in e.arxiv}
    for src in ports.idea_sources:
        run_item(ctx, res, f"source:{src.name}", lambda src=src: _pull(ctx, src, known_arxiv, cfg.max_new_ideas - used()))

    handled = ports.catalog.handled()
    covered, source_slugs = _covered_by_specs(specs)
    for e in sorted(_safe_entries(ctx), key=lambda e: e.id):
        if e.status_kind not in cfg.catalog_kinds or e.id in handled or e.id in covered:
            continue
        if any(e.id in src for src in source_slugs):
            continue
        if st.idea(f"catalog:{e.id}") is not None:
            continue
        if used() >= cfg.max_new_ideas:
            break
        run_item(ctx, res, f"catalog:{e.id}", lambda e=e: _catalog_idea(ctx, e))
    return res


def _safe_entries(ctx: Ctx) -> list:
    try:
        return ctx.ports.catalog.entries()
    except catalog_mod.CatalogError as exc:
        ctx.alert("critical", "research catalog does not parse", str(exc), "propose")
        return []


def _covered_by_specs(specs) -> tuple[set[str], list[str]]:
    """Catalog ids that spec files already name, and the slugs of their source lines (for loose matches)."""
    ids, sources = set(), []
    for r in specs:
        prov = r.raw.get("provenance") or {}
        if not isinstance(prov, dict):
            continue
        if prov.get("catalog_id"):
            ids.add(str(prov["catalog_id"]))
        if prov.get("source"):
            sources.append(catalog_mod.slug(str(prov["source"])))
    return ids, sources


def _catalog_idea(ctx: Ctx, e) -> dict:
    idea = Idea(id=f"catalog:{e.id}", source="catalog", title=e.name,
                summary=f"{e.source}. Data: {e.data}. Data today: {e.data_today}. Catalog status: {e.status}",
                arxiv=tuple(e.arxiv), assets=tuple(e.assets), family=e.family, horizon=e.horizon, catalog_id=e.id)
    new = ctx.state.add_idea(idea)
    if new:
        ctx.audit("propose", "idea_added", idea.id, source="catalog", title=e.name)
    return {"new": int(new), "idea": idea.id}


def _pull(ctx: Ctx, src, known_arxiv: set[str], budget: int) -> dict:
    ideas = src.fetch()
    for msg in getattr(src, "errors", []):
        ctx.alert("warning", f"idea file rejected ({src.name})", msg, "propose")
    new = dup = over = 0
    for idea in ideas:
        if ctx.state.idea(idea.id) is not None:
            continue
        if idea.arxiv and set(idea.arxiv) <= known_arxiv:
            ctx.state.add_idea(idea, status="duplicate")
            ctx.state.set_idea(idea.id, "duplicate", "already in the research catalog")
            ctx.audit("propose", "idea_duplicate", idea.id, source=idea.source, arxiv=list(idea.arxiv))
            dup += 1
            continue
        if new >= budget:
            over += 1
            continue
        ctx.state.add_idea(idea)
        ctx.audit("propose", "idea_added", idea.id, source=idea.source, title=idea.title)
        new += 1
    return {"new": new, "duplicate": dup, "over_budget": over, "seen": len(ideas)}


def _adopt(ctx: Ctx, rec) -> dict:
    """Register a spec file the loop has not seen, or notice that a known one changed or claims a status."""
    st = ctx.state
    h = content_hash(rec.raw)
    row = st.strategy(rec.spec_id, rec.version)
    if row is None:
        prior = st.strategy(rec.spec_id)
        if prior is not None and prior["version"] < rec.version and prior["status"] in ("proposed", "screened"):
            st.transition(prior["strategy_id"], prior["version"], "rejected", f"superseded by v{rec.version}",
                          run_id=ctx.run_id, stage="propose")
        elif prior is not None and prior["version"] < rec.version and prior["status"] in ("validated", "paper"):
            st.transition(prior["strategy_id"], prior["version"], "retired", f"superseded by v{rec.version}",
                          run_id=ctx.run_id, stage="propose")
        claimed = rec.status
        status, unverified = "proposed", ""
        if claimed in CLAIMS_THAT_MATTER:
            unverified = f"spec file says status {claimed!r}; no loop evidence"
            status = claimed if claimed in ("validated", "paper") else "proposed"
            ctx.alert("critical", f"{rec.spec_id} v{rec.version}: spec file claims {claimed} without loop evidence",
                      "it is not paper-eligible until it passes the loop's gates on its current content", "propose", rec.spec_id)
        elif claimed in ("rejected", "retired"):
            status = claimed
        raw = rec.raw
        st.register(rec.spec_id, rec.version, rec.path, h, family=str(raw.get("family", "")),
                    asset_class=str(raw.get("asset_class", "")), idea_id=str((raw.get("provenance") or {}).get("idea_id", "")),
                    status=status, unverified=unverified)
        ctx.audit("propose", "spec_adopted", rec.spec_id, version=rec.version, file_status=claimed, status=status, path=rec.path)
        return {"new": 0, "adopted": 1, "status": status}
    if row["spec_hash"] != h:
        if row["status"] == "proposed":
            st.update_spec(rec.spec_id, rec.version, rec.path, h)
            ctx.audit("propose", "spec_edited_before_screen", rec.spec_id, version=rec.version)
        else:
            ctx.alert("warning", f"{rec.spec_id} v{rec.version}: spec content changed after {row['status']}",
                      "evidence belongs to the old content; bump the version to re-run the gates", "propose", rec.spec_id)
    if rec.status != row["status"] and rec.status in ("validated", "paper", "live") and row["status"] != rec.status:
        ctx.alert("critical", f"{rec.spec_id} v{rec.version}: spec file says {rec.status}, the loop has {row['status']}",
                  "the file was edited outside the loop; the loop's record is the one that counts", "propose", rec.spec_id)
    return {"new": 0, "adopted": 0}


# --- 2 implement ------------------------------------------------------------------------------------

def stage_implement(ctx: Ctx) -> StageResult:
    res = StageResult("implement")
    ideas = [i for s in ("new", "awaiting_spec", "spec_invalid") for i in ctx.state.ideas(s)]
    written = 0
    for row in ideas:
        out = run_item(ctx, res, f"idea:{row['idea_id']}", lambda row=row: _implement(ctx, row))
        written += bool(out and out.get("outcome") == "specced")
        if written >= ctx.cfg.max_drafts_per_run:
            break
    return res


def idea_from_row(row: dict) -> Idea:
    p = row["payload"]
    return Idea(id=p["id"], source=p["source"], title=p["title"], summary=p.get("summary", ""), url=p.get("url", ""),
                arxiv=tuple(p.get("arxiv", ())), assets=tuple(p.get("assets", ())), family=p.get("family", "other"),
                horizon=p.get("horizon", ""), catalog_id=p.get("catalog_id", ""), published=p.get("published", ""),
                extra=p.get("extra", {}))


def normalise_draft(ctx: Ctx, idea: Idea, raw: dict) -> tuple[dict, list[str]]:
    """The draft as the loop will store it: status proposed, no performance claims, provenance filled in."""
    d = copy.deepcopy(raw)
    notes = []
    if d.get("status") not in (None, "proposed"):
        notes.append(f"draft claimed status {d.get('status')!r}; forced to proposed")
    d["status"] = "proposed"
    if d.pop("stats", None):
        notes.append("draft carried performance stats; removed")
    try:
        d["version"] = int(d.get("version", 1))
    except (TypeError, ValueError):
        d["version"] = 1
    prov = dict(d.get("provenance") or {}) if isinstance(d.get("provenance"), dict) else {}
    prov.update({"source": prov.get("source") or f"{idea.source}: {idea.title}", "idea_id": idea.id,
                 "loop_run": ctx.run_id, "author": prov.get("author") or "research-loop draft",
                 "created": ctx.now().strftime("%Y-%m-%d")})
    if idea.arxiv:
        prov["arxiv"] = list(idea.arxiv)
    if idea.catalog_id:
        prov["catalog_id"] = idea.catalog_id
    d["provenance"] = prov
    return d, notes


def _implement(ctx: Ctx, row: dict) -> dict:
    st, idea = ctx.state, idea_from_row(row)
    try:
        raw = ctx.ports.drafter.draft(idea)
    except DraftError as exc:
        st.set_idea(idea.id, "spec_invalid", str(exc))
        ctx.audit("implement", "draft_unreadable", idea.id, error=str(exc))
        return {"outcome": "spec_invalid", "errors": [str(exc)]}
    if raw is None:
        if row["status"] != "awaiting_spec":
            st.set_idea(idea.id, "awaiting_spec", "no draft spec available yet")
            ctx.audit("implement", "awaiting_spec", idea.id)
        return {"outcome": "awaiting_spec"}
    dh = content_hash(raw)
    if row["status"] == "spec_invalid" and row["draft_hash"] == dh:
        return {"outcome": "spec_invalid", "errors": [row["note"]], "unchanged": True}
    d, notes = normalise_draft(ctx, idea, raw)
    errs = _validate_draft(ctx, d)
    if errs:
        st.set_idea(idea.id, "spec_invalid", "; ".join(errs)[:500], draft_hash=dh)
        ctx.audit("implement", "draft_invalid", idea.id, errors=errs)
        return {"outcome": "spec_invalid", "errors": errs}
    path = ctx.ports.specs.write_new(d)
    again = StrategySpec.load(path)                      # the file as `tradex check` will read it
    errs = again.validate()
    if errs:
        raise RuntimeError(f"written spec {path} fails the schema check: {errs}")
    st.register(d["id"], d["version"], path, content_hash(d), family=str(d.get("family", "")),
                asset_class=str(d["asset_class"]), idea_id=idea.id)
    st.set_idea(idea.id, "specced", f"spec {d['id']} v{d['version']}", draft_hash=dh)
    ctx.audit("implement", "spec_written", d["id"], idea=idea.id, path=path, notes=notes)
    return {"outcome": "specced", "strategy_id": d["id"], "version": d["version"], "notes": notes}


def _validate_draft(ctx: Ctx, d: dict) -> list[str]:
    try:
        spec = StrategySpec.from_dict(d)
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        return [f"draft is not a spec: {type(exc).__name__}: {exc}"]
    errs = spec.validate()
    if ctx.ports.specs.get(spec.id) is not None or ctx.state.strategy(spec.id) is not None:
        errs.append(f"strategy id {spec.id} already exists")
    return errs


# --- 3 screen ---------------------------------------------------------------------------------------

def stage_screen(ctx: Ctx) -> StageResult:
    res = StageResult("screen")
    rows = [r for r in ctx.state.strategies("proposed") if not r["unverified"]]
    for row in rows[:ctx.cfg.max_screens_per_run]:
        run_item(ctx, res, key(row["strategy_id"], row["version"]), lambda row=row: _screen(ctx, row))
    return res


def _screen(ctx: Ctx, row: dict) -> dict:
    cfg = ctx.cfg
    spec, h = load_spec(ctx, row)
    errs = spec.validate()
    if errs:
        reject(ctx, "screen", row, "spec fails the schema check: " + "; ".join(errs))
        return {"outcome": "rejected", "reasons": errs}
    frames = research_frames(ctx, spec)
    m = ctx.ports.evaluator.screen(spec, frames, ctx.ports.trials, ctx.run_id)
    bars = max(len(df) for df in frames.values())
    reasons = []
    if bars < cfg.screen_min_bars:
        reasons.append(f"only {bars} research-window bars (need {cfg.screen_min_bars})")
    if m["trades"] < cfg.screen_min_trades:
        reasons.append(f"{m['trades']} trades (need {cfg.screen_min_trades})")
    if (m["profit_factor"] or 0) < cfg.screen_min_profit_factor:
        reasons.append(f"profit factor {m['profit_factor']} after costs (need {cfg.screen_min_profit_factor})")
    passed = not reasons
    data = m | {"bars": bars, "symbols": sorted(frames), "data_key": data_key(ctx, spec, frames), "reasons": reasons}
    ctx.state.add_evidence(row["strategy_id"], row["version"], "screen", ctx.run_id, passed, h, data)
    if passed:
        ctx.state.transition(row["strategy_id"], row["version"], "screened", "passed the screen", run_id=ctx.run_id, stage="screen")
        set_file_status(ctx, "screen", row["strategy_id"], "screened")
    else:
        reject(ctx, "screen", row, "; ".join(reasons))
    return {"outcome": "screened" if passed else "rejected", "reasons": reasons} | {
        k: m[k] for k in ("trades", "profit_factor", "sharpe", "max_drawdown")}


# --- 4 walk-forward ---------------------------------------------------------------------------------

def stage_walk_forward(ctx: Ctx) -> StageResult:
    res = StageResult("walk_forward")
    rows = [r for r in ctx.state.strategies("screened") if not r["unverified"]]
    for row in rows[:ctx.cfg.max_walk_forwards_per_run]:
        run_item(ctx, res, key(row["strategy_id"], row["version"]), lambda row=row: _walk_forward(ctx, row))
    return res


def _walk_forward(ctx: Ctx, row: dict) -> dict:
    from tradex.research.gate import gate_checks
    spec, h = load_spec(ctx, row)
    evidence_hash_ok(ctx, row, "screen", h)
    frames = research_frames(ctx, spec)
    t0 = _time.time()
    rep = ctx.ports.evaluator.walk_forward(spec, frames, ctx.ports.trials, data_key(ctx, spec, frames))
    th = ctx.ports.evaluator.thresholds
    checks = gate_checks(rep.oos, th)
    failing = [c["rung"] for c in checks if not c["ok"]]
    ledger_n = ctx.ports.trials.count(spec.id)
    if rep.n_trials < ledger_n:
        raise RuntimeError(f"deflated Sharpe used N={rep.n_trials}, the trial ledger holds {ledger_n} for {spec.id}")
    passed = not failing and rep.status == "validated"
    o = rep.oos
    data = {"checks": checks, "failing": failing, "reasons": list(rep.reasons), "n_trials": rep.n_trials,
            "n_trials_this_run": rep.n_trials_this_run, "ledger_count": ledger_n, "oos_trades": int(o.get("trades", 0)),
            "profit_factor": num(o.get("profit_factor")), "sharpe": num(o.get("sharpe")), "dsr": num(o.get("dsr")),
            "max_drawdown": num(o.get("max_drawdown")), "positive_folds": num(o.get("positive_folds")),
            "folds": len(rep.folds), "recommended_params": rep.recommended_params,
            "param_stability": num(rep.param_stability), "data_key": data_key(ctx, spec, frames),
            "seconds": round(_time.time() - t0, 1), "thresholds": {k: getattr(th, k) for k in th.__dataclass_fields__}}
    ctx.state.add_evidence(row["strategy_id"], row["version"], "walk_forward", ctx.run_id, passed, h, data)
    if passed:
        b = health_mod.baseline_from_returns(rep.oos_returns)
        ctx.state.set_baseline(row["strategy_id"], row["version"], b.mean, b.std, b.n, b.max_drawdown)
        ctx.state.transition(row["strategy_id"], row["version"], "validated", "passed the stage-3 gate",
                             run_id=ctx.run_id, stage="walk_forward")
        set_file_status(ctx, "walk_forward", row["strategy_id"], "validated")
    else:
        reject(ctx, "walk_forward", row, "; ".join(rep.reasons) or f"failed rungs {failing}")
    return {"outcome": "validated" if passed else "rejected", "failing": failing, "n_trials": rep.n_trials,
            "dsr": data["dsr"], "oos_trades": data["oos_trades"], "profit_factor": data["profit_factor"]}


# --- 5 holdout --------------------------------------------------------------------------------------

def stage_holdout(ctx: Ctx) -> StageResult:
    res = StageResult("holdout")
    for row in ctx.state.strategies("validated"):
        if row["unverified"] or ctx.state.latest_evidence(row["strategy_id"], row["version"], "holdout"):
            continue
        run_item(ctx, res, key(row["strategy_id"], row["version"]), lambda row=row: _holdout(ctx, row))
    return res


def _holdout(ctx: Ctx, row: dict) -> dict:
    sid, ver, st, hp = row["strategy_id"], row["version"], ctx.state, ctx.ports.holdout
    spec, h = load_spec(ctx, row)
    evidence_hash_ok(ctx, row, "walk_forward", h)
    wf = st.latest_evidence(sid, ver, "walk_forward")
    if wf is None or not wf["passed"]:
        raise RuntimeError(f"{sid} v{ver} is validated without passing walk-forward evidence")
    start = hp.start
    if hp.has_looked(sid, ver):      # the one look was used and no result was recorded: it cannot be repeated
        reason = "the holdout look was used earlier without a recorded result; a new version is needed"
        st.add_evidence(sid, ver, "holdout", ctx.run_id, False, h, {"reasons": [reason], "lost": True})
        reject(ctx, "holdout", row, reason)
        ctx.alert("critical", f"{sid} v{ver}: holdout look lost", reason, "holdout", sid)
        return {"outcome": "rejected", "reasons": [reason]}
    full = ctx.ports.data.frames(spec)               # reading the frames is not the look; the cut below hides the holdout
    research = {s: df for s, df in hp.research_view(full, spec.signal_tf).items() if len(df)}
    reach = max((df.index[-1] + duration(spec.signal_tf) for df in full.values() if len(df)), default=None)
    if reach is None or reach < start + pd.Timedelta(days=ctx.cfg.holdout_min_days):
        raise DataUnavailable(f"{sid}: bars reach {reach}, holdout needs {ctx.cfg.holdout_min_days} days after {start}; "
                              f"the look is not used yet")
    # one recorded look, from here on. Anything that goes wrong after this line is recorded as a failed holdout.
    held = hp.look(full, sid, ver, f"research loop {ctx.run_id}: stage 5")
    try:
        combined = {s: pd.concat([research.get(s, df.iloc[0:0]), held[s]]).sort_index() for s, df in full.items() if s in held}
        combined = {s: df[~df.index.duplicated()] for s, df in combined.items() if len(df)}
        m = ctx.ports.evaluator.holdout(spec, dict(wf["data"].get("recommended_params") or {}), combined, start)
    except Exception as exc:    # noqa: BLE001
        reason = f"evaluation failed after the look was used: {type(exc).__name__}: {exc}"
        st.add_evidence(sid, ver, "holdout", ctx.run_id, False, h, {"reasons": [reason], "lost": True})
        reject(ctx, "holdout", row, reason)
        ctx.alert("critical", f"{sid} v{ver}: holdout evaluation error", reason, "holdout", sid)
        return {"outcome": "rejected", "reasons": [reason]}
    th, cfg, reasons = ctx.ports.evaluator.thresholds, ctx.cfg, []
    if m["trades"] < cfg.holdout_min_trades:
        reasons.append(f"{m['trades']} holdout trades (need {cfg.holdout_min_trades})")
    if (m["profit_factor"] or 0) < th.min_profit_factor:
        reasons.append(f"holdout profit factor {m['profit_factor']} (need {th.min_profit_factor})")
    if (m["max_drawdown"] if m["max_drawdown"] is not None else 0) < th.max_drawdown:
        reasons.append(f"holdout drawdown {m['max_drawdown']} (limit {th.max_drawdown})")
    if (m["sharpe"] or 0) <= 0:
        reasons.append(f"holdout Sharpe {m['sharpe']} is not positive")
    passed = not reasons
    st.add_evidence(sid, ver, "holdout", ctx.run_id, passed, h, m | {"reasons": reasons, "holdout_start": str(start)})
    if not passed:
        reject(ctx, "holdout", row, "; ".join(reasons))
    else:
        ctx.audit("holdout", "holdout_passed", sid, version=ver, trades=m["trades"], profit_factor=m["profit_factor"])
    return {"outcome": "holdout_passed" if passed else "rejected", "reasons": reasons} | {
        k: m[k] for k in ("trades", "profit_factor", "sharpe", "max_drawdown")}


# --- 6 promote --------------------------------------------------------------------------------------

def stage_promote(ctx: Ctx) -> StageResult:
    res = StageResult("promote")
    st, cfg = ctx.state, ctx.cfg
    paper = st.strategies("paper")
    fam_count: dict[str, int] = {}
    for r in paper:
        fam_count[r["family"]] = fam_count.get(r["family"], 0) + 1
    slots = cfg.max_paper - len(paper)
    ranked = []
    for row in st.strategies("validated"):
        try:
            h = ctx.ports.specs.content_hash(row["strategy_id"])
        except KeyError:
            h = None
        problems = st.eligibility(row["strategy_id"], row["version"], h)
        if problems:
            res.items[key(row["strategy_id"], row["version"])] = {"outcome": "not_eligible", "problems": problems}
            continue
        wf = st.latest_evidence(row["strategy_id"], row["version"], "walk_forward")["data"]
        ho = st.latest_evidence(row["strategy_id"], row["version"], "holdout")["data"]
        ranked.append(((wf.get("dsr") or 0.0, ho.get("sharpe") or 0.0), row, h))
    for _, row, h in sorted(ranked, key=lambda x: x[0], reverse=True):
        k = key(row["strategy_id"], row["version"])
        if slots <= 0:
            res.items[k] = {"outcome": "waiting", "reason": f"paper queue full ({cfg.max_paper})"}
            continue
        if fam_count.get(row["family"], 0) >= cfg.max_paper_per_family:
            res.items[k] = {"outcome": "waiting", "reason": f"family {row['family']!r} already has "
                                                            f"{cfg.max_paper_per_family} in paper"}
            continue
        out = run_item(ctx, res, k, lambda row=row, h=h: _promote(ctx, row, h))
        if out and out.get("outcome") == "promoted":
            slots -= 1
            fam_count[row["family"]] = fam_count.get(row["family"], 0) + 1
        elif out and out.get("outcome") == "recommended":
            slots -= 1
            fam_count[row["family"]] = fam_count.get(row["family"], 0) + 1
    return res


def _promote(ctx: Ctx, row: dict, spec_hash: str) -> dict:
    sid, ver = row["strategy_id"], row["version"]
    ev = {"apply": ctx.cfg.apply}
    if not ctx.cfg.apply:
        ctx.state.add_evidence(sid, ver, "promotion", ctx.run_id, None, spec_hash, ev | {"decision": "recommended"})
        ctx.alert("info", f"{sid} v{ver}: recommended for the paper queue", "apply is off; no status was changed", "promote", sid)
        return {"outcome": "recommended"}
    try:
        ctx.state.transition(sid, ver, "paper", "passed screen, walk-forward and holdout", run_id=ctx.run_id,
                             stage="promote", spec_hash=spec_hash)
    except (NotEligible, IllegalTransition) as exc:
        return {"outcome": "refused", "reason": str(exc)}
    ctx.state.add_evidence(sid, ver, "promotion", ctx.run_id, True, spec_hash, ev | {"decision": "promoted"})
    set_file_status(ctx, "promote", sid, "paper")
    ctx.alert("info", f"{sid} v{ver}: promoted to the paper queue", "", "promote", sid)
    return {"outcome": "promoted"}


# --- 7 health ---------------------------------------------------------------------------------------

def stage_health(ctx: Ctx) -> StageResult:
    res = StageResult("health")
    for row in ctx.state.strategies("paper"):
        run_item(ctx, res, key(row["strategy_id"], row["version"]), lambda row=row: _health(ctx, row))
    horizon = pd.Timedelta(weeks=ctx.cfg.stale_weeks)
    for row in ctx.state.strategies("validated"):
        wf = ctx.state.latest_evidence(row["strategy_id"], row["version"], "walk_forward")
        if wf and ctx.now() - pd.Timestamp(wf["time"]) > horizon:
            run_item(ctx, res, f"stale:{key(row['strategy_id'], row['version'])}", lambda row=row: _stale(ctx, row))
    return res


def _stale(ctx: Ctx, row: dict) -> dict:
    reason = f"validated for more than {ctx.cfg.stale_weeks} weeks without reaching the paper queue"
    ctx.state.transition(row["strategy_id"], row["version"], "retired", reason, run_id=ctx.run_id, stage="health")
    set_file_status(ctx, "health", row["strategy_id"], "retired")
    return {"outcome": "retired_stale", "reason": reason}


def _health(ctx: Ctx, row: dict) -> dict:
    sid, ver, cfg = row["strategy_id"], row["version"], ctx.cfg
    base = ctx.state.baseline(sid, ver)
    if base is None or base["std"] <= 0:
        ctx.alert("warning", f"{sid} v{ver}: in paper without a usable baseline",
                  "no walk-forward baseline recorded by the loop; health cannot be judged", "health", sid)
        return {"outcome": "no_baseline"}
    since = pd.Timestamp(row["promoted_at"]) if row["promoted_at"] else None
    try:
        rets = ctx.ports.live.daily_returns(sid, ver, since)
    except LiveDataMissing as exc:
        ctx.audit("health", "no_paper_data", sid, reason=str(exc))
        return {"outcome": "no_data", "reason": str(exc)}
    if rets is None or len(rets) < cfg.health_min_obs:
        n = 0 if rets is None else len(rets)
        ctx.audit("health", "insufficient_history", sid, n=n, need=cfg.health_min_obs)
        return {"outcome": "insufficient_history", "n": n, "need": cfg.health_min_obs}
    b = health_mod.Baseline(base["mean"], base["std"], base["n"], base["max_drawdown"])
    hl = health_mod.assess(rets, b, k=cfg.cusum_k, h=cfg.cusum_h, warn_frac=cfg.cusum_warn_frac,
                           mean_shrink=cfg.cusum_mean_shrink, dd_multiple=cfg.health_dd_multiple)
    spec_hash = ctx.ports.specs.content_hash(sid) if ctx.ports.specs.get(sid) else row["spec_hash"]
    data = {"n": hl.n, "cusum_min": num(hl.cusum_min), "cusum_last": num(hl.cusum_last), "drawdown": num(hl.drawdown),
            "reference_mean": num(hl.reference_mean, 6), "sigma": num(hl.sigma, 6), "h": cfg.cusum_h,
            "cusum_alarm": hl.cusum_alarm, "drawdown_alarm": hl.drawdown_alarm}
    ctx.state.add_evidence(sid, ver, "health", ctx.run_id, not hl.alarm, spec_hash, data)
    if hl.alarm:
        why = ("CUSUM alarm" if hl.cusum_alarm else "") + (" and " if hl.cusum_alarm and hl.drawdown_alarm else "") + \
              ("drawdown beyond the baseline" if hl.drawdown_alarm else "")
        reason = f"{why}: CUSUM min {hl.cusum_min:.1f} (alarm at -{cfg.cusum_h}), drawdown {hl.drawdown:.1%} over {hl.n} days"
        if cfg.apply:
            ctx.state.transition(sid, ver, "retired", reason, run_id=ctx.run_id, stage="health")
            set_file_status(ctx, "health", sid, "retired")
            ctx.alert("critical", f"{sid} v{ver}: demoted out of paper", reason, "health", sid)
            return data | {"outcome": "retired", "reason": reason}
        ctx.alert("critical", f"{sid} v{ver}: would be demoted out of paper (apply is off)", reason, "health", sid)
        return data | {"outcome": "retire_recommended", "reason": reason}
    if hl.cusum_warn:
        ctx.alert("warning", f"{sid} v{ver}: CUSUM warning", f"min {hl.cusum_min:.1f} of alarm -{cfg.cusum_h}", "health", sid)
        return data | {"outcome": "warning"}
    return data | {"outcome": "healthy"}


# --- 8 report ---------------------------------------------------------------------------------------

def stage_report(ctx: Ctx) -> StageResult:
    from tradex.research.loop.report import write_report
    res = StageResult("report")
    path = write_report(ctx)
    res.done = 1
    res.items["report"] = {"path": str(path)}
    return res


STAGE_FUNCS: dict[str, Callable[[Ctx], StageResult]] = {
    "propose": stage_propose, "implement": stage_implement, "screen": stage_screen, "walk_forward": stage_walk_forward,
    "holdout": stage_holdout, "promote": stage_promote, "health": stage_health, "report": stage_report,
}
