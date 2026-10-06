"""The weekly report: what the loop did this run and where every strategy stands, as markdown (plus a JSON twin).

Written to ``research/results/loop/weekly-<run id>.md``. The summary is built from the run state, so a run
that was resumed still reports everything it did. Text that came from outside (paper titles, agent notes) is
escaped before it goes into a table or a heading.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

STAGE_TITLES = {
    "propose": "1. Propose", "implement": "2. Implement", "screen": "3. Screen", "walk_forward": "4. Walk-forward",
    "holdout": "5. Holdout", "promote": "6. Paper queue", "health": "7. Health", "report": "8. Report",
}


def md(text: Any, limit: int = 160) -> str:
    """Escape untrusted text for one markdown table cell."""
    s = re.sub(r"[\x00-\x1f\x7f]+", " ", str(text if text is not None else ""))
    s = s.replace("\\", "\\\\").replace("|", "\\|").replace("`", "'").replace("<", "&lt;").replace(">", "&gt;")
    s = s.replace("[", "\\[").replace("]", "\\]")
    return (s[:limit] + "...") if len(s) > limit else s


def fmt(x: Any, nd: int = 2) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def build_summary(ctx) -> dict[str, Any]:
    st = ctx.state
    run = st.run(ctx.run_id) or {}
    steps = st.steps(ctx.run_id)
    by_stage: dict[str, dict[str, int]] = {}
    for s in steps:
        c = by_stage.setdefault(s["stage"], {"done": 0, "blocked": 0, "failed": 0})
        c[s["status"]] = c.get(s["status"], 0) + 1
    funnel = {}
    for r in st.strategies():
        funnel[r["status"]] = funnel.get(r["status"], 0) + 1
    touched = sorted({s["item"].split(":")[0] for s in steps if s["stage"] in ("screen", "walk_forward", "holdout")})
    trial_n = {sid: ctx.ports.trials.count(sid) for sid in touched}
    alerts = [a["detail"] | {"stage": a["stage"], "strategy_id": a["strategy_id"]}
              for a in st.audit_rows(ctx.run_id) if a["event"] == "alert"]
    ok, bad = st.verify_audit()
    return {
        "run_id": ctx.run_id, "started": run.get("started"), "apply": ctx.cfg.apply, "generated": ctx.now().isoformat(),
        "by_stage": by_stage, "funnel": funnel, "steps": steps, "alerts": alerts, "trial_counts": trial_n,
        "paper": [r for r in st.strategies("paper")], "validated": [r for r in st.strategies("validated")],
        "awaiting_spec": [{"idea_id": i["idea_id"], "title": i["title"], "source": i["source"], "note": i["note"]}
                          for s in ("awaiting_spec", "spec_invalid") for i in st.ideas(s)],
        "unverified": [r for r in st.strategies() if r["unverified"]],
        "audit_ok": ok, "audit_bad_row": bad,
        "data_source": getattr(ctx.ports.data, "source", "?"),
    }


def render(summary: dict[str, Any]) -> str:
    s = summary
    L: list[str] = []
    L += [f"# Research loop, {md(s['run_id'])}", "",
          f"Generated {md(s['generated'])}. Data: {md(s['data_source'])}. Status changes into and out of paper: "
          f"**{'applied' if s['apply'] else 'not applied (recommendations only)'}**.", ""]
    ok = "intact" if s["audit_ok"] else f"BROKEN at row {s['audit_bad_row']}"
    L += [f"Audit trail: {ok}.", ""]

    L += ["## Where the strategies stand", "", "| Status | Strategies |", "|---|---|"]
    for k in ("proposed", "screened", "validated", "paper", "rejected", "retired"):
        L.append(f"| {k} | {s['funnel'].get(k, 0)} |")
    L.append("")

    L += ["## This run", "", "| Stage | Done | Blocked | Failed |", "|---|---|---|---|"]
    for k, title in STAGE_TITLES.items():
        c = s["by_stage"].get(k)
        if c:
            L.append(f"| {title} | {c.get('done', 0)} | {c.get('blocked', 0)} | {c.get('failed', 0)} |")
    L.append("")

    def rows(stage: str) -> list[dict]:
        return [x for x in s["steps"] if x["stage"] == stage and x["status"] == "done"]

    ideas = [x for x in rows("propose") if x["result"].get("new")]
    if ideas:
        L += ["## New ideas", ""] + [f"- `{md(x['item'])}`: {x['result']['new']} new" for x in ideas] + [""]
    drafted = [x for x in rows("implement") if x["result"].get("outcome") == "specced"]
    if drafted:
        L += ["## Specs written", ""] + [f"- `{md(x['result'].get('strategy_id'))}` from `{md(x['item'])}`" for x in drafted] + [""]

    sc = rows("screen")
    if sc:
        L += ["## Screen", "", "| Strategy | Trades | PF | Sharpe | Max DD | Result |", "|---|---|---|---|---|---|"]
        for x in sc:
            r = x["result"]
            L.append(f"| `{md(x['item'])}` | {fmt(r.get('trades'))} | {fmt(r.get('profit_factor'))} | {fmt(r.get('sharpe'))} | "
                     f"{fmt(r.get('max_drawdown'))} | **{md(r.get('outcome'))}** {md('; '.join(r.get('reasons', [])))} |")
        L.append("")

    wf = rows("walk_forward")
    if wf:
        L += ["## Walk-forward (stage-3 gate)", "",
              "| Strategy | OOS trades | PF | DSR | N trials | Failing rungs | Result |", "|---|---|---|---|---|---|---|"]
        for x in wf:
            r = x["result"]
            L.append(f"| `{md(x['item'])}` | {fmt(r.get('oos_trades'))} | {fmt(r.get('profit_factor'))} | {fmt(r.get('dsr'))} | "
                     f"{fmt(r.get('n_trials'))} | {md(', '.join(r.get('failing', [])) or '-')} | **{md(r.get('outcome'))}** |")
        L += ["", "N is every parameter set ever recorded for the strategy in the global trial ledger (screens included).", ""]

    ho = rows("holdout")
    if ho:
        L += ["## Holdout (one recorded look per version)", "",
              "| Strategy | Trades | PF | Sharpe | Max DD | Result |", "|---|---|---|---|---|---|"]
        for x in ho:
            r = x["result"]
            L.append(f"| `{md(x['item'])}` | {fmt(r.get('trades'))} | {fmt(r.get('profit_factor'))} | {fmt(r.get('sharpe'))} | "
                     f"{fmt(r.get('max_drawdown'))} | **{md(r.get('outcome'))}** {md('; '.join(r.get('reasons', [])))} |")
        L.append("")

    pr = [x for x in s["steps"] if x["stage"] == "promote"]
    if pr:
        L += ["## Paper queue decisions", ""]
        for x in pr:
            r = x["result"]
            L.append(f"- `{md(x['item'])}`: **{md(r.get('outcome'))}** {md(r.get('reason', ''))}")
        L.append("")

    hl = [x for x in s["steps"] if x["stage"] == "health"]
    if hl:
        L += ["## Health of paper strategies", "",
              "| Strategy | Days | CUSUM min (alarm at -h) | Drawdown | Outcome |", "|---|---|---|---|---|"]
        for x in hl:
            r = x["result"]
            L.append(f"| `{md(x['item'])}` | {fmt(r.get('n'))} | {fmt(r.get('cusum_min'), 1)} (-{fmt(r.get('h'), 1)}) | "
                     f"{fmt(r.get('drawdown'), 3)} | **{md(r.get('outcome'))}** {md(r.get('reason', ''))} |")
        L.append("")

    blocked = [x for x in s["steps"] if x["status"] in ("blocked", "failed")]
    if blocked:
        L += ["## Blocked or failed (retried next run)", "", "| Stage | Item | Status | Why |", "|---|---|---|---|"]
        for x in blocked:
            L.append(f"| {md(x['stage'])} | `{md(x['item'])}` | {md(x['status'])} | {md(x['result'].get('reason', x['error']), 240)} |")
        L.append("")

    if s["awaiting_spec"]:
        L += ["## Ideas waiting for a draft spec", "", "| Idea | Source | Title | Note |", "|---|---|---|---|"]
        for i in s["awaiting_spec"][:50]:
            L.append(f"| `{md(i['idea_id'])}` | {md(i['source'])} | {md(i['title'])} | {md(i['note'])} |")
        L += ["", "Drop a draft in `research/ideas/specs/<idea id>.yaml` (colons in the id become `__`). Drafts are re-validated; "
                  "their status is forced to proposed.", ""]

    if s["unverified"]:
        L += ["## Status claims without evidence", ""] + [
            f"- `{md(r['strategy_id'])}` v{r['version']}: {md(r['unverified'])}" for r in s["unverified"]] + [""]

    if s["alerts"]:
        L += ["## Alerts this run", ""] + [f"- {md(a['level'])}: {md(a['title'])} {md(a['detail'], 240)}" for a in s["alerts"]] + [""]

    if s["trial_counts"]:
        L += ["## Trial ledger (DSR's N)", ""] + [f"- `{md(k)}`: {v} parameter sets" for k, v in s["trial_counts"].items()] + [""]
    return "\n".join(L).rstrip() + "\n"


def write_report(ctx) -> Path:
    out = Path(ctx.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = build_summary(ctx)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", ctx.run_id)
    md_path = out / f"weekly-{safe}.md"
    md_path.write_text(render(summary))
    (out / f"weekly-{safe}.json").write_text(json.dumps(summary, indent=1, default=str))
    ctx.audit("report", "report_written", path=str(md_path))
    return md_path
