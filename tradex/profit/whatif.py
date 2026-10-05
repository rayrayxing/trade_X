"""Counterfactual ledger reports (profit spec P8): what the plans we did not trade would have done.

``tradex.core.counterfactual`` follows every blocked plan to the exit it would have had and writes
a ``counterfactual`` row. ``filter_report`` there says, per gate, how many and the mean R. This module
reads the same ledger and answers the questions a decision needs:

- ``gate_value``: per gate, did blocking pay? Net effect in R is the R we did NOT take, negated: a gate
  whose blocked plans would have lost 12R in total saved 12R. A 90% interval on the mean says whether
  that is distinguishable from nothing; below ``min_plans`` it says "too few" instead of a verdict.
- ``reason_breakdown``: the same by the veto's reason text with numbers masked, so "score 0.18 below
  0.25" and "score 0.21 below 0.25" count as one rule.
- ``monthly``: net effect per month and cumulative, for a chart.
- ``selection_edge``: mean R of the trades the gates let through (forward closes) against the mean R
  of those they blocked: whether the gates, taken together, pick better trades.
- ``under_policy``: the blocked plans re-followed with the exit engine instead of the plain exits.

It only reads. Rows are the same shape ``filter_report`` returns plus extra keys, so the dashboard's
existing table keeps working when it calls ``gate_rows`` instead.
"""
from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd

from tradex.profit.exits import BarsFor, ExitPolicy, evaluate_on_ledger

Z90 = 1.645


def _mask(reason: str) -> str:
    s = re.sub(r"\d+(\.\d+)?", "#", reason or "")
    s = re.sub(r"\b20\d\d-\d\d-\d\d\b", "date", s)
    return re.sub(r"\s+", " ", s).strip()[:120]


def blocked_frame(ledger) -> pd.DataFrame:
    """One row per blocked plan: the gate, the reason, the plan's own numbers and what following it showed."""
    plans = {p.decision_id: p for p in ledger.records("plan", book="ensemble")}
    vetoes: dict[str, dict] = {}
    for v in ledger.rows(kind="veto"):
        vetoes.setdefault(v["decision_id"], v)                 # the first gate that stopped it
    cfs: dict[str, dict] = {}
    for c in ledger.rows(kind="counterfactual"):
        cfs[c["decision_id"]] = c                              # last row wins
    rows = []
    for did, v in vetoes.items():
        p = plans.get(did)
        if p is None:
            continue
        c = cfs.get(did)
        rows.append({"decision_id": did, "time": p.time, "symbol": p.symbol, "asset_class": p.asset_class,
                     "direction": p.direction, "blocked_by": v["source"], "reason": v["reason"],
                     "reason_key": _mask(v["reason"]), "p_target": p.p_target, "ev_r": p.ev_r, "score": p.score,
                     "families": ",".join(p.families), "cf_r": None if c is None else c["r_multiple"],
                     "cf_exit": None if c is None else c["exit_reason"]})
    cols = ["decision_id", "time", "symbol", "asset_class", "direction", "blocked_by", "reason", "reason_key",
            "p_target", "ev_r", "score", "families", "cf_r", "cf_exit"]
    return pd.DataFrame(rows, columns=cols)


def _verdict(n: int, mean: float, se: float | None, min_plans: int) -> str:
    if n < min_plans or se is None:
        return f"too few ({n} of {min_plans})"
    lo, hi = mean - Z90 * se, mean + Z90 * se
    if hi < 0:
        return "saves R"
    if lo > 0:
        return "costs R"
    return "inconclusive"


def _stats(r: pd.Series, min_plans: int) -> dict[str, Any]:
    n = int(r.notna().sum())
    if n == 0:
        return {"followed": 0, "mean_r": None, "total_r": 0.0, "win_rate": None, "ci90": None, "net_effect_r": 0.0,
                "verdict": _verdict(0, 0.0, None, min_plans)}
    x = r.dropna().astype(float)
    se = float(x.std(ddof=1) / math.sqrt(n)) if n > 1 else None
    mean = float(x.mean())
    return {"followed": n, "mean_r": round(mean, 3), "total_r": round(float(x.sum()), 2),
            "win_rate": round(float((x > 0).mean()), 3),
            "ci90": None if se is None else [round(mean - Z90 * se, 3), round(mean + Z90 * se, 3)],
            "net_effect_r": round(-float(x.sum()), 2), "verdict": _verdict(n, mean, se, min_plans)}


def gate_value(ledger, min_plans: int = 20) -> pd.DataFrame:
    df = blocked_frame(ledger)
    return _gate_table(df, min_plans)


def _gate_table(df: pd.DataFrame, min_plans: int) -> pd.DataFrame:
    cols = ["blocked_by", "plans", "pending", "followed", "mean_r", "total_r", "win_rate", "ci90", "net_effect_r",
            "verdict"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    out = []
    for gate, g in df.groupby("blocked_by"):
        s = _stats(g["cf_r"], min_plans)
        out.append({"blocked_by": gate, "plans": len(g), "pending": int(g["cf_r"].isna().sum()), **s})
    return pd.DataFrame(out, columns=cols).sort_values("net_effect_r", ascending=False).reset_index(drop=True)


def reason_breakdown(ledger, min_plans: int = 20, top: int = 15) -> pd.DataFrame:
    df = blocked_frame(ledger)
    cols = ["blocked_by", "reason", "plans", "followed", "mean_r", "total_r", "net_effect_r", "verdict"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    out = []
    for (gate, key), g in df.groupby(["blocked_by", "reason_key"]):
        s = _stats(g["cf_r"], min_plans)
        out.append({"blocked_by": gate, "reason": key, "plans": len(g), **{k: s[k] for k in
                    ("followed", "mean_r", "total_r", "net_effect_r", "verdict")}})
    res = pd.DataFrame(out, columns=cols)
    return res.reindex(res["plans"].sort_values(ascending=False).index).head(top).reset_index(drop=True)


def monthly(ledger) -> list[dict[str, Any]]:
    df = blocked_frame(ledger)
    df = df[df["cf_r"].notna()]
    if df.empty:
        return []
    df = df.assign(month=pd.to_datetime(df["time"], utc=True).dt.strftime("%Y-%m"))
    g = df.groupby("month")["cf_r"].agg(["size", "sum"]).sort_index()
    cum = (-g["sum"]).cumsum()
    return [{"month": m, "blocked": int(r["size"]), "would_have_made_r": round(float(r["sum"]), 2),
             "net_effect_r": round(-float(r["sum"]), 2), "cumulative_net_effect_r": round(float(c), 2)}
            for (m, r), c in zip(g.iterrows(), cum)]


def selection_edge(ledger, min_each: int = 10) -> dict[str, Any]:
    """Mean R of trades the gates let through against the blocked plans' counterfactual mean R (Welch t)."""
    taken: dict[str, float] = {}
    for c in ledger.rows(kind="close", book="ensemble"):
        taken[c["decision_id"]] = taken.get(c["decision_id"], 0.0) + float(c["r_multiple"])
    blocked = blocked_frame(ledger)["cf_r"].dropna().astype(float).to_numpy()
    a = np.array(list(taken.values()), float)
    out: dict[str, Any] = {"taken_n": len(a), "blocked_n": len(blocked),
                           "taken_mean_r": None if not len(a) else round(float(a.mean()), 3),
                           "blocked_mean_r": None if not len(blocked) else round(float(blocked.mean()), 3),
                           "edge_r": None, "t": None, "verdict": f"too few (need {min_each} each)"}
    if len(a) >= min_each and len(blocked) >= min_each:
        se = math.sqrt(a.var(ddof=1) / len(a) + blocked.var(ddof=1) / len(blocked))
        out["edge_r"] = round(float(a.mean() - blocked.mean()), 3)
        out["t"] = None if se == 0 else round(float((a.mean() - blocked.mean()) / se), 2)
        t = out["t"]
        out["verdict"] = ("gates pick better trades" if t is not None and t >= 1.645 else
                          "gates pick worse trades" if t is not None and t <= -1.645 else "no clear difference")
    return out


def under_policy(ledger, bars_for: BarsFor, policies: dict[str, ExitPolicy], **kw) -> dict[str, Any]:
    """The blocked plans re-followed under each exit policy (see ``compare_policies``)."""
    return evaluate_on_ledger(ledger, bars_for, policies, which="blocked", **kw)


def gate_rows(cfs: list[dict[str, Any]], min_plans: int = 20) -> list[dict[str, Any]]:
    """Drop-in for the dashboard's ``_filters(cfs)``: ``cfs`` are counterfactual payload dicts; the rows keep the keys
    ``blocked_by``, ``plans``, ``mean_r``, ``total_r``, ``win_rate`` and add the verdict columns."""
    if not cfs:
        return []
    df = pd.DataFrame(cfs)
    out = []
    for gate, g in df.groupby("blocked_by"):
        s = _stats(g["r_multiple"], min_plans)
        out.append({"blocked_by": gate, "plans": len(g), "mean_r": s["mean_r"], "total_r": s["total_r"],
                    "win_rate": s["win_rate"], "net_effect_r": s["net_effect_r"], "ci90": s["ci90"],
                    "verdict": s["verdict"]})
    return out


def dashboard_payload(ledger, min_plans: int = 20) -> dict[str, Any]:
    gates = _gate_table(blocked_frame(ledger), min_plans)
    return {"gates": gates.to_dict("records"), "reasons": reason_breakdown(ledger, min_plans).to_dict("records"),
            "monthly": monthly(ledger), "selection": selection_edge(ledger)}
