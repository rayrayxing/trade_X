"""Stage 5: rank live-eligible strategies, group correlated ones, split the risk budget, and blend.

Inputs are strategy records carrying out-of-sample daily returns and trades
(from walk-forward validation, and later from paper and live ledgers). The output
is an allocation snapshot: risk % of equity per strategy, with the reasons.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from tradex.backtest import metrics
from tradex.risk.sizing import conservative_kelly, drawdown_scale

ELIGIBLE = {"research": {"validated", "paper", "live"}, "paper": {"validated", "paper", "live"}, "live": {"live"}}


@dataclass
class StrategyRecord:
    id: str
    asset_class: str
    status: str
    oos_returns: pd.Series                 # daily returns at the backtest's reference risk
    oos_trades: pd.DataFrame
    dsr: float = 0.0
    reference_risk_pct: float = 1.0        # risk % per trade the returns were generated at
    cap_risk_pct: float = 3.0
    symbols: list[str] = field(default_factory=list)


@dataclass
class AllocationConfig:
    total_risk_pct: float = 12.0           # open-risk budget across the book (aggressive default)
    per_strategy_cap_pct: float = 3.0      # design doc ceiling before paper
    per_strategy_cap_promoted_pct: float = 5.0
    per_cluster_cap_pct: float = 6.0       # correlated strategies count as one bet
    corr_threshold: float = 0.6
    min_overlap_days: int = 60
    recent_halflife_days: int = 60
    weight_recent_sharpe: float = 0.4
    weight_dsr: float = 0.3
    weight_regime_fit: float = 0.3
    regime_shrink_days: int = 120          # shrink regime-specific stats toward overall below this many days
    kelly_fraction: float = 0.25
    mode: str = "paper"                    # research | paper | live


@dataclass
class Allocation:
    strategy_id: str
    risk_pct: float
    score: float
    cluster: int
    components: dict
    notes: list[str]


@dataclass
class AllocationSnapshot:
    asof: str
    regime: dict
    total_risk_pct: float
    allocations: list[Allocation]
    excluded: dict[str, str]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


def recent_sharpe(r: pd.Series, halflife: int) -> float:
    r = r.dropna()
    if len(r) < 20:
        return 0.0
    w = 0.5 ** (np.arange(len(r))[::-1] / halflife)
    m = np.average(r, weights=w)
    sd = np.sqrt(np.average((r - m) ** 2, weights=w))
    return float(m / sd * np.sqrt(metrics.PERIODS_PER_YEAR)) if sd > 0 else 0.0


def regime_fit(r: pd.Series, labels: pd.Series, current: str, shrink_days: int) -> float:
    """Annualised Sharpe in the current regime, shrunk toward the overall Sharpe when evidence is thin."""
    r = r.dropna()
    if r.empty:
        return 0.0
    overall = metrics.sharpe(r)
    lab = labels.reindex(r.index.normalize(), method="ffill")
    lab.index = r.index
    sub = r[lab == current]
    if len(sub) < 5:
        return overall
    w = len(sub) / (len(sub) + shrink_days)
    return float(w * metrics.sharpe(sub) + (1 - w) * overall)


def correlation_clusters(records: list[StrategyRecord], threshold: float, min_overlap: int) -> dict[str, int]:
    """Union-find on pairwise correlation of daily returns. Strategies trading the same symbols in the
    same asset class with too little overlap to measure are grouped too (conservative)."""
    ids = [r.id for r in records]
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a in range(len(records)):
        for b in range(a + 1, len(records)):
            ra, rb = records[a], records[b]
            joined = pd.concat([ra.oos_returns, rb.oos_returns], axis=1, join="inner").dropna()
            joined = joined[(joined != 0).any(axis=1)]
            if len(joined) >= min_overlap:
                rho = joined.corr().iloc[0, 1]
                linked = bool(rho >= threshold) if not np.isnan(rho) else False
            else:
                linked = ra.asset_class == rb.asset_class and bool(set(ra.symbols) & set(rb.symbols))
            if linked:
                parent[find(ra.id)] = find(rb.id)
    roots = {}
    return {i: roots.setdefault(find(i), len(roots)) for i in ids}


def _water_fill(scores: dict[str, float], caps: dict[str, float], budget: float) -> dict[str, float]:
    """Split ``budget`` in proportion to scores, respecting caps and redistributing the excess."""
    alloc = {k: 0.0 for k in scores}
    active = {k for k, s in scores.items() if s > 0 and caps[k] > 0}
    remaining = budget
    while active and remaining > 1e-9:
        tot = sum(scores[k] for k in active)
        give = {k: remaining * scores[k] / tot for k in active}
        remaining = 0.0
        for k in list(active):
            room = caps[k] - alloc[k]
            if give[k] >= room:
                alloc[k] = caps[k]
                remaining += give[k] - room
                active.discard(k)
            else:
                alloc[k] += give[k]
    return alloc


def allocate(
    records: list[StrategyRecord],
    regime_label: str,
    regime_labels: pd.Series | None = None,
    cfg: AllocationConfig | None = None,
    equity: float | None = None,
    peak_equity: float | None = None,
    asof: str | None = None,
) -> AllocationSnapshot:
    cfg = cfg or AllocationConfig()
    eligible_status = ELIGIBLE[cfg.mode]
    excluded: dict[str, str] = {}
    live = []
    for r in records:
        if r.status not in eligible_status:
            excluded[r.id] = f"status {r.status} not eligible in {cfg.mode} mode"
        else:
            live.append(r)

    budget = cfg.total_risk_pct
    dd_mult = drawdown_scale(equity, peak_equity) if equity and peak_equity else 1.0
    budget *= dd_mult

    comps, scores, caps = {}, {}, {}
    for r in live:
        rs = recent_sharpe(r.oos_returns, cfg.recent_halflife_days)
        rf = regime_fit(r.oos_returns, regime_labels, regime_label, cfg.regime_shrink_days) if regime_labels is not None else rs
        k = conservative_kelly(r.oos_trades, cfg.kelly_fraction)
        # DSR (a probability) is mapped to -2..+2 so it sits on the same scale as annual Sharpe.
        score = cfg.weight_recent_sharpe * rs + cfg.weight_dsr * (4 * r.dsr - 2) + cfg.weight_regime_fit * rf
        comps[r.id] = {"recent_sharpe": rs, "dsr": r.dsr, "regime_fit": rf, "kelly": k, "score": score}
        if r.status in ("paper", "live"):
            cap = cfg.per_strategy_cap_promoted_pct
        else:
            cap = min(cfg.per_strategy_cap_pct, r.cap_risk_pct)
        cap = min(cap, k["risk_pct"])  # never exceed fractional Kelly on the lower-bound win rate
        caps[r.id] = cap
        scores[r.id] = score
        if score <= 0:
            excluded[r.id] = f"score {score:.2f} not positive"
        elif cap <= 0:
            excluded[r.id] = f"Kelly on lower-bound win rate is {k['kelly']:.3f}; no edge to size"

    clusters = correlation_clusters(live, cfg.corr_threshold, cfg.min_overlap_days)
    # Budget per cluster by its best member's score, then within the cluster by score.
    cluster_score: dict[int, float] = {}
    for sid, c in clusters.items():
        cluster_score[c] = max(cluster_score.get(c, 0.0), max(scores.get(sid, 0.0), 0.0))
    cluster_caps = {c: min(cfg.per_cluster_cap_pct, sum(caps[s] for s, cc in clusters.items() if cc == c and scores[s] > 0))
                    for c in cluster_score}
    cluster_budget = _water_fill(cluster_score, cluster_caps, budget)

    allocs = []
    for c, cb in cluster_budget.items():
        members = {s: max(scores[s], 0.0) for s, cc in clusters.items() if cc == c}
        split = _water_fill(members, {s: caps[s] for s in members}, cb)
        for s, risk in split.items():
            notes = []
            if risk >= caps[s] - 1e-9 and risk > 0:
                notes.append(f"at cap {caps[s]:.2f}% (strategy cap or quarter-Kelly)")
            if len(members) > 1:
                notes.append(f"shares cluster {c} with {len(members) - 1} correlated strateg{'y' if len(members) == 2 else 'ies'}")
            if dd_mult < 1:
                notes.append(f"budget scaled x{dd_mult:.2f} for drawdown")
            allocs.append(Allocation(s, round(risk, 4), round(scores[s], 4), c, comps[s], notes))
    allocs.sort(key=lambda a: -a.risk_pct)
    return AllocationSnapshot(
        asof=asof or str(pd.Timestamp.now(tz="UTC")), regime={"label": regime_label, "drawdown_multiplier": dd_mult},
        total_risk_pct=round(sum(a.risk_pct for a in allocs), 4), allocations=allocs, excluded=excluded,
    )


def blend_returns(records: list[StrategyRecord], snapshot: AllocationSnapshot) -> pd.Series:
    """Daily returns of the blended book: each strategy's returns rescaled from its reference risk
    to its allocated risk, then summed."""
    by_id = {r.id: r for r in records}
    parts = []
    for a in snapshot.allocations:
        if a.risk_pct <= 0:
            continue
        r = by_id[a.strategy_id]
        parts.append(r.oos_returns * (a.risk_pct / r.reference_risk_pct))
    if not parts:
        return pd.Series(dtype=float)
    return pd.concat(parts, axis=1).fillna(0.0).sum(axis=1)


def blend_report(records: list[StrategyRecord], snapshot: AllocationSnapshot, initial: float = 10_000.0) -> dict:
    """Compare the blend with each strategy alone (each at its own allocated risk)."""
    out = {}
    blend = blend_returns(records, snapshot)
    if len(blend):
        eq = (1 + blend).cumprod() * initial
        out["blend"] = {"sharpe": metrics.sharpe(blend), "max_drawdown": metrics.max_drawdown(eq),
                        "total_return": float(eq.iloc[-1] / initial - 1)}
    by_id = {r.id: r for r in records}
    for a in snapshot.allocations:
        r = by_id[a.strategy_id].oos_returns * (a.risk_pct / by_id[a.strategy_id].reference_risk_pct)
        if len(r):
            eq = (1 + r).cumprod() * initial
            out[a.strategy_id] = {"sharpe": metrics.sharpe(r), "max_drawdown": metrics.max_drawdown(eq),
                                  "total_return": float(eq.iloc[-1] / initial - 1)}
    return out
