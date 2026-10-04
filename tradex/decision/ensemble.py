"""Gate 2: turn strategy votes into one finalised trade plan (Ray, 4 Oct 2026).

Strategies vote; strategies in the same family (declared, or merged because their
signals correlate above ``family_signal_corr``) count once. A plan is formed only when
at least ``min_families`` independent families agree, no family opposes strongly, the
weighted score clears ``min_score``, and target 1 pays at least ``min_reward_risk``
after round-trip costs. The plan is then complete: direction, entry, stop, targets, time
stop, invalidation, probability and expected value. Size is left to the risk gate.

Prices from the agreeing votes:
- stop: the farthest of their stops, so the thesis is invalid only when every agreeing
  strategy is invalidated (size comes from this distance, so the risk in dollars is the same)
- target 1: the nearest of their targets; target 2: the farthest
- time stop: the shortest of their limits in time, counted in bars of the finest timeframe
  among them (votes from different timeframes can agree: a higher-timeframe vote stays
  valid until that timeframe's next close, so H1 and H4 strategies can form one plan)
- entry reference and plan time: the freshest vote's (the close being decided); a plan whose
  price is already through a held vote's stop, or past every target, is not traded

Probability is the mean of the agreeing strategies' measured hit rates, labelled
``base_rate`` until the meta-label size model is calibrated in paper.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tradex.core.records import TradePlan, Vote
from tradex.costs.models import CostModel
from tradex.timeframes import duration

DEFAULT_HIT_RATE = 0.40


@dataclass
class PlanRules:
    min_families: int = 2
    min_score: float = 0.25
    max_opposing_strength: float = 0.6
    min_reward_risk: float = 1.5

    @classmethod
    def from_policy(cls, policy: dict) -> "PlanRules":
        p = policy.get("plan", {})
        return cls(int(p.get("min_families", 2)), float(p.get("min_score", 0.25)),
                   float(p.get("max_opposing_strength", 0.6)), float(p.get("min_reward_risk", 1.5)))


def family_votes(votes: list[Vote], weights: dict[str, float] | None = None) -> dict[str, tuple[int, float, float]]:
    """Collapse votes to one per family: (direction, strength, weight).

    A family's direction is the sign of its members' summed strength; its strength is
    the strongest member on that side, so adding a correlated twin adds nothing."""
    out: dict[str, tuple[int, float, float]] = {}
    by_fam: dict[str, list[Vote]] = {}
    for v in votes:
        if v.direction != 0:
            by_fam.setdefault(v.family, []).append(v)
    for fam, vs in by_fam.items():
        s = sum(v.direction * v.strength for v in vs)
        if s == 0:
            continue
        d = 1 if s > 0 else -1
        strength = max(v.strength for v in vs if v.direction == d)
        out[fam] = (d, strength, (weights or {}).get(fam, 1.0))
    return out


def finalise(votes: list[Vote], decision_id: str, rules: PlanRules, cost: CostModel,
             weights: dict[str, float] | None = None, book: str = "ensemble"
             ) -> tuple[TradePlan | None, str]:
    """Return (plan, "") or (None, why no plan). A plan returned may still fail the cost
    rule; it is returned with its reason so the counterfactual ledger can follow it."""
    if not votes:
        return None, "no votes"
    fams = family_votes(votes, weights)
    if not fams:
        return None, "votes cancel out"
    wsum = sum(w for _, _, w in fams.values())
    score = sum(d * s * w for d, s, w in fams.values()) / wsum
    direction = 1 if score > 0 else -1
    agree = [f for f, (d, _, _) in fams.items() if d == direction]
    oppose = max((s for d, s, _ in fams.values() if d != direction), default=0.0)
    agreeing = [v for v in votes if v.family in agree and v.direction == direction]

    v0 = max(votes, key=lambda v: (pd.Timestamp(v.time), -_dur(v.tf)))      # the decision time; first on ties
    entry = v0.entry_ref
    stops = [v.stop for v in agreeing]
    stop = min(stops) if direction > 0 else max(stops)
    tgts = sorted({t for v in agreeing for t in v.targets}, key=lambda t: abs(t - entry))
    ahead = [t for t in tgts if direction * (t - entry) > 0]              # a held vote's target may be behind price
    t1, t2 = (ahead or tgts)[0], (ahead or tgts)[-1]
    _, unit = cost.fill(v0.symbol, direction, entry, pd.Timestamp(v0.time))
    cost_u = 2 * (unit["spread"] + unit["slippage"])
    risk = abs(entry - stop)
    rr = (abs(t1 - entry) - cost_u) / (risk + cost_u) if risk > 0 else 0.0
    p = float(np.mean([v.strength for v in agreeing]))
    max_bars, tf = _time_stop(agreeing, v0)
    plan = TradePlan(
        decision_id=decision_id, time=v0.time, symbol=v0.symbol, asset_class=v0.asset_class,
        direction=direction, entry_type="market", entry_price=entry, stop=stop,
        targets=[t1] if t1 == t2 else [t1, t2], max_bars=max_bars,
        invalidation=f"price trades through {stop:.5g}, or the time stop of {max_bars} {tf or 'signal'} bars passes",
        families=sorted(agree), strategies=sorted(v.strategy_id for v in agreeing),
        score=round(abs(score), 4), p_target=round(p, 4), p_source="base_rate",
        reward_risk=round(rr, 4), ev_r=round(p * rr - (1 - p), 4),
        cost_r=round(cost_u / risk, 4) if risk else 0.0, book=book, tf=tf,
    )
    if direction * (entry - stop) <= 0:
        return plan, f"price {entry:.5g} is already through the stop {stop:.5g} of a held vote"
    if not ahead:
        return plan, f"price {entry:.5g} is already past every target"
    if len(agree) < rules.min_families:
        return plan, f"only {len(agree)} famil{'y' if len(agree) == 1 else 'ies'} agree ({', '.join(sorted(agree))}); need {rules.min_families}"
    if oppose > rules.max_opposing_strength:
        return plan, f"a family opposes with strength {oppose:.2f}"
    if abs(score) < rules.min_score:
        return plan, f"score {abs(score):.2f} below {rules.min_score}"
    if rr < rules.min_reward_risk:
        return plan, f"reward to risk {rr:.2f} after costs below {rules.min_reward_risk}"
    return plan, ""


def _dur(tf: str) -> pd.Timedelta:
    return duration(tf) if tf else pd.Timedelta(0)


def _time_stop(agreeing: list[Vote], v0: Vote) -> tuple[int, str]:
    """(bars, timeframe): the shortest time limit among the votes, in bars of the finest timeframe."""
    if not all(v.tf for v in agreeing):
        return min(v.max_bars for v in agreeing), v0.tf
    fine = min((v.tf for v in agreeing), key=duration)
    horizon = min(v.max_bars * duration(v.tf) for v in agreeing)
    return max(1, int(horizon // duration(fine))), fine


def merge_correlated_families(signals: dict[str, pd.Series], declared: dict[str, str], threshold: float = 0.7
                              ) -> dict[str, str]:
    """Strategies whose direction signals (+1/0/-1 per bar) correlate above ``threshold``
    share one family label, whatever they declared. Returns strategy -> family."""
    ids = list(signals)
    fam = dict(declared)
    if len(ids) < 2:
        return fam
    df = pd.DataFrame(signals).fillna(0.0)
    df = df.loc[:, df.std() > 0]
    corr = df.corr() if df.shape[1] > 1 else pd.DataFrame()
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if a in corr.index and b in corr.columns and corr.loc[a, b] > threshold and fam[a] != fam[b]:
                old, new = fam[b], fam[a]
                fam = {k: (new if v == old else v) for k, v in fam.items()}
    return fam
