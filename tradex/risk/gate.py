"""The risk gate: the last of the four gates, and the only place a trade gets its size.

It receives a finalised plan (direction, entry, stop, targets, probability, expected
value) and the current book, and returns a Verdict with a quantity, possibly zero. It
never moves the entry, stop or targets (Ray, 4 Oct 2026). Every check it runs is written
into the verdict with its value and limit, which is what the dashboard's decision drawer
and size slider show.

Size: quarter Kelly on the plan's probability and reward to risk, capped per trade,
multiplied by the governor tier and any context cut (an event window halves size).
Then the size shrinks until every book limit holds:

- heat: summed loss-to-stop across the book
- currency: loss-to-stop attributed to each non-USD currency (net open position view)
- expected shortfall: the plan is charged its marginal 97.5% ES, so a trade that offsets
  the book is cheap and one that repeats an existing bet is charged as the bigger bet
- stress: worst named scenario loss
- leverage per asset class, from the governor tier
- position count, and a minimum economic size (fees under 10% of expected profit)
- free margin at the venue the trade goes to, when the caller supplies it

Protected path: agents cannot change tradex/risk/ or config/risk/.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from tradex.core.records import TradePlan, Verdict
from tradex.costs.models import split_pair, usd_per_unit
from tradex.risk.exposure import (ESModel, Leg, Scenario, book_exposures, factor_exposures,
                                  scenarios_from_config, stop_risk_by_currency, worst_stress)
from tradex.risk.sizing import kelly_fraction

DEFAULT_POLICY = Path(__file__).resolve().parents[2] / "config" / "risk" / "policy.yaml"


def load_policy(path: str | Path | None = None) -> dict[str, Any]:
    return yaml.safe_load(Path(path or DEFAULT_POLICY).read_text())


@dataclass
class BookState:
    equity: float
    legs: list[Leg]                       # every open position, Ray's own included (account="ray")
    tier: int = 2
    fx: dict | None = None
    margin_max_qty: float | None = None   # what the venue's free margin can carry; None = not checked
    margin: dict | None = None            # the numbers behind it, for the decision drawer


@dataclass
class RiskGate:
    policy: dict[str, Any]
    es_model: ESModel | None = None
    scenarios: list[Scenario] = field(default_factory=list)

    @classmethod
    def from_policy(cls, path: str | Path | None = None, es_model: ESModel | None = None) -> "RiskGate":
        pol = load_policy(path)
        return cls(pol, es_model, scenarios_from_config(pol.get("scenarios", [])))

    def review(self, plan: TradePlan, book: BookState, base_to_usd: float,
               fee_fn: Callable[[float], float] | None = None, size_factor: float = 1.0,
               promoted: bool = False, lot: float = 1.0) -> Verdict:
        """Size ``plan`` against ``book``. ``fee_fn(qty)`` returns round-trip fees in USD."""
        sz, bk = self.policy["sizing"], self.policy["book"]
        tier = bk["tiers"][book.tier]
        checks: dict[str, Any] = {}
        reasons: list[str] = []
        eq = book.equity

        def reject(why: str) -> Verdict:
            return Verdict(plan.decision_id, plan.time, "rejected", 0.0, 0.0, 0.0, reasons + [why], checks)

        if eq <= 0:
            return reject("no equity")
        unit_risk = plan.risk_per_unit * base_to_usd
        if unit_risk <= 0:
            return reject("plan has no distance to its stop")

        # 1. size from confidence
        f = kelly_fraction(plan.p_target, plan.reward_risk)
        cap = sz["per_trade_cap_promoted_pct"] if promoted else sz["per_trade_cap_pct"]
        risk_pct = min(cap, max(0.0, f) * sz["kelly_fraction"] * 100.0)
        risk_pct *= tier["size_multiplier"] * size_factor
        checks["kelly"] = {"p": plan.p_target, "p_source": plan.p_source, "reward_risk": plan.reward_risk,
                           "kelly_f": round(f, 4), "risk_pct": round(risk_pct, 4), "cap_pct": cap,
                           "tier": book.tier, "size_factor": size_factor}
        if f <= 0:
            return reject(f"no edge: Kelly fraction {f:.3f} at p={plan.p_target:.2f}, R:R={plan.reward_risk:.2f}")
        want = eq * risk_pct / 100.0 / unit_risk

        # 2. shrink until every book limit holds
        mine = [l for l in book.legs if l.account == "agent"]
        cand = lambda q: Leg(plan.symbol, plan.asset_class, plan.direction, q, plan.entry_price, plan.stop)  # noqa: E731

        heat_now = sum(max(0.0, l.direction * (l.price - l.stop)) * l.qty * _bu(l, book) for l in mine if l.stop is not None)
        heat_room = eq * bk["heat_cap_pct"] / 100.0 - heat_now
        q_heat = max(0.0, heat_room / unit_risk)
        checks["heat"] = {"now_usd": round(heat_now, 2), "cap_usd": round(eq * bk["heat_cap_pct"] / 100, 2),
                          "max_qty": q_heat}

        q_ccy = math.inf
        if plan.asset_class == "forex":
            now = stop_risk_by_currency(mine, book.fx)
            one = stop_risk_by_currency([cand(1.0)], book.fx)
            limit = eq * bk["currency_stop_risk_cap_pct"] / 100.0
            per = {}
            for c, r1 in one.items():
                room = limit - now.get(c, 0.0)
                per[c] = {"now_usd": round(now.get(c, 0.0), 2), "cap_usd": round(limit, 2)}
                q_ccy = min(q_ccy, max(0.0, room / r1) if r1 > 0 else math.inf)
            checks["currency"] = per

        lev_cap = tier["max_leverage"][plan.asset_class]
        gross_ac = sum(l.qty * l.price * _bu(l, book) for l in mine if l.asset_class == plan.asset_class)
        unit_notional = plan.entry_price * base_to_usd
        q_lev = max(0.0, (eq * lev_cap - gross_ac) / unit_notional)
        checks["leverage"] = {"asset_class": plan.asset_class, "gross_usd": round(gross_ac, 2),
                              "cap_x": lev_cap, "max_qty": q_lev}

        q_margin = math.inf if book.margin_max_qty is None else max(0.0, book.margin_max_qty)
        if book.margin_max_qty is not None:
            checks["margin"] = dict(book.margin or {}, max_qty=q_margin)

        all_exp = book_exposures(book.legs, book.fx)          # Ray's holdings count in exposure
        q = min(want, q_heat, q_ccy, q_lev, q_margin)
        binding = min((("confidence", want), ("heat", q_heat), ("currency", q_ccy), ("leverage", q_lev),
                       ("margin", q_margin)), key=lambda x: x[1])[0]

        if self.es_model is not None and q > 0:
            budget = eq * bk["es_budget_pct"] / 100.0
            add = lambda qq: factor_exposures(cand(qq), book.fx)  # noqa: E731
            es_now = self.es_model.es(all_exp)
            missing = self.es_model.missing(add(1.0))
            fits = lambda qq: self.es_model.es(_merge(all_exp, add(qq))) <= max(budget, es_now)  # noqa: E731
            q_es = _largest(fits, q)
            checks["expected_shortfall"] = {
                "now_usd": round(es_now, 2), "budget_usd": round(budget, 2),
                "marginal_usd": round(self.es_model.marginal(all_exp, add(max(q_es, 0.0))), 2),
                "max_qty": q_es, "missing_history": missing}
            if q_es < q:
                q, binding = q_es, "expected_shortfall"

        if self.scenarios and q > 0:
            limit = eq * bk["stress_loss_cap_pct"] / 100.0
            name_now, pnl_now = worst_stress(all_exp, self.scenarios)
            fits = lambda qq: -worst_stress(_merge(all_exp, factor_exposures(cand(qq), book.fx)), self.scenarios)[1] <= max(limit, -pnl_now)  # noqa: E731
            q_st = _largest(fits, q)
            name_new, pnl_new = worst_stress(_merge(all_exp, factor_exposures(cand(q_st), book.fx)), self.scenarios)
            checks["stress"] = {"worst_now": name_now, "loss_now_usd": round(max(0.0, -pnl_now), 2),
                                "worst_with_trade": name_new, "loss_with_trade_usd": round(max(0.0, -pnl_new), 2),
                                "cap_usd": round(limit, 2), "max_qty": q_st}
            if q_st < q:
                q, binding = q_st, "stress"

        if len(mine) >= bk["max_positions"]:
            checks["positions"] = {"open": len(mine), "cap": bk["max_positions"]}
            return reject(f"position cap: {len(mine)} open")

        qty = math.floor(q / lot) * lot
        if qty <= 0:
            return reject(f"no room: {binding} limit leaves less than one unit")
        if binding != "confidence":
            reasons.append(f"size cut by {binding} limit")

        # 3. minimum economic size
        risk_usd = qty * unit_risk
        exp_profit = plan.ev_r * risk_usd
        fees = fee_fn(qty) if fee_fn else 0.0
        checks["economics"] = {"fees_usd": round(fees, 2), "expected_profit_usd": round(exp_profit, 2),
                               "max_fee_share": sz["min_fee_share_of_edge"]}
        if exp_profit <= 0:
            return reject("expected value is not positive after costs")
        if fees > sz["min_fee_share_of_edge"] * exp_profit:
            return reject(f"fees {fees:.2f} USD exceed {sz['min_fee_share_of_edge']:.0%} of expected profit {exp_profit:.2f} USD")

        return Verdict(plan.decision_id, plan.time, "accepted", float(qty), round(risk_usd, 2),
                       round(100.0 * risk_usd / eq, 4), reasons, checks)


def _bu(leg: Leg, book: BookState) -> float:
    if leg.asset_class != "forex":
        return 1.0
    return usd_per_unit(split_pair(leg.symbol)[1], None, book.fx)


def _merge(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0.0) + v
    return out


def _largest(fits: Callable[[float], bool], hi: float, iters: int = 30) -> float:
    """Largest quantity in [0, hi] for which ``fits`` holds (limits grow with size)."""
    if hi <= 0 or fits(hi):
        return max(hi, 0.0)
    lo = 0.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if fits(mid) else (lo, mid)
    return lo
