"""The single object the core calls (``TradingCore(..., profit=hooks)``).

Everything here is optional and off by default: a core built with ``profit=None`` behaves exactly as before.
Each method takes plain values and returns plain values, so the core keeps ownership of every order, and
the risk gate keeps ownership of every size:

- ``annotate_plan(plan)``: the plan with a calibrated probability and EV when the model is calibrated and beats
  the raw base rate out of sample, and the exit ladder written into its invalidation text; prices are never touched.
- ``size_factor(...)``: a number in (0, 1] the core multiplies into the quantity the gate approved, after any agent
  shrink. Vol targeting can only shrink, and never sizes above the gate.
- ``attach_target(plan)``: the take-profit to attach to the entry order. With partial targets configured it is
  the LAST target, so a venue-side order does not close the whole position at target 1 before the engine has
  taken its partial.
- ``manage(...)``: the position reviewer's actions merged with the exit engine's. One close, the tightest stop,
  and a reduce expressed as a share of what is open now.
- ``daily(...)``: refit the probability model and recalibrate the cost models from the ledger, as of ``t``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any, Callable

import pandas as pd

from tradex.core.records import TradePlan
from tradex.positions.review import Action, ActionKind
from tradex.profit.calibration import CalibrationConfig, TradeProbabilityModel, fit_from_ledger
from tradex.profit.costcal import CalibratedCosts, CostCalConfig, calibrate_costs, calibrated_costs
from tradex.profit.exits import ExitPolicy, ExitState, evaluate_exit, exit_ladder
from tradex.profit.suggest import VolTargetPolicy, vol_target_suggestion


@dataclass
class ProfitHooks:
    exit_policy: ExitPolicy | None = None
    calibration: CalibrationConfig | None = None
    vol_policy: VolTargetPolicy | None = None
    cost_cfg: CostCalConfig | None = None
    prob_model: TradeProbabilityModel | None = field(default=None, init=False)
    last_cost_report: dict[str, Any] = field(default_factory=dict, init=False)
    _reduced: dict[tuple[str, str], int] = field(default_factory=dict, init=False)
    _base_costs: dict[str, Any] = field(default_factory=dict, init=False)

    # --- plans -------------------------------------------------------------------------------------

    def annotate_plan(self, plan: TradePlan) -> TradePlan:
        """Calibrated probability and EV when available, and the exit ladder written into the plan's invalidation text
        so the card, the dashboard drawer and ``why`` say how the trade will be managed. Prices are never touched."""
        if self.prob_model is not None:
            plan = self.prob_model.annotate(plan)
        if self.exit_policy is not None:
            plan = replace(plan, invalidation=f"{plan.invalidation}; managed by: {exit_ladder(plan, self.exit_policy).text()}")
        return plan

    def attach_target(self, plan: TradePlan) -> float:
        if self.exit_policy is not None and self.exit_policy.partials and len(plan.targets) > 1:
            return plan.targets[-1]
        return plan.targets[0]

    # --- size ----------------------------------------------------------------------------------------

    def size_factor(self, ledger, book: str, base_risk_pct: float) -> float:
        """Vol-target multiplier for the gate, never above 1.0 and 1.0 when there is too little history."""
        if self.vol_policy is None:
            return 1.0
        rows = ledger.db.execute("SELECT payload FROM events WHERE kind='snapshot' AND book=? ORDER BY seq DESC LIMIT ?",
                                 (book, self.vol_policy.lookback + 1)).fetchall()
        curve = [json.loads(r[0])["equity_usd"] for r in reversed(rows)]
        return vol_target_suggestion(curve, base_risk_pct, self.vol_policy).shrink_factor

    # --- open positions ----------------------------------------------------------------------------------

    def manage(self, book: str, decision_id: str, *, direction: int, entry_price: float, entry_time: pd.Timestamp,
               initial_stop: float, stop: float | None, targets: list[float], max_bars: int, cost_r: float,
               qty0: float, qty: float, bars: pd.DataFrame, base: list[Action], tf_duration: pd.Timedelta
               ) -> list[Action]:
        if self.exit_policy is None or not len(bars) or qty0 <= 0:
            return base
        done = self._reduced.get((book, decision_id), 0)
        entry_bar = pd.Timestamp(entry_time).floor(tf_duration)
        st = ExitState(direction, entry_price, initial_stop, stop if stop is not None else initial_stop, tuple(targets),
                       entry_bar, max_bars, cost_r, None, done, min(1.0, qty / qty0))
        eng = [a for a in evaluate_exit(self.exit_policy, st, bars) if a.kind != ActionKind.HOLD]
        acts = [a for a in base if a.kind != ActionKind.HOLD]
        closes = [a for a in acts + eng if a.kind == ActionKind.CLOSE]
        if closes:
            return closes[:1]
        out: list[Action] = []
        stops = [a for a in acts + eng if a.kind == ActionKind.MOVE_STOP and a.price is not None]
        if stops:
            best = max(stops, key=lambda a: direction * a.price)
            if stop is None or direction * best.price > direction * stop:
                out.append(best)
        reduces = [a for a in acts + eng if a.kind == ActionKind.REDUCE]
        if reduces:
            r = reduces[0]
            if r.price is not None and r.fraction is not None:      # the engine's: share of the ORIGINAL position
                share_now = max(1e-9, min(1.0, qty / qty0))
                r = Action(r.kind, r.reason, r.price, min(0.999, r.fraction / share_now))
            out.append(r)
        return out or [Action(ActionKind.HOLD, "within plan")]

    def note_reduce(self, book: str, decision_id: str) -> None:
        k = (book, decision_id)
        self._reduced[k] = self._reduced.get(k, 0) + 1

    # --- once a day -----------------------------------------------------------------------------------------

    def daily(self, ledger, costs: dict[str, Any], rate_fn: Callable[[str, pd.Timestamp], float] | None,
              t: pd.Timestamp) -> None:
        """Refit what has enough forward evidence. ``costs`` is the core's own dict: wrapped models are swapped in place."""
        until = t.isoformat()
        if self.calibration is not None:
            self.prob_model = fit_from_ledger(ledger, self.calibration, until=until)
        if self.cost_cfg is not None:
            for ac, m in costs.items():
                self._base_costs.setdefault(ac, m.base if isinstance(m, CalibratedCosts) else m)
            cal = calibrate_costs(ledger, self._base_costs, rate_fn, self.cost_cfg, until=until)
            costs.update(calibrated_costs(self._base_costs, cal))
            self.last_cost_report = {ac: c.to_dict() for ac, c in cal.items()}
