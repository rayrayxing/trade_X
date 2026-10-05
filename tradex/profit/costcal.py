"""Self-calibrating cost model: realised fills against the model (profit spec P7).

The planner charges every plan a round-trip cost from ``tradex.costs`` (spread, slippage, fees).
Those are estimates until a venue fills real orders. This module reads the fills in the ledger,
recomputes what the model predicted for each, and learns two multipliers per asset class:

- ``spread_mult``: observed spread-plus-slippage dollars / model dollars
- ``fee_mult``: observed fee dollars / model fee dollars

``CalibratedCosts`` wraps a base model with them and satisfies the same ``CostModel`` protocol, so
it drops into ``TradingCore(costs=...)`` and every later plan, size and EV is charged what trading
here really costs. Rules that keep it honest:

- Only fills of books given (default the ensemble book, never the virtual books, which are filled
  by the simulator from the same model and would calibrate it against itself).
- Fewer than ``min_fills`` usable fills in a class: multipliers stay 1.0 and the status says why.
- Shrinkage toward 1.0 by ``n / (n + prior_fills)`` so a few odd fills move little.
- By default the model may only get MORE pessimistic (``floor=1.0``): a venue that happens to fill
  better than the model for a few weeks does not make the plans cheaper. Lower the floor in config
  once the sample is large.
- Always calibrate against the BASE model (``CalibratedCosts`` is unwrapped), so applying the result
  and recalibrating later does not feed back on itself.
- Forex fills need a USD rate function; without one they are skipped and counted, not converted at a guess.
- Entry slippage against the plan's expected price is reported as a diagnostic (it carries overnight
  gaps, so it is noisy and does not drive the multipliers).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from tradex.costs.models import CostModel, RateMissing, split_pair

RateFn = Callable[[str, pd.Timestamp], float]            # (currency, time) -> USD per unit


@dataclass(frozen=True)
class CostCalConfig:
    min_fills: int = 30
    prior_fills: int = 20
    floor: float = 1.0
    cap: float = 5.0
    bootstrap: int = 400
    seed: int = 0


@dataclass
class CostCalibration:
    asset_class: str
    spread_mult: float = 1.0
    fee_mult: float = 1.0
    n_fills: int = 0
    calibrated: bool = False
    status: str = ""
    spread_ratio: float | None = None                     # raw observed / model before shrinkage and clamps
    fee_ratio: float | None = None
    spread_ci: tuple[float, float] | None = None
    fee_ci: tuple[float, float] | None = None
    skipped: dict[str, int] = field(default_factory=dict)
    entry_shortfall_bps: float | None = None              # mean adverse move from the plan's price to the entry fill
    entry_shortfall_se_bps: float | None = None
    observed_usd: dict[str, float] = field(default_factory=dict)
    model_usd: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class CalibratedCosts:
    """A cost model scaled by the learned multipliers. Everything not scaled is the base model's own."""

    def __init__(self, base: CostModel, spread_mult: float = 1.0, fee_mult: float = 1.0):
        while isinstance(base, CalibratedCosts):
            base = base.base
        self.base, self.spread_mult, self.fee_mult = base, float(spread_mult), float(fee_mult)
        self.asset_class = base.asset_class

    def fill(self, symbol, side, mid, ts):
        px, unit = self.base.fill(symbol, side, mid, ts)
        m = self.spread_mult
        return mid + (px - mid) * m, {k: v * m for k, v in unit.items()}

    def order_fees(self, symbol, side, qty, price, ts):
        return {k: v * self.fee_mult for k, v in self.base.order_fees(symbol, side, qty, price, ts).items()}

    def holding_cost(self, symbol, direction, qty, price, t0, t1, base_to_usd=1.0, borrowed_usd=0.0):
        return self.base.holding_cost(symbol, direction, qty, price, t0, t1, base_to_usd, borrowed_usd)

    def __getattr__(self, name: str):                     # annual_rate, rates, ... of the underlying model
        if name in ("base", "spread_mult", "fee_mult"):
            raise AttributeError(name)
        return getattr(self.base, name)


def _ratio(obs: np.ndarray, mod: np.ndarray) -> float | None:
    s = float(mod.sum())
    return float(obs.sum() / s) if s > 0 else None


def _boot_ci(obs: np.ndarray, mod: np.ndarray, n: int, seed: int) -> tuple[float, float] | None:
    if len(obs) < 5 or n <= 0 or mod.sum() <= 0:
        return None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(obs), size=(n, len(obs)))
    num, den = obs[idx].sum(axis=1), mod[idx].sum(axis=1)
    ok = den > 0
    if not ok.any():
        return None
    r = num[ok] / den[ok]
    return float(np.quantile(r, 0.05)), float(np.quantile(r, 0.95))


def _shrunk(ratio: float | None, n: int, cfg: CostCalConfig) -> float:
    if ratio is None:
        return 1.0
    m = 1.0 + (ratio - 1.0) * n / (n + cfg.prior_fills)
    return float(min(cfg.cap, max(cfg.floor, m)))


def calibrate_costs(ledger, costs: dict[str, CostModel], rate_fn: RateFn | None = None,
                    cfg: CostCalConfig | None = None, books: Iterable[str] = ("ensemble",),
                    since: str | None = None, until: str | None = None) -> dict[str, CostCalibration]:
    """Per asset class: how the ledger's real fills compare with what the (base) model predicted."""
    cfg = cfg or CostCalConfig()
    base = {ac: (m.base if isinstance(m, CalibratedCosts) else m) for ac, m in costs.items()}
    books = set(books)
    plans = {p.decision_id: p for p in ledger.records("plan")}
    obs_ss: dict[str, list[float]] = {}
    mod_ss: dict[str, list[float]] = {}
    obs_fee: dict[str, list[float]] = {}
    mod_fee: dict[str, list[float]] = {}
    short: dict[str, list[float]] = {}
    skipped: dict[str, dict[str, int]] = {}
    for f in ledger.rows(kind="fill", since=since, until=until):
        if f.get("book") not in books:
            continue
        pl = plans.get(f["decision_id"])
        if pl is None:
            continue
        ac = pl.asset_class
        sk = skipped.setdefault(ac, {})
        cm = base.get(ac)
        if cm is None:
            sk["no_cost_model"] = sk.get("no_cost_model", 0) + 1
            continue
        ts = pd.Timestamp(f["time"])
        side, qty, price = int(f["side"]), float(f["qty"]), float(f["price"])
        bu = 1.0
        if ac == "forex":
            if rate_fn is None:
                sk["no_rate_function"] = sk.get("no_rate_function", 0) + 1
                continue
            try:
                bu = float(rate_fn(split_pair(f["symbol"])[1], ts))
            except (RateMissing, LookupError, KeyError):
                sk["rate_missing"] = sk.get("rate_missing", 0) + 1
                continue
        try:
            _, unit = cm.fill(f["symbol"], side, price, ts)
            fees = sum(cm.order_fees(f["symbol"], side, qty, price, ts).values())
        except (RateMissing, LookupError):
            sk["spread_missing"] = sk.get("spread_missing", 0) + 1
            continue
        obs_ss.setdefault(ac, []).append(float(f["spread_slippage_usd"]))
        mod_ss.setdefault(ac, []).append((unit["spread"] + unit["slippage"]) * qty * bu)
        obs_fee.setdefault(ac, []).append(float(f["fees_usd"]))
        mod_fee.setdefault(ac, []).append(fees)
        if side == pl.direction and pl.entry_price > 0:
            short.setdefault(ac, []).append(1e4 * side * (price - pl.entry_price) / pl.entry_price)
    out: dict[str, CostCalibration] = {}
    for ac in sorted(set(skipped) | set(base)):
        n = len(obs_ss.get(ac, []))
        cal = CostCalibration(ac, n_fills=n, skipped={k: v for k, v in skipped.get(ac, {}).items() if v})
        if n:
            os_, ms_ = np.array(obs_ss[ac]), np.array(mod_ss[ac])
            of_, mf_ = np.array(obs_fee[ac]), np.array(mod_fee[ac])
            cal.spread_ratio, cal.fee_ratio = _ratio(os_, ms_), _ratio(of_, mf_)
            cal.spread_ci, cal.fee_ci = _boot_ci(os_, ms_, cfg.bootstrap, cfg.seed), _boot_ci(of_, mf_, cfg.bootstrap, cfg.seed)
            cal.observed_usd = {"spread_slippage": round(float(os_.sum()), 4), "fees": round(float(of_.sum()), 4)}
            cal.model_usd = {"spread_slippage": round(float(ms_.sum()), 4), "fees": round(float(mf_.sum()), 4)}
            sh = np.array(short.get(ac, []))
            if len(sh):
                cal.entry_shortfall_bps = round(float(sh.mean()), 3)
                cal.entry_shortfall_se_bps = round(float(sh.std(ddof=1) / np.sqrt(len(sh))), 3) if len(sh) > 1 else None
        if n < cfg.min_fills:
            cal.status = f"{n} usable fills, need {cfg.min_fills}: model unchanged"
        else:
            cal.spread_mult = _shrunk(cal.spread_ratio, n, cfg)
            cal.fee_mult = _shrunk(cal.fee_ratio, n, cfg)
            cal.calibrated = True
            cal.status = f"calibrated on {n} fills"
        out[ac] = cal
    return out


def calibrated_costs(costs: dict[str, CostModel], cal: dict[str, CostCalibration]) -> dict[str, CostModel]:
    """The cost models the core should use: wrapped where a class calibrated, otherwise the originals."""
    out: dict[str, CostModel] = {}
    for ac, m in costs.items():
        c = cal.get(ac)
        out[ac] = CalibratedCosts(m, c.spread_mult, c.fee_mult) if c is not None and c.calibrated else m
    return out


def calibration_report(cal: dict[str, CostCalibration]) -> list[dict[str, Any]]:
    """Rows for the dashboard and Telegram."""
    rows = []
    for ac, c in sorted(cal.items()):
        rows.append({"asset_class": ac, "fills": c.n_fills, "status": c.status, "calibrated": c.calibrated,
                     "spread_mult": round(c.spread_mult, 3), "fee_mult": round(c.fee_mult, 3),
                     "spread_ratio": None if c.spread_ratio is None else round(c.spread_ratio, 3),
                     "fee_ratio": None if c.fee_ratio is None else round(c.fee_ratio, 3),
                     "entry_shortfall_bps": c.entry_shortfall_bps, "skipped": c.skipped})
    return rows
