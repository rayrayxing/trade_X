"""Ruin radar (profit spec P9): replay named historical shocks over the book that is open now.

For each named window in ``config/shock_windows.yaml`` (dates only) the radar compounds the ACTUAL
daily factor returns inside the window (``FX:<ccy>`` and ``EQ:<symbol>``, the same factors the risk
gate's expected-shortfall model uses) and applies them to the open book's factor exposures. The shock
sizes come from price history, never from this file or from memory; a window the history does not
cover is reported as ``no_data``, and exposure the history cannot price is reported as blind, not
treated as safe. The risk policy's own named scenarios are replayed alongside (source ``policy``).

What comes back is a ``RuinReport``: loss per scenario, the worst one, an expected-shortfall number,
and flags the Telegram bot and the dashboard can show as they are.

Expected shortfall here is two numbers and the headline is the larger:
- ``scenario_es_usd``: the mean loss of the worst quarter of the covered scenarios;
- ``hist_es_usd`` (1 day and ``horizon_days``): the mean of the worst 2.5% of rolling windows of the
  actual factor history applied to the book, when there are at least ``min_history_days`` of it.

It only reads exposures and returns; it places and changes nothing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from tradex.risk.exposure import Leg, Scenario, book_exposures, scenarios_from_config

RateFn = Callable[[str, pd.Timestamp], float]


# --- inputs -------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ShockWindow:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp
    note: str = ""


def load_shock_windows(path: str | Path = "config/shock_windows.yaml") -> list[ShockWindow]:
    import yaml
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"shock windows file {p} not found")
    out = []
    for w in (yaml.safe_load(p.read_text()) or {}).get("windows", []):
        s, e = pd.Timestamp(w["start"], tz="UTC"), pd.Timestamp(w["end"], tz="UTC")
        if e < s:
            raise ValueError(f"window {w['name']}: end before start")
        out.append(ShockWindow(w["name"], s, e, w.get("note", "")))
    return out


@dataclass(frozen=True)
class RuinLimits:
    equity: float
    stress_loss_cap_pct: float | None = None       # the policy's worst-scenario cap (book.stress_loss_cap_pct)
    es_budget_pct: float | None = None             # the policy's one-day expected-shortfall budget
    ruin_drawdown: float = 0.5
    warn_fraction: float = 0.5                     # warn at this share of a cap
    concentration: float = 0.6
    blind_share: float = 0.25
    gap_multiple: float = 2.0                      # worst loss / loss-to-stop above this: stops will not hold

    @classmethod
    def from_policy(cls, policy: dict[str, Any], equity: float, **kw) -> "RuinLimits":
        bk = policy.get("book", {})
        return cls(equity, bk.get("stress_loss_cap_pct"), bk.get("es_budget_pct"), **kw)


def legs_from_snapshot(snap: dict[str, Any], marks: dict[str, float] | None = None) -> list[Leg]:
    """Legs from an ``EquitySnapshot`` payload. Snapshots do not carry the asset class, so it is read from the symbol's
    shape: ``AAA_BBB`` is a forex pair, anything else a stock. Ray's own holdings count (they are real exposure)."""
    out = []
    for p in snap.get("positions", []):
        sym = p["symbol"]
        price = (marks or {}).get(sym, p.get("mark") or p["entry"])
        out.append(Leg(sym, "forex" if "_" in sym else "stocks", int(p["direction"]), float(p["qty"]), float(price),
                       p.get("stop"), p.get("account", "agent")))
    return out


def exposures_from_legs(legs: list[Leg], rate_fn: RateFn, ts: pd.Timestamp) -> dict[str, float]:
    """Factor exposures in USD. Rates come from ``rate_fn`` (live or ledger-recorded); nothing is assumed."""
    ccys = {c for l in legs if l.asset_class == "forex" for c in l.symbol.split("_")} - {"USD"}
    fx = {c: pd.Series([float(rate_fn(c, ts))], index=[ts]) for c in ccys}
    return book_exposures(legs, fx, ts)


def stop_risk_usd(legs: list[Leg], rate_fn: RateFn, ts: pd.Timestamp) -> float:
    """Loss to the stops if every one is hit at its level (no gap), in USD."""
    tot = 0.0
    for l in legs:
        if l.stop is None:
            continue
        bu = 1.0 if l.asset_class != "forex" or l.symbol.split("_")[1] == "USD" else float(rate_fn(l.symbol.split("_")[1], ts))
        tot += max(0.0, l.direction * (l.price - l.stop)) * l.qty * bu
    return tot


# --- the replay ---------------------------------------------------------------------------------------

def window_shocks(returns: pd.DataFrame, w: ShockWindow, min_obs: int = 2, min_cover: float = 0.6
                  ) -> tuple[dict[str, float], int, list[str]]:
    """Compounded factor returns inside the window. Returns (shocks, days of data, factors without enough data).
    A factor needs ``min_obs`` returns and ``min_cover`` of the window's weekdays."""
    sub = returns[(returns.index >= w.start) & (returns.index <= w.end + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))]
    weekdays = max(1, int(np.busday_count(w.start.date(), (w.end + pd.Timedelta(days=1)).date())))
    shocks: dict[str, float] = {}
    blind: list[str] = []
    for col in returns.columns:
        r = sub[col].dropna()
        if len(r) >= min_obs and len(r) >= min_cover * weekdays:
            shocks[col] = float(np.prod(1.0 + r.to_numpy()) - 1.0)
        else:
            blind.append(col)
    return shocks, len(sub), blind


@dataclass
class ScenarioResult:
    name: str
    source: str                                     # history | policy
    status: str                                     # ok | no_data | partial
    pnl_usd: float = 0.0                            # negative is a loss
    pnl_pct: float = 0.0
    top_factors: list[tuple[str, float]] = field(default_factory=list)
    blind_exposure_usd: float = 0.0                 # gross exposure the window's history could not price
    blind_factors: list[str] = field(default_factory=list)
    note: str = ""

    @property
    def loss_usd(self) -> float:
        return max(0.0, -self.pnl_usd)


@dataclass
class Flag:
    level: str                                      # info | warn | breach
    code: str
    message: str


@dataclass
class RuinReport:
    equity: float
    exposures: dict[str, float]
    results: list[ScenarioResult]
    scenario_es_usd: float | None
    hist_es_1d_usd: float | None
    hist_es_horizon_usd: float | None
    horizon_days: int
    es_usd: float | None
    flags: list[Flag]
    stop_risk_usd: float | None = None

    @property
    def worst(self) -> ScenarioResult | None:
        cov = [r for r in self.results if r.status != "no_data"]
        return min(cov, key=lambda r: r.pnl_usd) if cov else None

    @property
    def level(self) -> str:
        order = {"info": 0, "warn": 1, "breach": 2}
        return max((f.level for f in self.flags), key=order.__getitem__, default="info")

    def to_dict(self) -> dict[str, Any]:
        w = self.worst
        return {"equity": self.equity, "level": self.level, "worst": None if w is None else
                {"name": w.name, "pnl_usd": round(w.pnl_usd, 2), "pnl_pct": round(w.pnl_pct, 4)},
                "es_usd": _r(self.es_usd), "scenario_es_usd": _r(self.scenario_es_usd),
                "hist_es_1d_usd": _r(self.hist_es_1d_usd), "hist_es_horizon_usd": _r(self.hist_es_horizon_usd),
                "horizon_days": self.horizon_days, "stop_risk_usd": _r(self.stop_risk_usd),
                "scenarios": [{"name": r.name, "source": r.source, "status": r.status, "pnl_usd": round(r.pnl_usd, 2),
                               "pnl_pct": round(r.pnl_pct, 4), "top_factors": [[k, round(v, 2)] for k, v in r.top_factors],
                               "blind_exposure_usd": round(r.blind_exposure_usd, 2), "note": r.note}
                              for r in sorted(self.results, key=lambda r: r.pnl_usd)],
                "flags": [{"level": f.level, "code": f.code, "message": f.message} for f in self.flags],
                "exposures": {k: round(v, 2) for k, v in sorted(self.exposures.items())}}

    def telegram(self, top: int = 3) -> str:
        w = self.worst
        if not self.exposures:
            return "Ruin radar: no open positions."
        lines = [f"Ruin radar [{self.level.upper()}] equity ${self.equity:,.0f}"]
        if w is not None:
            lines.append(f"Worst replay: {w.name} {w.pnl_usd:+,.0f} USD ({w.pnl_pct:+.1%})")
        if self.es_usd is not None:
            lines.append(f"Expected shortfall: {self.es_usd:,.0f} USD ({self.es_usd / self.equity:.1%})")
        cov = sorted((r for r in self.results if r.status != "no_data"), key=lambda r: r.pnl_usd)[:top]
        lines += [f"  {r.name}: {r.pnl_usd:+,.0f}" for r in cov]
        lines += [f"{f.level.upper()}: {f.message}" for f in self.flags if f.level != "info"]
        return "\n".join(lines)


def _r(x: float | None) -> float | None:
    return None if x is None else round(float(x), 2)


def _es(pnl: np.ndarray, alpha: float) -> float:
    k = max(1, int(math.ceil(len(pnl) * (1 - alpha) - 1e-9)))     # the epsilon: 600 * (1 - 0.975) is 15.000000000000013
    return float(-np.sort(pnl)[:k].mean())


def hist_es(returns: pd.DataFrame, exposures: dict[str, float], horizon_days: int = 1, alpha: float = 0.975,
            min_history_days: int = 250, window: int = 1000) -> tuple[float | None, list[str]]:
    """Expected shortfall of the book over rolling ``horizon_days`` windows of actual factor history.
    Returns (ES in USD or None when history is short, factors the history lacks)."""
    cols = [k for k in exposures if k in returns.columns]
    missing = [k for k in exposures if k not in returns.columns]
    if not cols:
        return None, missing
    r = returns[cols].tail(window).fillna(0.0)
    if horizon_days > 1:
        r = (1.0 + r).rolling(horizon_days).apply(np.prod, raw=True).dropna() - 1.0
    if len(r) < min_history_days - horizon_days + 1:
        return None, missing
    e = np.array([exposures[k] for k in cols])
    return _es(r.to_numpy() @ e, alpha), missing


def ruin_report(exposures: dict[str, float], limits: RuinLimits, returns: pd.DataFrame,
                windows: Iterable[ShockWindow], policy_scenarios: Iterable[Scenario] = (),
                stop_risk: float | None = None, horizon_days: int = 5, tail_fraction: float = 0.25,
                min_history_days: int = 250) -> RuinReport:
    eq = limits.equity
    results: list[ScenarioResult] = []
    for w in windows:
        shocks, days, blind = window_shocks(returns, w)
        if not shocks or days == 0:
            results.append(ScenarioResult(w.name, "history", "no_data", note="no factor history in this window"))
            continue
        contrib = {k: v * shocks[k] for k, v in exposures.items() if k in shocks}
        blind_exp = sum(abs(v) for k, v in exposures.items() if k not in shocks)
        pnl = float(sum(contrib.values()))
        status = "partial" if blind_exp > 0 else "ok"
        top = sorted(contrib.items(), key=lambda kv: kv[1])[:3]
        results.append(ScenarioResult(w.name, "history", status, pnl, pnl / eq if eq else 0.0, top, blind_exp,
                                      sorted(k for k in exposures if k not in shocks), w.note))
    for s in policy_scenarios:
        pnl = s.pnl(exposures)
        top = sorted(((k, v * s.shocks.get(k, s.default_equity if k.startswith("EQ:") else 0.0))
                      for k, v in exposures.items()), key=lambda kv: kv[1])[:3]
        results.append(ScenarioResult(s.name, "policy", "ok", pnl, pnl / eq if eq else 0.0, top, 0.0, [], s.note))

    covered = sorted((r for r in results if r.status != "no_data"), key=lambda r: r.pnl_usd)
    scen_es = None
    if covered:
        k = max(1, int(math.ceil(len(covered) * tail_fraction)))
        scen_es = max(0.0, -float(np.mean([r.pnl_usd for r in covered[:k]])))
    h1, miss1 = hist_es(returns, exposures, 1, min_history_days=min_history_days)
    hh, _ = hist_es(returns, exposures, horizon_days, min_history_days=min_history_days)
    es_candidates = [x for x in (scen_es, hh) if x is not None]
    rep = RuinReport(eq, dict(exposures), results, scen_es, h1, hh, horizon_days,
                     max(es_candidates) if es_candidates else None, [], stop_risk)
    rep.flags = _flags(rep, limits, miss1)
    return rep


def _flags(rep: RuinReport, lim: RuinLimits, hist_missing: list[str]) -> list[Flag]:
    fl: list[Flag] = []
    eq = lim.equity
    if not rep.exposures:
        return [Flag("info", "no_book", "no open positions: nothing to replay")]
    if eq <= 0:
        return [Flag("breach", "no_equity", "equity is not positive")]
    w = rep.worst
    if w is None:
        fl.append(Flag("warn", "no_history", "no shock window has factor history: the radar is blind"))
    else:
        loss_pct = w.loss_usd / eq
        if lim.stress_loss_cap_pct is not None:
            cap = lim.stress_loss_cap_pct / 100.0
            if loss_pct > cap:
                fl.append(Flag("breach", "stress_breach", f"{w.name} would cost {loss_pct:.1%} of equity, over the {cap:.0%} cap"))
            elif loss_pct > lim.warn_fraction * cap:
                fl.append(Flag("warn", "stress_warn", f"{w.name} would cost {loss_pct:.1%} of equity ({loss_pct / cap:.0%} of the {cap:.0%} cap)"))
        if loss_pct >= lim.ruin_drawdown:
            fl.append(Flag("breach", "ruin", f"{w.name} would take {loss_pct:.0%} of equity: past the {lim.ruin_drawdown:.0%} ruin line"))
        gross = sum(abs(v) for v in rep.exposures.values())
        if w.loss_usd > 0:
            k, v = w.top_factors[0] if w.top_factors else ("", 0.0)
            if v < 0 and abs(v) / w.loss_usd > lim.concentration:
                fl.append(Flag("warn", "concentration", f"{abs(v) / w.loss_usd:.0%} of the worst replay's loss is one factor, {k}"))
        if rep.stop_risk_usd and w.loss_usd > lim.gap_multiple * rep.stop_risk_usd:
            fl.append(Flag("warn", "gap_through_stops", f"{w.name} loses {w.loss_usd / rep.stop_risk_usd:.1f}x the loss-to-stop: "
                           "a gap would run through the stops"))
        if gross and w.blind_exposure_usd / gross > lim.blind_share:
            fl.append(Flag("warn", "blind", f"{w.blind_exposure_usd / gross:.0%} of exposure has no history in {w.name}: "
                           f"{', '.join(w.blind_factors[:4])}"))
    nodata = [r.name for r in rep.results if r.status == "no_data"]
    if nodata:
        fl.append(Flag("info", "windows_without_data", f"no history for: {', '.join(nodata)}"))
    if lim.es_budget_pct is not None:
        if rep.hist_es_1d_usd is None:
            fl.append(Flag("warn", "es_no_history", "not enough factor history for the one-day expected shortfall"))
        elif rep.hist_es_1d_usd > lim.es_budget_pct / 100.0 * eq:
            fl.append(Flag("breach", "es_breach", f"one-day expected shortfall {rep.hist_es_1d_usd:,.0f} USD is over the "
                           f"{lim.es_budget_pct:g}% budget"))
    if hist_missing:
        fl.append(Flag("warn", "missing_factors", f"no return history for {', '.join(hist_missing[:5])}: left out of expected shortfall"))
    return fl


def radar_from_snapshot(snap: dict[str, Any], returns: pd.DataFrame, windows: list[ShockWindow], rate_fn: RateFn,
                        policy: dict[str, Any] | None = None, ts: pd.Timestamp | None = None,
                        marks: dict[str, float] | None = None, **kw) -> RuinReport:
    """The whole thing from a ledger snapshot: legs, exposures at the rates ``rate_fn`` gives, the policy's limits and
    scenarios, the named windows."""
    ts = ts if ts is not None else pd.Timestamp(snap["time"])
    legs = legs_from_snapshot(snap, marks)
    expo = exposures_from_legs(legs, rate_fn, ts)
    pol = policy or {}
    lim = RuinLimits.from_policy(pol, float(snap["equity_usd"])) if pol else RuinLimits(float(snap["equity_usd"]))
    return ruin_report(expo, lim, returns, windows, scenarios_from_config(pol.get("scenarios", [])),
                       stop_risk_usd(legs, rate_fn, ts), **kw)
