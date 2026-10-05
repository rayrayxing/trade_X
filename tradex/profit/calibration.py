"""Calibrated trade probability and expected value in R, from forward results (profit spec P4).

The plan finaliser labels its probability ``base_rate``: the mean of the agreeing strategies'
backtest hit rates. Backtest hit rates are optimistic, so this module learns the map from that
number to how often plans actually reach target 1 before the stop, using only real forward
trades read from the ledger (paper and live closes of the ensemble book; never counterfactuals,
never replays unless the caller says so).

It refuses to be clever with thin evidence. Below ``min_trades`` closed forward trades (or when
every outcome is the same) ``predict`` returns the base rate unchanged, flagged
``calibrated=False``, and ``annotate`` leaves the plan exactly as the finaliser wrote it.
From ``min_trades`` it fits Platt scaling (two parameters, safe on small samples); from
``isotonic_min`` it uses isotonic regression (pool-adjacent-violators). Cross-validated Brier
scores against the raw base rate and a constant forward rate are reported next to every fit, so a
calibration that does not beat the raw number says so, and ``annotate`` then declines to use it.

Expected value in R is the plan's two-outcome form, ``p * reward_risk - (1 - p) * loss_r``, with
``loss_r`` measured on forward trades that did not reach target 1 (their mean loss in R, which
can sit under or over 1 because of time stops, breakeven exits and gaps).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Iterable

import numpy as np

from tradex.backtest.metrics import wilson_lower
from tradex.core.records import TradePlan

EPS = 1e-3


@dataclass(frozen=True)
class CalibrationConfig:
    min_trades: int = 100                # below this: the base rate, flagged uncalibrated
    isotonic_min: int = 300              # below this Platt is used even if method="isotonic" was not forced
    method: str = "auto"                 # auto | platt | isotonic
    cv_folds: int = 5
    min_loss_trades: int = 20            # non-target trades needed to measure loss_r; else 1.0
    require_cv_gain: bool = True         # annotate only when the CV Brier beats the raw base rate's
    label: str = "target"                # target: reached target 1 first | profit: closed with R > 0

    def __post_init__(self) -> None:
        if self.method not in ("auto", "platt", "isotonic"):
            raise ValueError(f"unknown calibration method {self.method!r}")
        if self.label not in ("target", "profit"):
            raise ValueError(f"unknown label {self.label!r}")


@dataclass(frozen=True)
class ForwardTrade:
    decision_id: str
    close_time: str
    p_raw: float                         # the plan's probability as the finaliser wrote it
    y: int                               # 1: reached target 1 first (or profit, per config)
    r: float                             # total R of the trade
    strategies: tuple[str, ...] = ()


@dataclass(frozen=True)
class CalibratedProbability:
    p: float
    p_low: float                         # one-sided 95% lower bound, for sizing that must not trust a thin edge
    p_raw: float
    calibrated: bool
    method: str                          # base_rate | platt | isotonic
    n_forward: int
    ev_r: float | None
    reason: str = ""


# --- reading forward trades ------------------------------------------------------------------------

def forward_trades(ledger, books: Iterable[str] = ("ensemble",), since: str | None = None, until: str | None = None,
                   label: str = "target") -> list[ForwardTrade]:
    """Closed, fully exited trades of the given books whose last close is in [since, until].

    A trade is complete when the quantity closed has reached the quantity filled on entry, so a partial exit
    does not count until the rest is out. ``until`` is how a replay avoids learning from its own future."""
    books = set(books)
    plans = {p.decision_id: p for p in ledger.records("plan") if p.book in books}
    if not plans:
        return []
    entry_qty: dict[str, float] = {}
    for f in ledger.rows(kind="fill"):
        pl = plans.get(f["decision_id"])
        if pl is not None and f["side"] == pl.direction:
            entry_qty[f["decision_id"]] = entry_qty.get(f["decision_id"], 0.0) + f["qty"]
    closes: dict[str, list[dict]] = {}
    for c in ledger.rows(kind="close"):
        if c["decision_id"] in plans:
            closes.setdefault(c["decision_id"], []).append(c)
    out = []
    for did, cs in closes.items():
        cs.sort(key=lambda c: (c["time"], c["_seq"]))
        done = sum(c["qty"] for c in cs)
        if not entry_qty.get(did) or done < entry_qty[did] * (1 - 1e-6):
            continue
        last = cs[-1]["time"]
        if (since is not None and last < since) or (until is not None and last > until):
            continue
        r = float(sum(c["r_multiple"] for c in cs))
        y = int(cs[0]["reason"].startswith("target")) if label == "target" else int(r > 0)
        pl = plans[did]
        out.append(ForwardTrade(did, last, float(pl.p_target), y, r, tuple(pl.strategies)))
    out.sort(key=lambda t: t.close_time)
    return out


# --- the two calibrators ------------------------------------------------------------------------------

def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def fit_platt(p_raw: np.ndarray, y: np.ndarray, iters: int = 100) -> tuple[float, float]:
    """Platt scaling P = sigmoid(a * logit(p_raw) + b), Newton's method, Platt's smoothed targets, a small ridge on
    the slope. When p_raw barely varies the slope is not identifiable and a = 0: b alone carries the forward rate."""
    x, y = _logit(np.asarray(p_raw, float)), np.asarray(y, float)
    n1, n0 = y.sum(), len(y) - y.sum()
    t = np.where(y > 0.5, (n1 + 1) / (n1 + 2), 1 / (n0 + 2))
    if float(np.var(x)) < 1e-8:
        m = float(t.mean())
        return 0.0, math.log(m / (1 - m))
    a, b = 1.0, 0.0
    ridge = 1e-3
    for _ in range(iters):
        p = _sigmoid(a * x + b)
        g = np.array([((p - t) * x).sum() + ridge * (a - 1.0), (p - t).sum()])
        w = p * (1 - p)
        hess = np.array([[(w * x * x).sum() + ridge, (w * x).sum()], [(w * x).sum(), w.sum() + 1e-9]])
        step = np.linalg.solve(hess, g)
        a, b = a - step[0], b - step[1]
        if np.abs(step).max() < 1e-9:
            break
    return float(a), float(b)


def fit_isotonic(p_raw: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pool-adjacent-violators on y ordered by p_raw. Returns (x knots, fitted values), both increasing in x."""
    p_raw, y = np.asarray(p_raw, float), np.asarray(y, float)
    order = np.argsort(p_raw, kind="stable")
    xs, ys = p_raw[order], y[order]
    ux, inv = np.unique(xs, return_inverse=True)           # ties share one point
    w = np.bincount(inv).astype(float)
    v = np.bincount(inv, weights=ys) / w
    vals, wts, spans = list(v), list(w), [[i, i] for i in range(len(ux))]
    i = 0
    while i < len(vals) - 1:
        if vals[i] > vals[i + 1] + 1e-12:
            tot = wts[i] + wts[i + 1]
            vals[i] = (vals[i] * wts[i] + vals[i + 1] * wts[i + 1]) / tot
            wts[i] = tot
            spans[i][1] = spans[i + 1][1]
            del vals[i + 1], wts[i + 1], spans[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    fit = np.empty(len(ux))
    for val, (a, b) in zip(vals, spans):
        fit[a:b + 1] = val
    return ux, fit


def _iso_predict(knots: np.ndarray, fit: np.ndarray, p: float) -> float:
    return float(np.interp(p, knots, fit))


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


# --- the model ------------------------------------------------------------------------------------------

@dataclass
class TradeProbabilityModel:
    cfg: CalibrationConfig
    n: int
    method: str = "base_rate"              # base_rate | platt | isotonic
    reason: str = ""
    a: float = 1.0
    b: float = 0.0
    knots: list[float] = field(default_factory=list)
    fitted: list[float] = field(default_factory=list)
    loss_r: float = 1.0
    forward_rate: float | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    beats_raw: bool = False

    @property
    def calibrated(self) -> bool:
        return self.method != "base_rate"

    @classmethod
    def fit(cls, trades: list[ForwardTrade], cfg: CalibrationConfig | None = None) -> "TradeProbabilityModel":
        cfg = cfg or CalibrationConfig()
        n = len(trades)
        m = cls(cfg, n)
        if n == 0:
            m.reason = f"no closed forward trades yet (need {cfg.min_trades})"
            return m
        y = np.array([t.y for t in trades], float)
        m.forward_rate = float(y.mean())
        if n < cfg.min_trades:
            m.reason = f"{n} closed forward trades, need {cfg.min_trades}"
            return m
        if y.min() == y.max():
            m.reason = f"all {n} forward trades had the same outcome"
            return m
        p = np.array([t.p_raw for t in trades], float)
        method = cfg.method if cfg.method != "auto" else ("isotonic" if n >= cfg.isotonic_min else "platt")
        if method == "isotonic" and n < cfg.isotonic_min and cfg.method == "isotonic":
            method = "platt"
            m.reason = f"isotonic needs {cfg.isotonic_min} trades; using platt"
        m.method = method
        m._fit_all(p, y)
        lost = np.array([-t.r for t in trades if t.y == 0], float)
        if len(lost) >= cfg.min_loss_trades:
            m.loss_r = max(0.0, float(lost.mean()))
        m.diagnostics = m._cross_validate(p, y)
        m.beats_raw = m.diagnostics["brier_cv"] < m.diagnostics["brier_raw"]
        return m

    def _fit_all(self, p: np.ndarray, y: np.ndarray) -> None:
        if self.method == "platt":
            self.a, self.b = fit_platt(p, y)
        else:
            k, f = fit_isotonic(p, y)
            self.knots, self.fitted = k.tolist(), f.tolist()

    def _apply(self, p_raw: float) -> float:
        if self.method == "platt":
            q = float(_sigmoid(np.array(self.a * _logit(np.array(p_raw)) + self.b)))
        elif self.method == "isotonic":
            q = _iso_predict(np.array(self.knots), np.array(self.fitted), p_raw)
        else:
            return p_raw
        return min(1 - EPS, max(EPS, q))

    def _cross_validate(self, p: np.ndarray, y: np.ndarray) -> dict[str, float]:
        n = len(y)
        folds = np.arange(n) % self.cfg.cv_folds           # time-ordered interleave: every fold spans the sample
        pred = np.empty(n)
        for f in range(self.cfg.cv_folds):
            tr, te = folds != f, folds == f
            sub = TradeProbabilityModel(self.cfg, int(tr.sum()), method=self.method)
            sub._fit_all(p[tr], y[tr])
            pred[te] = [sub._apply(v) for v in p[te]]
        return {"brier_raw": brier(np.clip(p, EPS, 1 - EPS), y), "brier_cv": brier(pred, y),
                "brier_constant": brier(np.full(n, y.mean()), y), "forward_rate": float(y.mean())}

    def ev_r(self, p: float, reward_risk: float) -> float:
        return round(p * reward_risk - (1 - p) * self.loss_r, 4)

    def predict(self, p_raw: float, reward_risk: float | None = None) -> CalibratedProbability:
        if not self.calibrated:
            return CalibratedProbability(p_raw, 0.0, p_raw, False, "base_rate", self.n,
                                         None if reward_risk is None else round(p_raw * reward_risk - (1 - p_raw), 4),
                                         self.reason)
        p = self._apply(p_raw)
        low = wilson_lower(int(round(p * self.n)), self.n)
        return CalibratedProbability(round(p, 4), round(min(low, p), 4), p_raw, True, self.method, self.n,
                                     None if reward_risk is None else self.ev_r(p, reward_risk), self.reason)

    def annotate(self, plan: TradePlan) -> TradePlan:
        """The plan with its probability and EV replaced by the calibrated ones, ``p_source="model"``; the plan
        unchanged when the model is uncalibrated or does not beat the raw base rate out of sample."""
        if not self.calibrated or (self.cfg.require_cv_gain and not self.beats_raw):
            return plan
        c = self.predict(plan.p_target, plan.reward_risk)
        return replace(plan, p_target=c.p, p_source="model", ev_r=c.ev_r if c.ev_r is not None else plan.ev_r)

    def reliability(self, trades: list[ForwardTrade], bins: int = 5) -> list[dict[str, Any]]:
        """Predicted versus realised hit rate by quantile bin, for the dashboard."""
        if not trades:
            return []
        pr = np.array([self.predict(t.p_raw).p for t in trades])
        y = np.array([t.y for t in trades], float)
        edges = np.quantile(pr, np.linspace(0, 1, bins + 1))
        rows = []
        for i in range(bins):
            hi = edges[i + 1]
            sel = (pr >= edges[i]) & ((pr <= hi) if i == bins - 1 else (pr < hi))
            if sel.any():
                rows.append({"predicted": round(float(pr[sel].mean()), 4), "realised": round(float(y[sel].mean()), 4),
                             "n": int(sel.sum())})
        return rows

    def to_dict(self) -> dict[str, Any]:
        return {"method": self.method, "n": self.n, "a": self.a, "b": self.b, "knots": self.knots,
                "fitted": self.fitted, "loss_r": self.loss_r, "forward_rate": self.forward_rate,
                "reason": self.reason, "beats_raw": self.beats_raw, "diagnostics": self.diagnostics}


def fit_from_ledger(ledger, cfg: CalibrationConfig | None = None, books: Iterable[str] = ("ensemble",),
                    since: str | None = None, until: str | None = None) -> TradeProbabilityModel:
    cfg = cfg or CalibrationConfig()
    return TradeProbabilityModel.fit(forward_trades(ledger, books, since, until, cfg.label), cfg)
