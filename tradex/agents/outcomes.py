"""Outcome resolvers: what happened after an agent proposal, from bars and the ledger.

A resolver takes one shadow decision (dict with ``asof`` and ``body``) and ``now`` and returns
an ``Outcome`` or ``None`` when the answer is not knowable yet. All forward looks start at the
first bar that opens at or after the decision time, so the proposal can never see the bar it was
made on. Bars come from an injected ``bars_fn(symbol) -> DataFrame | None`` (the same OHLCV
frames the rest of tradex uses).

Units. One outcome can hold several scored items (a watchlist has up to 20 picks); ``units`` and
``hits`` add up across decisions, and ``value_sum / units`` is the mean value per item:

- scout: pick direction times (its return minus the universe's mean return); hit when positive
- position close: R saved = R at the proposal minus R the plan would have reached if held; hit
  when positive (counterfactual: the plan's own stop, target and time limit)
- chart: expected direction times the move over the horizon in ATR units; hit when positive
- incident: hit when a medium or high flag was followed by the same check failing again within
  the horizon, or a low flag was not (value +1 or -1)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean, median
from typing import Any, Callable

import pandas as pd

from tradex.core.ledger import Ledger

BarsFn = Callable[[str], "pd.DataFrame | None"]
Resolver = Callable[[dict[str, Any], pd.Timestamp], "Outcome | None"]


@dataclass
class Outcome:
    units: int
    hits: int
    value_sum: float
    detail: dict[str, Any] = field(default_factory=dict)


def _ts(x: Any) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _forward(df: pd.DataFrame | None, asof: pd.Timestamp, n: int) -> pd.DataFrame | None:
    """The ``n`` bars from the first one opening at or after ``asof``, or None if fewer exist."""
    if df is None or df.empty:
        return None
    fut = df[df.index >= asof]
    return fut.iloc[:n] if len(fut) >= n else None


class ScoutResolver:
    def __init__(self, bars_fn: BarsFn, universe: list[str], horizon_bars: int = 1, min_universe: int = 3,
                 grace: pd.Timedelta = pd.Timedelta(days=10)):
        self.bars_fn, self.universe, self.h = bars_fn, list(universe), horizon_bars
        self.min_universe, self.grace = min_universe, grace

    def __call__(self, d: dict[str, Any], now: pd.Timestamp) -> Outcome | None:
        asof = _ts(d["asof"])
        picks = d["body"].get("picks") or []
        if not picks:
            return None
        rets: dict[str, float] = {}
        for s in dict.fromkeys(self.universe + [p["symbol"] for p in picks]):
            f = _forward(self.bars_fn(s), asof, self.h)
            if f is not None and f["open"].iloc[0] > 0:
                rets[s] = float(f["close"].iloc[-1] / f["open"].iloc[0] - 1)
        have = [p for p in picks if p["symbol"] in rets]
        missing = [p["symbol"] for p in picks if p["symbol"] not in rets]
        if len(rets) < self.min_universe or not have:
            return None
        if missing and now - asof < self.grace:
            return None                       # data for some picks may still be arriving
        base = mean(rets.values())
        med_abs = median(abs(r - base) for r in rets.values())
        hits, value, per = 0, 0.0, []
        for p in have:
            adj = rets[p["symbol"]] - base
            if p["direction"]:
                v = p["direction"] * adj
            else:
                v = abs(adj) - med_abs
            hits += v > 0
            value += v
            per.append({"symbol": p["symbol"], "direction": p["direction"], "ret": round(rets[p["symbol"]], 5),
                        "value": round(v, 5)})
        return Outcome(len(have), int(hits), value, {"baseline_ret": round(base, 5), "missing": missing, "picks": per})


class PositionResolver:
    """Replays the plan's own stop, target and time limit over the bars after the proposal."""

    def __init__(self, bars_fn: BarsFn, grace: pd.Timedelta = pd.Timedelta(days=30), min_saved_r: float = 0.0):
        self.bars_fn, self.grace, self.min_saved_r = bars_fn, grace, min_saved_r

    def __call__(self, d: dict[str, Any], now: pd.Timestamp) -> Outcome | None:
        b, asof = d["body"], _ts(d["asof"])
        df = self.bars_fn(b["symbol"])
        if df is None or b.get("initial_risk", 0) <= 0:
            return None
        dirn, entry, stop, target = b["direction"], b["entry_price"], b["stop"], b["target"]
        remaining = max(1, int(b["max_bars"]) - int(b["bars_held"]))
        fut = df[df.index >= asof].iloc[:remaining]
        exit_px, reason = None, ""
        for ts, bar in fut.iterrows():
            stop_hit = bar["low"] <= stop if dirn > 0 else bar["high"] >= stop
            tgt_hit = bar["high"] >= target if dirn > 0 else bar["low"] <= target
            if stop_hit:                      # stop wins when both are touched; a gap fills at the open
                gap = bar["open"] <= stop if dirn > 0 else bar["open"] >= stop
                exit_px, reason = (float(bar["open"]) if gap else float(stop)), "stop"
                break
            if tgt_hit:
                exit_px, reason = float(target), "target"
                break
        if exit_px is None:
            if len(fut) < remaining and now - asof < self.grace:
                return None
            if fut.empty:
                return None
            exit_px, reason = float(fut["close"].iloc[-1]), "time"
        r_final = dirn * (exit_px - entry) / b["initial_risk"]
        saved = float(b["r_now"]) - r_final
        return Outcome(1, int(saved > self.min_saved_r), saved,
                       {"r_at_proposal": b["r_now"], "r_if_held": round(r_final, 4), "exit_if_held": reason})


class ChartResolver:
    def __init__(self, bars_fn: BarsFn, horizon_bars: int = 5, grace: pd.Timedelta = pd.Timedelta(days=15)):
        self.bars_fn, self.h, self.grace = bars_fn, horizon_bars, grace

    def __call__(self, d: dict[str, Any], now: pd.Timestamp) -> Outcome | None:
        b, asof = d["body"], _ts(d["asof"])
        exp, atr = b.get("expected_direction", 0), b.get("atr", 0)
        if not exp or atr <= 0:
            return None
        f = _forward(self.bars_fn(b["symbol"]), asof, self.h)
        if f is None:
            return None
        move = float((f["close"].iloc[-1] - f["open"].iloc[0]) / atr)
        v = exp * move
        return Outcome(1, int(v > 0), v, {"move_atr": round(move, 4), "expected": exp})


class IncidentResolver:
    def __init__(self, ledger: Ledger, horizon: pd.Timedelta = pd.Timedelta(hours=24)):
        if not ledger.read_only:
            raise PermissionError("open the ledger with read_only=True to score incidents")
        self.ledger, self.horizon = ledger, horizon

    def __call__(self, d: dict[str, Any], now: pd.Timestamp) -> Outcome | None:
        b = d["body"]
        end = _ts(b["window_end"])
        if now < end + self.horizon:
            return None
        checks = set(b["checks"])
        again = [r for r in self.ledger.rows(kind="health")
                 if not r.get("ok") and r["check"] in checks and end < _ts(r["time"]) <= end + self.horizon]
        recurred = bool(again)
        hit = recurred if b["severity"] in ("medium", "high") else not recurred
        return Outcome(1, int(hit), 1.0 if hit else -1.0, {"recurred": recurred, "later_failures": len(again)})
