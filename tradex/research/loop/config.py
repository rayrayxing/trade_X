"""Loop settings: workload bounds, the lenient screen, the paper queue and the health monitor.

The stage-3 gate itself (trades, profit factor, deflated Sharpe, drawdown, positive folds)
is NOT here. It is read from the protected ``config/gates/thresholds.yaml`` by the
evaluator, and the loop can only add checks on top of it, never relax one. This file
holds the knobs around the gate. It lives in ``config/research_loop.yaml`` (not a
protected path) and every key is optional.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "config" / "research_loop.yaml"
DEFAULT_DB = ROOT / "data" / "research" / "loop.sqlite"
DEFAULT_RESULTS = ROOT / "research" / "results" / "loop"
DEFAULT_IDEAS_DIR = ROOT / "research" / "ideas"
DB_ENV = "TRADEX_LOOP_DB"


@dataclass(frozen=True)
class LoopConfig:
    # propose
    catalog_kinds: tuple[str, ...] = ("Build",)    # catalog status kinds that become ideas
    max_new_ideas: int = 10                        # per run, across all sources
    # implement
    max_drafts_per_run: int = 10
    # screen: a cheap look at the spec's own parameters on research-window data only
    screen_min_bars: int = 250
    screen_min_trades: int = 30
    screen_min_profit_factor: float = 1.0
    max_screens_per_run: int = 20
    # walk-forward (the stage-3 gate); compute is the limit, not the thresholds
    max_walk_forwards_per_run: int = 5
    # holdout
    holdout_min_trades: int = 20
    holdout_min_days: int = 60                     # holdout bars must cover at least this many calendar days
    # paper queue
    max_paper: int = 5
    max_paper_per_family: int = 2
    # health: lower CUSUM on standardised daily returns against the walk-forward baseline
    cusum_k: float = 0.5                           # allowance, in baseline standard deviations
    cusum_h: float = 8.0                           # alarm level (k=0.5, h=8: about one false alarm in 10,000 days)
    cusum_warn_frac: float = 0.5                   # alert (no demotion) at this share of h
    cusum_mean_shrink: float = 0.5                 # share of the walk-forward mean credited as the reference mean
    health_min_obs: int = 20                       # paper days needed before health is judged
    health_dd_multiple: float = 1.5                # demote when paper drawdown is this many times the baseline's
    stale_weeks: int = 12                          # validated, never promoted, for this long -> retired
    # status changes into and out of paper only happen with apply=True
    apply: bool = False

    @classmethod
    def load(cls, path: str | Path | None = None, **overrides) -> "LoopConfig":
        p = Path(path) if path else DEFAULT_CONFIG
        doc = (yaml.safe_load(p.read_text()) or {}) if p.exists() else {}
        known = {f.name for f in fields(cls)}
        bad = sorted(set(doc) - known)
        if bad:
            raise ValueError(f"{p}: unknown research-loop settings {bad}")
        vals = {k: (tuple(v) if k == "catalog_kinds" else v) for k, v in doc.items()}
        vals.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**vals)


def db_path(path: str | Path | None = None) -> Path:
    return Path(path or os.environ.get(DB_ENV) or DEFAULT_DB)
