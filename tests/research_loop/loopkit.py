"""Fixtures and fakes for the research-loop tests. Synthetic bars are fine here: tests only, never the loop itself."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pandas as pd
import yaml

from tradex.backtest.validation import Thresholds
from tradex.data.synthetic import synthetic_bars
from tradex.research import catalog as catalog_mod
from tradex.research.loop.adapters import CollectingAlerter
from tradex.research.loop.config import LoopConfig
from tradex.research.loop.ports import (DataUnavailable, HoldoutAlreadyLooked, Idea, LiveDataMissing, Ports)
from tradex.research.loop.sources import FileCatalog, FileIdeaSource, FileSpecDrafter
from tradex.research.loop.specstore import YamlSpecStore
from tradex.research.loop.state import LoopState
from tradex.research.trials import TrialLedger
from tradex.timeframes import duration

HOLDOUT_START = pd.Timestamp("2022-01-03", tz="UTC")
NOW = pd.Timestamp("2026-10-05 12:00", tz="UTC")           # a Monday: ISO week 2026-W41


def spec_dict(sid: str = "stk-rsi-test", family: str = "mean_reversion", universe=("AAA", "BBB"), **over) -> dict[str, Any]:
    d = {"id": sid, "version": 1, "asset_class": "stocks", "family": family,
         "hypothesis": "Oversold names in an uptrend bounce within days.",
         "universe": list(universe), "timeframes": {"signal": "D1"},
         "features": {"rsi2": {"fn": "talib.RSI", "period": 2}, "ema50": {"fn": "talib.EMA", "period": 50}},
         "entry": {"long": "close > ema50 and rsi2 < 20"},
         "exit": {"stop_atr": 2.0, "target_r": 1.5, "max_bars": 7},
         "holding": {"expected_hours": 72}, "sizing": {"model": "kelly_fraction", "cap_risk_pct": 2},
         "search_space": {"stop_atr": [1.5, 3.0]},
         "provenance": {"source": "test fixture", "author": "test", "created": "2026-10-01"},
         "status": "proposed"}
    d.update(over)
    return d


def write_spec(directory, raw: dict) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    p = directory / f"{raw['id']}.yaml"
    p.write_text(yaml.safe_dump(raw, sort_keys=False, default_flow_style=None))
    return str(p)


def make_frames(symbols=("AAA", "BBB"), n: int = 1500, start: str = "2017-01-02", seed0: int = 1) -> dict[str, pd.DataFrame]:
    return {s: synthetic_bars(n, seed=seed0 + i, start=start) for i, s in enumerate(symbols)}


class FakeData:
    real = True
    source = "fixture-bars"

    def __init__(self, frames: dict[str, pd.DataFrame] | None = None, missing: set[str] | None = None):
        self.frames_by_symbol = frames if frames is not None else make_frames()
        self.missing = missing or set()
        self.calls: list[str] = []

    def frames(self, spec):
        self.calls.append(spec.id)
        if spec.id in self.missing:
            raise DataUnavailable(f"no bars for {spec.id}")
        return {s: self.frames_by_symbol[s] for s in spec.universe if s in self.frames_by_symbol}


class FakeHoldout:
    """The behaviour of tradex.backtest.holdout: cut research bars, one recorded look per version."""

    def __init__(self, start: pd.Timestamp = HOLDOUT_START):
        self.start = start
        self.looks: list[tuple[str, int, str]] = []
        self.look_log: list[str] = []          # which stage was running is recorded by the test through `where`

    def research_view(self, frames, tf):
        last_open = self.start - duration(tf)
        return {s: df[df.index <= last_open] for s, df in frames.items()}

    def has_looked(self, strategy_id, version):
        return any(k[:2] == (strategy_id, version) for k in self.looks)

    def look(self, frames, strategy_id, version, reason):
        if self.has_looked(strategy_id, version):
            raise HoldoutAlreadyLooked(f"{strategy_id} v{version} already looked")
        self.looks.append((strategy_id, version, reason))
        return {s: df[df.index >= self.start] for s, df in frames.items()}


class FakeLive:
    def __init__(self, series: dict[str, pd.Series] | None = None, missing: set[str] | None = None):
        self.series = series or {}
        self.missing = missing or set()

    def daily_returns(self, strategy_id, version, since):
        if strategy_id in self.missing:
            raise LiveDataMissing(f"no snapshots for {strategy_id}")
        return self.series.get(strategy_id)


def oos_returns(n: int = 250, mean: float = 0.0008, std: float = 0.01, seed: int = 3) -> pd.Series:
    import numpy as np
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mean, std, n), index=pd.bdate_range("2020-01-01", periods=n, tz="UTC"))


def good_report(spec, **over):
    oos = {"trades": 150, "profit_factor": 1.6, "dsr": 0.98, "max_drawdown": -0.12, "positive_folds": 0.8, "sharpe": 1.4}
    oos.update(over.pop("oos", {}))
    rep = SimpleNamespace(strategy_id=spec.id, version=spec.version, status="validated", reasons=[], n_trials=8,
                          n_trials_this_run=4, oos=oos, folds=[object()] * 5, param_stability=0.6,
                          recommended_params={"stop_atr": 2.0}, oos_returns=oos_returns(), oos_trades=None)
    for k, v in over.items():
        setattr(rep, k, v)
    return rep


def bad_report(spec):
    return good_report(spec, status="rejected", reasons=["deflated Sharpe 0.40 (need 0.95)"],
                       oos={"dsr": 0.40, "trades": 150})


class ScriptedEvaluator:
    """Returns scripted measurements and records the frames each call was given."""

    def __init__(self, thresholds: Thresholds | None = None):
        self.thresholds = thresholds or Thresholds(min_trades=100, min_profit_factor=1.2, min_dsr=0.95,
                                                   max_drawdown=-0.5, min_positive_folds=0.5)
        self.screen_out: dict[str, dict] = {}
        self.wf_out: dict[str, Any] = {}
        self.holdout_out: dict[str, dict] = {}
        self.calls: list[tuple[str, str]] = []
        self.seen_end: dict[tuple[str, str], pd.Timestamp] = {}
        self.fail: dict[tuple[str, str], Exception] = {}

    def _note(self, stage, spec, frames):
        self.calls.append((stage, spec.id))
        self.seen_end[(stage, spec.id)] = max(df.index[-1] for df in frames.values())
        if (stage, spec.id) in self.fail:
            raise self.fail[(stage, spec.id)]

    def screen(self, spec, frames, trials, run_id):
        self._note("screen", spec, frames)
        trials.record(spec.id, spec.version, {"stop_atr": spec.exit.stop_atr}, run_id=run_id, source="loop_screen")
        return self.screen_out.get(spec.id, {"trades": 60, "profit_factor": 1.3, "sharpe": 0.8, "max_drawdown": -0.1,
                                             "total_return": 0.4, "win_rate": 0.5, "costs_share_of_gross": 0.1,
                                             "params": {}, "warnings": []})

    def walk_forward(self, spec, frames, trials, data_key):
        self._note("walk_forward", spec, frames)
        trials.record_many([dict(strategy_id=spec.id, version=spec.version, params={"stop_atr": x}, run_id="wf")
                            for x in (1.5, 2.0, 2.5, 3.0)])
        rep = self.wf_out.get(spec.id) or good_report(spec)
        rep.n_trials = max(rep.n_trials, trials.count(spec.id))
        return rep

    def holdout(self, spec, params, frames, start):
        self._note("holdout", spec, frames)
        return self.holdout_out.get(spec.id, {"trades": 40, "profit_factor": 1.5, "sharpe": 1.1, "max_drawdown": -0.08,
                                               "total_return": 0.2, "win_rate": 0.5, "costs_share_of_gross": 0.1,
                                               "params": params, "days": 200, "warnings": []})


CATALOG_ROWS = [
    {"family": "Trend", "name": "Alpha momentum", "asset": "stocks", "horizon": "Weeks", "data": "Daily bars",
     "source": "Paper A 2020", "data_today": "Yes", "status": "Build", "arxiv": ["2001.00001"]},
    {"family": "Trend", "name": "Beta reversal", "asset": "FX", "horizon": "Days", "data": "Daily bars",
     "source": "Paper B 2021", "data_today": "Yes", "status": "Build", "arxiv": ["2002.00002"]},
    {"family": "Carry", "name": "Gamma carry", "asset": "FX", "horizon": "Months", "data": "Rates",
     "source": "Paper C", "data_today": "No", "status": "Needs data: rates", "arxiv": []},
    {"family": "Trend", "name": "Delta handled", "asset": "stocks", "horizon": "Weeks", "data": "Daily bars",
     "source": "Paper D", "data_today": "Yes", "status": "Build", "arxiv": []},
]


@dataclass
class Rig:
    ports: Ports
    state: LoopState
    cfg: LoopConfig
    alerts: CollectingAlerter
    specs: YamlSpecStore
    data: FakeData
    holdout: FakeHoldout
    live: FakeLive
    ev: ScriptedEvaluator
    trials: TrialLedger
    root: Any
    ideas_dir: Any
    drafts_dir: Any
    out_dir: Any
    sources: list = field(default_factory=list)

    def run(self, **kw):
        from tradex.research.loop.engine import run_loop
        kw.setdefault("out_dir", self.out_dir)
        return run_loop(self.ports, self.cfg, self.state, **kw)

    def status(self, sid: str, version: int | None = None) -> str:
        return self.state.strategy(sid, version)["status"]

    def file_status(self, sid: str) -> str:
        return self.specs.get(sid).status

    def add_spec(self, raw: dict | None = None, where: str = "proposed") -> dict:
        raw = raw or spec_dict()
        write_spec(self.root / "strategies" / where, raw)
        return raw


def make_rig(tmp_path, *, cfg: LoopConfig | None = None, evaluator: ScriptedEvaluator | None = None,
             catalog_rows=None, handled: dict | None = None, data: FakeData | None = None,
             holdout: FakeHoldout | None = None, live: FakeLive | None = None, extra_sources=()) -> Rig:
    root = tmp_path / "repo"
    (root / "strategies" / "proposed").mkdir(parents=True)
    (root / "strategies" / "seeds").mkdir(parents=True)
    ideas_dir, drafts_dir = root / "research" / "ideas", root / "research" / "ideas" / "specs"
    specs = YamlSpecStore(root / "strategies" / "proposed", root / "strategies" / "seeds")
    cat_path = tmp_path / "catalog.yaml"
    cat_path.write_text(yaml.safe_dump(catalog_rows if catalog_rows is not None else []))
    catalog = FileCatalog(cat_path, plans=[SimpleNamespace(catalog_id=k, spec="x.yaml", verdict=None, why=v)
                                           for k, v in (handled or {}).items()])
    alerts = CollectingAlerter()
    ev = evaluator or ScriptedEvaluator()
    data = data or FakeData()
    holdout = holdout or FakeHoldout()
    live = live or FakeLive()
    trials = TrialLedger(tmp_path / "loop-trials.sqlite")
    sources = [FileIdeaSource(ideas_dir), *extra_sources]
    ports = Ports(catalog=catalog, specs=specs, idea_sources=sources, drafter=FileSpecDrafter(drafts_dir), data=data,
                  holdout=holdout, live=live, alerts=alerts, evaluator=ev, trials=trials, now=lambda: NOW)
    state = LoopState(tmp_path / "loop.sqlite", now=lambda: NOW)
    cfg = cfg or LoopConfig(screen_min_bars=100)
    return Rig(ports=ports, state=state, cfg=cfg, alerts=alerts, specs=specs, data=data, holdout=holdout, live=live,
               ev=ev, trials=trials, root=root, ideas_dir=ideas_dir, drafts_dir=drafts_dir, out_dir=tmp_path / "out",
               sources=sources)


def idea_file(rig: Rig, name: str, **fields) -> None:
    rig.ideas_dir.mkdir(parents=True, exist_ok=True)
    doc = {"title": f"Idea {name}", "summary": "Test idea.", "family": "trend", "assets": ["stocks"]} | fields
    (rig.ideas_dir / f"{name}.yaml").write_text(yaml.safe_dump(doc))


def draft_file(rig: Rig, idea_id: str, raw: dict) -> None:
    rig.drafts_dir.mkdir(parents=True, exist_ok=True)
    from tradex.research.loop.sources import safe_name
    (rig.drafts_dir / f"{safe_name(idea_id)}.yaml").write_text(yaml.safe_dump(raw, sort_keys=False, default_flow_style=None))
