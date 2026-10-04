"""Chart reader agent: a candlestick and pattern read at a bar that is already knowable.

Look-ahead rule. ``knowable_bars`` keeps only bars that have CLOSED by ``asof`` (bar open
time plus its duration is at or before ``asof``; bars are stamped with their open time). The
model sees those bars and facts computed from them (candlestick detectors on the last bar,
ATR-zigzag swing points confirmed so far, moving averages, 20-bar range), nothing later, so a
read made from stored bars matches the read made live.

Output. The model gives pattern, bias, confidence and levels. What the agent may propose:

- with a trade plan (``ChartPlan``) whose direction the read contradicts: ``veto`` the plan's
  decision ID at high confidence, ``shrink`` it (factor 0.5) at moderate confidence;
- without a plan, a confident non-neutral read: ``flag`` on the symbol.

Every proposal records ``expected_direction`` (the way the agent thinks price moves from
here), which is what the scorecard checks against later bars.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import talib

from tradex.agents.common import (DECISION_ID_RE, INJECTION_GUARD, SYMBOL_RE, InboxSink, ModelGateway, ShadowStore,
                                  ask, clean, data_block, num, parse_json)
from tradex.ta.registry import FeatureContext, double_top_bottom, zigzag_pivots
from tradex.timeframes import duration

AGENT = "chart_reader"

SYSTEM = ("You are a technical analyst reading a candlestick chart for a rules-based trading system. Use only "
          "the bars and facts given. Name the dominant pattern if there is one, say whether it leans bullish, "
          "bearish or neutral over the next few bars, and give key levels. Neutral is a fine answer. "
          + INJECTION_GUARD)

SCHEMA_HINT = ('{"pattern":"short name","bias":"bullish|bearish|neutral","confidence":0.0-1.0,'
               '"support":number|null,"resistance":number|null,"invalidation":number|null,"note":"one sentence"}')

_BIAS = {"bullish": 1, "bearish": -1, "neutral": 0}


@dataclass
class ChartPlan:
    """A trade plan the chart read is a context check for (facts only; the agent cannot change it)."""
    symbol: str
    direction: int                      # +1 long, -1 short
    decision_id: str
    entry: float | None = None


@dataclass
class ChartAgentConfig:
    tf: str = "D1"
    bars_shown: int = 40
    min_bars: int = 60
    veto_confidence: float = 0.75
    shrink_confidence: float = 0.6
    flag_confidence: float = 0.65
    shrink_factor: float = 0.5
    veto_bars: int = 1
    max_tokens: int = 500
    category: str = AGENT


@dataclass
class ChartRead:
    symbol: str
    status: str                          # proposed | agree | neutral | invalid | skipped
    bar_open: pd.Timestamp | None = None
    pattern: str = ""
    bias: int = 0
    confidence: float = 0.0
    levels: dict[str, float | None] | None = None
    note: str = ""
    action: str | None = None
    decision_id: int | None = None


def knowable_bars(bars: pd.DataFrame, tf: str, asof: pd.Timestamp) -> pd.DataFrame:
    """Bars whose close time is at or before ``asof``."""
    asof = pd.Timestamp(asof)
    asof = asof.tz_localize("UTC") if asof.tzinfo is None else asof.tz_convert("UTC")
    return bars[bars.index + duration(tf) <= asof]


def chart_facts(b: pd.DataFrame) -> dict[str, Any]:
    """Deterministic facts from knowable bars only."""
    o, hi, lo, c = (b[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    atr = float(talib.ATR(hi, lo, c, 14)[-1])
    facts: dict[str, Any] = {"close": float(c[-1]), "atr14": atr}
    for n in (20, 50):
        facts[f"sma{n}"] = float(c[-n:].mean()) if len(c) >= n else None
    facts["range20_high"], facts["range20_low"] = float(hi[-20:].max()), float(lo[-20:].min())
    candles = []
    for name in talib.get_function_groups()["Pattern Recognition"]:
        v = int(getattr(talib, name)(o, hi, lo, c)[-1])
        if v:
            candles.append(f"{name[3:].lower()} ({'bullish' if v > 0 else 'bearish'})")
    facts["candles_last_bar"] = candles
    piv = zigzag_pivots(b, 2.0)
    facts["swings"] = [{"kind": "high" if int(r.kind) == 1 else "low", "price": round(float(r.price), 4),
                        "formed": b.index[int(r.at)].strftime("%Y-%m-%d %H:%M"),
                        "confirmed": b.index[int(r.confirmed_at)].strftime("%Y-%m-%d %H:%M")}
                       for r in piv.tail(4).itertuples()]
    dt = double_top_bottom(FeatureContext(b)).tail(5)
    facts["double_top_bottom_last5"] = ("bottom breakout" if (dt > 0).any() else
                                        "top breakdown" if (dt < 0).any() else "none")
    return facts


class ChartReader:
    def __init__(self, gateway: ModelGateway, store: ShadowStore, sink: InboxSink, cfg: ChartAgentConfig | None = None):
        self.gw, self.store, self.sink = gateway, store, sink
        self.cfg = cfg or ChartAgentConfig()

    def build_prompt(self, symbol: str, b: pd.DataFrame, plan: ChartPlan | None, f: dict[str, Any] | None = None) -> str:
        c = self.cfg
        f = f or chart_facts(b)
        rows = [f"{ts.strftime('%Y-%m-%d %H:%M')} o={r.open:.4f} h={r.high:.4f} l={r.low:.4f} c={r.close:.4f} v={r.volume:.0f}"
                for ts, r in b.tail(c.bars_shown).iterrows()]
        facts = [f"symbol {symbol}, timeframe {c.tf}, last closed bar opened {b.index[-1].isoformat()}",
                 f"close {f['close']:.4f}, ATR14 {f['atr14']:.4f}, SMA20 {f['sma20']}, SMA50 {f['sma50']}",
                 f"20-bar range {f['range20_low']:.4f} to {f['range20_high']:.4f}",
                 f"candlestick detectors on the last bar: {', '.join(f['candles_last_bar']) or 'none'}",
                 f"double top/bottom in the last 5 bars: {f['double_top_bottom_last5']}",
                 "confirmed swing points: " + ("; ".join(f"{s['kind']} {s['price']} (confirmed {s['confirmed']})"
                                                         for s in f["swings"]) or "none")]
        if plan is not None:
            facts.append(f"the system is considering a {'long' if plan.direction > 0 else 'short'} here"
                         + (f" at {plan.entry:g}" if plan.entry else ""))
        return (f"Read this chart.\nReply as JSON: {SCHEMA_HINT}\n\nFACTS (computed from the bars below):\n"
                + "\n".join(f"- {x}" for x in facts) + "\n\n" + data_block("BARS", rows))

    def read(self, asof: pd.Timestamp, symbol: str, bars: pd.DataFrame, plan: ChartPlan | None = None) -> ChartRead:
        c = self.cfg
        asof = pd.Timestamp(asof)
        asof = asof.tz_localize("UTC") if asof.tzinfo is None else asof.tz_convert("UTC")
        if not SYMBOL_RE.match(symbol):
            raise ValueError(f"bad symbol {symbol!r}")
        if plan is not None and (plan.symbol != symbol or not DECISION_ID_RE.match(plan.decision_id)):
            raise ValueError("plan does not match the symbol or has a malformed decision ID")
        b = knowable_bars(bars, c.tf, asof)
        if len(b) < c.min_bars:
            did = self.store.add_decision(AGENT, asof, "skipped", target=symbol, note=f"only {len(b)} closed bars")
            return ChartRead(symbol, "skipped", note="not enough closed bars", decision_id=did)
        facts = chart_facts(b)
        prompt = self.build_prompt(symbol, b, plan, facts)
        res, cid = ask(self.store, self.gw, AGENT, c.category, prompt, SYSTEM, asof, c.max_tokens)
        if not res.ok:
            did = self.store.add_decision(AGENT, asof, "skipped", target=symbol, call_ids=[cid],
                                          note=f"model call failed: {res.error}")
            return ChartRead(symbol, "skipped", b.index[-1], note=res.error, decision_id=did)
        reply = parse_json(res.text) or {}
        bias = _BIAS.get(str(reply.get("bias") or "").lower())
        conf = num(reply.get("confidence"), 0.0, 1.0)
        if bias is None or conf is None:
            did = self.store.add_decision(AGENT, asof, "invalid", target=symbol, call_ids=[cid],
                                          note="reply had no valid bias and confidence")
            return ChartRead(symbol, "invalid", b.index[-1], decision_id=did)
        last = float(b["close"].iloc[-1])
        levels = {k: self._level(reply.get(k), last) for k in ("support", "resistance", "invalidation")}
        pattern, note = clean(reply.get("pattern"), 60), clean(reply.get("note"), 300)
        atr = float(facts["atr14"])
        base = {"kind": "chart_read", "symbol": symbol, "tf": c.tf, "bar_open": b.index[-1].isoformat(),
                "asof": asof.isoformat(), "close": last, "atr": atr, "pattern": pattern, "bias": bias,
                "confidence": conf, "levels": levels, "note": note}
        read = ChartRead(symbol, "neutral", b.index[-1], pattern, bias, conf, levels, note)
        action, target, extra = self._decide(symbol, bias, conf, plan)
        if action is None:
            status = "agree" if plan is not None and bias == plan.direction else "neutral"
            read.status = status
            read.decision_id = self.store.add_decision(AGENT, asof, status, target=symbol, body=base,
                                                       call_ids=[cid], note=note)
            return read
        body = {**base, **extra, "expected_direction": bias,
                "reason": f"chart read ({pattern or 'no named pattern'}, {conf:.0%}): {note}"}
        if plan is not None:
            body["plan_direction"] = plan.direction
        read.decision_id = self.sink.propose(AGENT, asof, action, target, body, [cid], allowed_targets=[target],
                                             providers=[res.provider], models=[res.model])
        read.status, read.action = "proposed", action
        return read

    def _decide(self, symbol: str, bias: int, conf: float, plan: ChartPlan | None) -> tuple[str | None, str, dict]:
        c = self.cfg
        if plan is not None:
            if bias == -plan.direction and conf >= c.veto_confidence:
                return "veto", plan.decision_id, {"bars": c.veto_bars}
            if bias == -plan.direction and conf >= c.shrink_confidence:
                return "shrink", plan.decision_id, {"factor": c.shrink_factor}
            return None, symbol, {}
        if bias != 0 and conf >= c.flag_confidence:
            return "flag", symbol, {}
        return None, symbol, {}

    @staticmethod
    def _level(x: Any, last: float) -> float | None:
        v = num(x)
        return round(v, 4) if v is not None and 0.5 * last <= v <= 1.5 * last else None
