"""Position reviewer agent: two models from different providers must agree before a close proposal.

The rule-based ``PositionReviewer`` stays the owner of time limits, stale trades and
rollover. This agent adds a second look that can read news: for each open position it asks
two voters (different gateways or categories, ideally different providers) the same
question. A ``close`` proposal is written only when

1. both calls succeeded and parsed,
2. the providers that ACTUALLY answered (taken from the response, not the route name) are
   known and different,
3. both say close, each with confidence at least ``min_confidence``.

Anything else is logged in the shadow store (``agree_hold``, ``disagree``, ``no_quorum``)
and writes nothing to the inbox. A position with an unresolved proposal is not asked again.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import pandas as pd

from tradex.agents.common import (INJECTION_GUARD, InboxSink, ShadowStore, Voter, ask, clean, data_block,
                                  known_provider, num, parse_json)
from tradex.positions.review import Action, MarketSnapshot, OpenPosition, PositionReviewer
from tradex.scout.base import NewsFeed, NewsItem

AGENT = "position_reviewer"

SYSTEM = ("You are a second reviewer of one open position in a rules-based trading system. The position "
          "already has a stop, a target and a time limit that other code enforces. Decide only whether the "
          "position should be CLOSED NOW because the facts or news make the plan's premise no longer hold, or "
          "HELD. Default to hold. " + INJECTION_GUARD)

SCHEMA_HINT = '{"decision":"close|hold","confidence":0.0-1.0,"reason":"one sentence"}'


@dataclass
class PositionInput:
    pos: OpenPosition
    mkt: MarketSnapshot
    decision_id: str | None = None

    @property
    def target(self) -> str:
        return self.decision_id or self.pos.symbol


@dataclass
class Vote:
    label: str
    ok: bool
    decision: str | None = None        # close | hold
    confidence: float = 0.0
    reason: str = ""
    provider: str | None = None
    model: str | None = None
    call_id: int | None = None
    error: str = ""


@dataclass
class PositionReviewResult:
    target: str
    status: str                        # proposed | agree_hold | disagree | no_quorum | skipped | duplicate
    votes: list[Vote] = field(default_factory=list)
    decision_id: int | None = None
    note: str = ""


@dataclass
class PositionAgentConfig:
    min_confidence: float = 0.6
    news_lookback: pd.Timedelta = pd.Timedelta(hours=48)
    max_headlines: int = 6
    max_tokens: int = 600


class PositionReviewAgent:
    def __init__(self, voters: Sequence[Voter], store: ShadowStore, sink: InboxSink, feed: NewsFeed | None = None,
                 rules: PositionReviewer | None = None, cfg: PositionAgentConfig | None = None):
        if len(voters) < 2:
            raise ValueError("a close proposal needs at least two voters")
        if len({v.label for v in voters}) != len(voters):
            raise ValueError("voter labels must be unique")
        self.voters, self.store, self.sink, self.feed = list(voters), store, sink, feed
        self.rules = rules or PositionReviewer()
        self.cfg = cfg or PositionAgentConfig()

    # --- prompt --------------------------------------------------------------------------

    def _news(self, p: PositionInput) -> list[NewsItem]:
        if self.feed is None:
            return []
        t = p.mkt.time
        try:
            items = self.feed.fetch(t - self.cfg.news_lookback, t, [p.pos.symbol])
        except Exception:  # noqa: BLE001 - no news is a normal case for a reviewer
            return []
        items = [i for i in items if i.symbol == p.pos.symbol and t - self.cfg.news_lookback <= i.published <= t]
        return sorted(items, key=lambda i: i.published, reverse=True)[: self.cfg.max_headlines]

    def build_prompt(self, p: PositionInput, rule_actions: list[Action]) -> str:
        pos, m = p.pos, p.mkt
        side = "long" if pos.direction > 0 else "short"
        facts = [
            f"symbol {pos.symbol} ({pos.asset_class}), {side}, strategy {pos.strategy_id}",
            f"entry {pos.entry_price:g} at {pos.entry_time.isoformat()}, now {m.price:g} at {m.time.isoformat()}",
            f"stop {pos.stop:g}, target {pos.target:g}, initial risk {pos.initial_risk:g}",
            f"result now {pos.r_multiple(m.price):+.2f}R, held {pos.bars_held} of {pos.max_bars} bars, ATR {m.atr:g}",
            "rule-based reviewer says: " + "; ".join(f"{a.kind.value} ({a.reason})" for a in rule_actions),
        ]
        if m.next_event_time is not None:
            facts.append(f"next scheduled event: {m.next_event_kind or 'event'} at {m.next_event_time.isoformat()}")
        news = []
        for n, it in enumerate(self._news(p), 1):
            news.append(f"N{n} {it.published.strftime('%m-%d %H:%MZ')} [{clean(it.source, 40)}] {clean(it.headline)}")
        return (f"Review this open position.\nReply as JSON: {SCHEMA_HINT}\n\nFACTS (from the trading system):\n"
                  + "\n".join(f"- {f}" for f in facts) + "\n\n"
                  + data_block("HEADLINES", news or ["(none in the last "
                                                    f"{int(self.cfg.news_lookback.total_seconds() // 3600)}h)"]))

    # --- run -----------------------------------------------------------------------------

    def review(self, positions: Sequence[PositionInput]) -> list[PositionReviewResult]:
        return [self.review_one(p) for p in positions]

    def review_one(self, p: PositionInput) -> PositionReviewResult:
        asof, target = p.mkt.time, p.target
        if self.store.has_open_proposal(AGENT, target):
            return PositionReviewResult(target, "duplicate", note="an earlier close proposal is still unresolved")
        prompt = self.build_prompt(p, self.rules.review(p.pos, p.mkt))
        votes: list[Vote] = []
        for v in self.voters:
            res, cid = ask(self.store, v.gateway, AGENT, v.category, prompt, SYSTEM, asof, self.cfg.max_tokens, v.label)
            vote = Vote(v.label, res.ok, provider=res.provider, model=res.model, call_id=cid, error=res.error)
            if res.ok:
                reply = parse_json(res.text) or {}
                dec = str(reply.get("decision") or "").lower()
                conf = num(reply.get("confidence"), 0.0, 1.0)
                if dec in ("close", "hold") and conf is not None:
                    vote.decision, vote.confidence = dec, conf
                    vote.reason = clean(reply.get("reason"), 300)
                else:
                    vote.ok, vote.error = False, "unparseable decision"
            votes.append(vote)
        call_ids = [v.call_id for v in votes if v.call_id is not None]
        status, note = self.quorum(votes)
        if status != "proposed":
            did = self.store.add_decision(AGENT, asof, status, target=target, body=self._body(p, votes),
                                          call_ids=call_ids, note=note)
            return PositionReviewResult(target, status, votes, did, note)
        body = self._body(p, votes)
        body["reason"] = " | ".join(f"{v.label} ({v.provider}): {v.reason}" for v in votes)
        body["fraction"] = 1.0
        did = self.sink.propose(AGENT, asof, "close", target, body, call_ids, allowed_targets=[target],
                                providers=[v.provider for v in votes], models=[v.model for v in votes])
        return PositionReviewResult(target, "proposed", votes, did)

    def quorum(self, votes: list[Vote]) -> tuple[str, str]:
        if not all(v.ok for v in votes):
            return "no_quorum", "a voter failed: " + "; ".join(f"{v.label}: {v.error}" for v in votes if not v.ok)
        provs = [v.provider for v in votes]
        if not all(known_provider(x) for x in provs):
            return "no_quorum", f"provider not known for every voter: {provs}"
        if len(set(provs)) != len(provs):
            return "no_quorum", f"voters answered from the same provider: {provs}"
        if all(v.decision == "hold" for v in votes):
            return "agree_hold", ""
        if all(v.decision == "close" for v in votes):
            if min(v.confidence for v in votes) < self.cfg.min_confidence:
                return "disagree", f"both said close but confidence below {self.cfg.min_confidence}"
            return "proposed", ""
        return "disagree", "voters split"

    def _body(self, p: PositionInput, votes: list[Vote]) -> dict[str, Any]:
        pos, m = p.pos, p.mkt
        return {"kind": "position_close", "symbol": pos.symbol, "direction": pos.direction,
                "entry_price": pos.entry_price, "entry_time": pos.entry_time.isoformat(), "stop": pos.stop,
                "target": pos.target, "initial_risk": pos.initial_risk, "max_bars": pos.max_bars,
                "bars_held": pos.bars_held, "price": m.price, "r_now": pos.r_multiple(m.price),
                "asof": m.time.isoformat(),
                "votes": [{"label": v.label, "decision": v.decision, "confidence": v.confidence,
                           "reason": v.reason, "provider": v.provider, "model": v.model, "ok": v.ok,
                           "error": v.error} for v in votes]}
