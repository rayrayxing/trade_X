"""Daily scout agent: news and chatter in, a 10 to 20 stock watchlist with justifications out.

The model only ranks and explains what the feed delivered. Every pick must cite headlines
that were actually in the prompt and published before ``asof``; picks naming a symbol that
was not offered, citing a headline that does not exist, or giving no reason are dropped (and
counted), so the watchlist cannot contain a name the news did not support. If fewer than
``min_items`` valid picks survive, the list is written short and marked
``meets_minimum=False``; it is never padded.

Shadow output: one ``flag`` row in the agent inbox, target ``watchlist:YYYY-MM-DD``, whose
body carries the picks. The scorecard later compares the picks with what the stocks did.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from tradex.agents.common import (INJECTION_GUARD, InboxSink, ModelGateway, ShadowStore, ask, clean, data_block, num,
                                  parse_json)
from tradex.scout.base import NewsFeed, NewsItem, ScoutSource, WatchItem, Watchlist

AGENT = "scout_analyst"

SYSTEM = ("You are the daily market scout for a rules-based US stock trading system. From the numbered "
          "headlines choose the stocks most worth watching today, with direction long, short or watch, and "
          "justify each with the headline IDs that support it. " + INJECTION_GUARD)

SCHEMA_HINT = ('{"picks":[{"symbol":"TICKER","direction":"long|short|watch","score":0.0-3.0,'
               '"justification":"one sentence","evidence":["N1","N4"]}]}')


@dataclass
class ScoutAgentConfig:
    min_items: int = 10
    max_items: int = 20
    lookback: pd.Timedelta = pd.Timedelta(hours=24)
    max_headlines_per_symbol: int = 4
    max_symbols_in_prompt: int = 60
    max_tokens: int = 2000
    category: str = AGENT


@dataclass
class ScoutPick:
    symbol: str
    direction: int
    score: float
    justification: str
    evidence: list[dict[str, Any]]


@dataclass
class ScoutRun:
    asof: pd.Timestamp
    watchlist: Watchlist
    status: str                                   # proposed | skipped | invalid | none
    meets_minimum: bool = False
    dropped: dict[str, int] = field(default_factory=dict)
    decision_id: int | None = None
    feed_errors: list[str] = field(default_factory=list)


class ScoutAgent:
    def __init__(self, gateway: ModelGateway, feed: NewsFeed, store: ShadowStore, sink: InboxSink,
                 universe: list[str], cfg: ScoutAgentConfig | None = None, technical: ScoutSource | None = None):
        self.gw, self.feed, self.store, self.sink = gateway, feed, store, sink
        self.universe = [s for s in dict.fromkeys(universe)]
        self.cfg = cfg or ScoutAgentConfig()
        self.technical = technical

    # --- inputs --------------------------------------------------------------------------

    def _headlines(self, asof: pd.Timestamp) -> dict[str, list[NewsItem]]:
        start = asof - self.cfg.lookback
        items = self.feed.fetch(start, asof, self.universe)
        uni = set(self.universe)
        by: dict[str, list[NewsItem]] = {}
        for it in items:
            if it.symbol in uni and start <= it.published <= asof:   # never trust the feed's own filtering
                by.setdefault(it.symbol, []).append(it)
        return by

    def build_prompt(self, asof: pd.Timestamp, by: dict[str, list[NewsItem]]) -> tuple[str, dict[str, NewsItem], list[str]]:
        c = self.cfg
        ranked = sorted(by, key=lambda s: (-len(by[s]), s))[: c.max_symbols_in_prompt]
        tech = {}
        if self.technical is not None:
            try:
                tech = self.technical.score(asof, ranked)
            except Exception:  # noqa: BLE001 - an optional context source must not stop the scout
                tech = {}
        ids: dict[str, NewsItem] = {}
        lines: list[str] = []
        n = 0
        for sym in ranked:
            lines.append(f"{sym}" + (f"  [technical: {clean(tech[sym].reason, 120)}]" if sym in tech else ""))
            for it in sorted(by[sym], key=lambda i: i.published, reverse=True)[: c.max_headlines_per_symbol]:
                n += 1
                nid = f"N{n}"
                ids[nid] = it
                lines.append(f"  {nid} {it.published.strftime('%m-%d %H:%MZ')} [{clean(it.source, 40)}] {clean(it.headline)}")
        prompt = (f"As of {asof.isoformat()}. Pick between {c.min_items} and {c.max_items} symbols from this list "
                  f"(fewer if the news does not support that many; never add symbols).\n"
                  f"Reply as JSON: {SCHEMA_HINT}\n\n" + data_block("HEADLINES", lines))
        return prompt, ids, ranked

    # --- run -----------------------------------------------------------------------------

    def run(self, asof: pd.Timestamp) -> ScoutRun:
        asof = pd.Timestamp(asof)
        asof = asof.tz_localize("UTC") if asof.tzinfo is None else asof.tz_convert("UTC")
        empty = Watchlist(asof=str(asof), items=[])
        errs = list(getattr(self.feed, "errors", []) or [])
        try:
            by = self._headlines(asof)
        except Exception as exc:  # noqa: BLE001 - a dead feed means no watchlist today, not a crash
            did = self.store.add_decision(AGENT, asof, "skipped", note=f"news feed failed: {type(exc).__name__}: {exc}")
            return ScoutRun(asof, empty, "skipped", decision_id=did, feed_errors=errs + [str(exc)])
        if not by:
            did = self.store.add_decision(AGENT, asof, "skipped", note="no news in the lookback window")
            return ScoutRun(asof, empty, "skipped", decision_id=did, feed_errors=errs)
        prompt, ids, offered = self.build_prompt(asof, by)
        res, cid = ask(self.store, self.gw, AGENT, self.cfg.category, prompt, SYSTEM, asof, self.cfg.max_tokens)
        if not res.ok:
            did = self.store.add_decision(AGENT, asof, "skipped", call_ids=[cid], note=f"model call failed: {res.error}")
            return ScoutRun(asof, empty, "skipped", decision_id=did, feed_errors=errs)
        picks, dropped = self.validate(parse_json(res.text), ids, offered)
        if not picks:
            did = self.store.add_decision(AGENT, asof, "invalid", call_ids=[cid], body={"dropped": dropped},
                                          note="no valid picks in the reply")
            return ScoutRun(asof, empty, "invalid", dropped=dropped, decision_id=did, feed_errors=errs)
        meets = len(picks) >= self.cfg.min_items
        wl = Watchlist(asof=str(asof), items=[
            WatchItem(p.symbol, p.score, p.direction, [p.justification], [AGENT]) for p in picks])
        body = {"kind": "watchlist", "asof": str(asof), "meets_minimum": meets, "dropped": dropped,
                "reason": f"daily watchlist, {len(picks)} names",
                "picks": [{"symbol": p.symbol, "direction": p.direction, "score": p.score,
                           "justification": p.justification, "evidence": p.evidence} for p in picks]}
        did = self.sink.propose(AGENT, asof, "flag", f"watchlist:{asof.strftime('%Y-%m-%d')}", body, [cid],
                                providers=[res.provider], models=[res.model])
        return ScoutRun(asof, wl, "proposed", meets, dropped, did, errs)

    def validate(self, reply: dict | None, ids: dict[str, NewsItem], offered: list[str]) -> tuple[list[ScoutPick], dict[str, int]]:
        dropped: dict[str, int] = {}

        def drop(why: str) -> None:
            dropped[why] = dropped.get(why, 0) + 1

        raw = (reply or {}).get("picks")
        if not isinstance(raw, list):
            return [], {"malformed_reply": 1}
        out: dict[str, ScoutPick] = {}
        for p in raw:
            if not isinstance(p, dict):
                drop("malformed_pick")
                continue
            sym = str(p.get("symbol") or "").strip().upper()
            if sym not in offered:
                drop("symbol_not_offered")
                continue
            if sym in out:
                drop("duplicate")
                continue
            why = clean(p.get("justification"), 300)
            if not why:
                drop("no_justification")
                continue
            ev_ids = p.get("evidence")
            if not isinstance(ev_ids, list):
                drop("no_evidence")
                continue
            ev = []
            for e in ev_ids:
                it = ids.get(str(e))
                if it is not None and it.symbol == sym and not any(x["id"] == str(e) for x in ev):
                    ev.append({"id": str(e), "headline": it.headline, "source": it.source, "url": it.url,
                               "published": it.published.isoformat()})
            if not ev:
                drop("evidence_not_in_prompt")
                continue
            score = num(p.get("score"), 0.0, 3.0)
            if score is None:
                drop("bad_score")
                continue
            direction = {"long": 1, "short": -1}.get(str(p.get("direction") or "watch").lower(), 0)
            out[sym] = ScoutPick(sym, direction, round(score, 3), why, ev)
        picks = sorted(out.values(), key=lambda x: (-x.score, x.symbol))[: self.cfg.max_items]
        return picks, dropped
