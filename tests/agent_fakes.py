"""Test doubles shared by the agent tests: a scripted fake gateway and a stub news feed."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pandas as pd

from tradex.agents.gateway import AgentResult
from tradex.scout.base import NewsItem

FIXTURES = Path(__file__).parent / "fixtures" / "agents"


def fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


class FakeGateway:
    """Answers every call from ``reply(category, prompt, system) -> str | dict | AgentResult``."""

    def __init__(self, reply: Callable[[str, str, str], object] | object, provider: str | None = "anthropic",
                 model: str = "claude-sonnet-5-5"):
        self.reply, self.provider, self.model = reply, provider, model
        self.calls: list[tuple[str, str, str]] = []

    def call(self, category: str, prompt: str, system: str = "", max_tokens: int = 1024) -> AgentResult:
        self.calls.append((category, prompt, system))
        out = self.reply(category, prompt, system) if callable(self.reply) else self.reply
        if isinstance(out, AgentResult):
            return out
        text = out if isinstance(out, str) else json.dumps(out)
        return AgentResult(True, category, text=text, provider=self.provider, model=self.model, route="proxy",
                           latency_ms=5, tokens_in=10, tokens_out=5)


def failed(category: str = "x", error: str = "skipped: proxy http 500") -> AgentResult:
    return AgentResult(False, category, error=error)


class StubFeed:
    """A NewsFeed returning fixed items (it does NOT filter by the window, to test the agent's own guard)."""

    def __init__(self, items: list[NewsItem], raises: Exception | None = None):
        self.items, self.raises, self.calls = items, raises, []

    def fetch(self, start, end, symbols):
        self.calls.append((start, end, symbols))
        if self.raises:
            raise self.raises
        return list(self.items)


def item(symbol: str, when: str, headline: str, source: str = "test:wire") -> NewsItem:
    return NewsItem(symbol, pd.Timestamp(when, tz="UTC"), headline, source, f"https://example.test/{symbol}")
