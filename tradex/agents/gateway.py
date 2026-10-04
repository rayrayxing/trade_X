"""Model gateway: one place every agent call goes through.

Default route for every category is EasyCLIProxyAPI (OpenAI-compatible); a category can
be switched to a direct anthropic / openai / google key in ``config/models.yaml``, with
ordered fallbacks. Each call writes a row to ``agent_calls`` with the provider and model
that ACTUALLY answered (taken from the response body and headers, not the route name),
latency, tokens and hashes of prompt and response (never the text or any key).

Agents are never on the critical path: ``call`` never raises and never blocks past the
time budget; on failure the caller gets ``ok=False`` and the skip is logged. ``submit``
runs the call on a thread so the core can keep trading.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time as _time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

from tradex.core.inbox import Mailbox

CATEGORIES = ("scout_analyst", "position_reviewer", "chart_reader", "researcher", "red_team",
              "incident_analyst", "strategy_designer", "critic")
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "models.yaml"
PROVIDER_HEADERS = ("x-upstream-provider", "x-provider", "x-llm-provider")


@dataclass
class HttpResponse:
    status: int
    body: dict[str, Any]
    headers: dict[str, str] = field(default_factory=dict)


Http = Callable[[str, dict[str, str], dict[str, Any], float], HttpResponse]


def requests_http(url: str, headers: dict[str, str], body: dict[str, Any], timeout: float) -> HttpResponse:
    import requests
    r = requests.post(url, headers=headers, json=body, timeout=timeout)
    try:
        data = r.json()
    except ValueError:
        data = {}
    return HttpResponse(r.status_code, data if isinstance(data, dict) else {}, {k.lower(): v for k, v in r.headers.items()})


@dataclass
class AgentResult:
    ok: bool
    category: str
    text: str = ""
    provider: str | None = None        # who actually answered
    model: str | None = None
    route: str = ""
    latency_ms: int = 0
    tokens_in: int | None = None
    tokens_out: int | None = None
    error: str = ""
    attempts: list[dict[str, Any]] = field(default_factory=list)


def h(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def infer_provider(model: str | None, headers: dict[str, str], route: str) -> str:
    for k in PROVIDER_HEADERS:
        if headers.get(k):
            return headers[k].lower()
    m = (model or "").lower()
    for prefix, name in (("claude", "anthropic"), ("gpt", "openai"), ("o1", "openai"), ("o3", "openai"),
                         ("o4", "openai"), ("gemini", "google")):
        if m.startswith(prefix) or f"/{prefix}" in m:
            return name
    return route if route != "proxy" else "unknown(proxy)"  # a direct route answers as itself


class GatewayError(Exception):
    pass


def _build(kind: str, base: str, key: str, model: str, system: str, prompt: str, max_tokens: int):
    if kind == "openai_compat":
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        return (base.rstrip("/") + "/chat/completions", {"authorization": f"Bearer {key}"},
                {"model": model, "messages": msgs, "max_tokens": max_tokens})
    if kind == "anthropic":
        body: dict[str, Any] = {"model": model, "max_tokens": max_tokens,
                                "messages": [{"role": "user", "content": prompt}]}
        if system:
            body["system"] = system
        return (base.rstrip("/") + "/v1/messages", {"x-api-key": key, "anthropic-version": "2023-06-01"}, body)
    if kind == "google":
        body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": max_tokens}}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        return (f"{base.rstrip('/')}/v1beta/models/{model}:generateContent", {"x-goog-api-key": key}, body)
    raise GatewayError(f"unknown route kind {kind}")


def _parse(kind: str, body: dict[str, Any]) -> tuple[str, str | None, int | None, int | None]:
    """text, answering model, tokens in, tokens out."""
    if kind == "openai_compat":
        ch = (body.get("choices") or [{}])[0]
        u = body.get("usage") or {}
        return (ch.get("message") or {}).get("content") or "", body.get("model"), u.get("prompt_tokens"), u.get("completion_tokens")
    if kind == "anthropic":
        u = body.get("usage") or {}
        text = "".join(b.get("text", "") for b in body.get("content") or [] if b.get("type") == "text")
        return text, body.get("model"), u.get("input_tokens"), u.get("output_tokens")
    cand = (body.get("candidates") or [{}])[0]
    text = "".join(p.get("text", "") for p in (cand.get("content") or {}).get("parts") or [])
    u = body.get("usageMetadata") or {}
    return text, body.get("modelVersion"), u.get("promptTokenCount"), u.get("candidatesTokenCount")


class Gateway:
    def __init__(self, mailbox: Mailbox | None = None, config: dict[str, Any] | str | Path | None = None,
                 http: Http | None = None, secret: Callable[[str], str] | None = None,
                 now: Callable[[], datetime] | None = None, clock: Callable[[], float] = _time.monotonic):
        if not isinstance(config, dict):
            config = yaml.safe_load(Path(config or DEFAULT_CONFIG).read_text())
        self.cfg = config
        self.mb = mailbox
        self.http = http or requests_http
        self._secret = secret
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.clock = clock
        self.timeout = float(config.get("timeout_s", 20))
        self.budget = float(config.get("total_budget_s", 2 * self.timeout))

    def secret(self, name: str) -> str:
        if self._secret is not None:
            return self._secret(name)
        from tradex import secrets  # lazy
        return secrets.get(name)

    def plan(self, category: str) -> list[tuple[str, str]]:
        c = self.cfg["categories"].get(category)
        if c is None:
            raise GatewayError(f"unknown category {category}")
        return [(c["route"], c["model"])] + [(f["route"], f["model"]) for f in c.get("fallbacks", [])]

    def _attempt(self, route: str, model: str, system: str, prompt: str, max_tokens: int, timeout: float):
        r = self.cfg["routes"][route]
        try:
            key = self.secret(r["key_secret"])
            base = self.secret(r["base_url_secret"]) if "base_url_secret" in r else r["base_url"]
        except Exception:  # noqa: BLE001 - missing secret: skip this route
            raise GatewayError("secret missing") from None
        url, hdrs, body = _build(r["kind"], base, key, model, system, prompt, max_tokens)
        resp = self.http(url, {"content-type": "application/json", **hdrs}, body, timeout)
        if resp.status >= 400:
            raise GatewayError(f"http {resp.status}")
        text, answered, tin, tout = _parse(r["kind"], resp.body)
        if not text:
            raise GatewayError("empty response")
        return text, answered or model, infer_provider(answered, resp.headers, route), tin, tout

    def call(self, category: str, prompt: str, system: str = "", max_tokens: int = 1024) -> AgentResult:
        """Never raises. On failure returns ok=False and logs the skip."""
        start = self.clock()
        res = AgentResult(False, category)
        try:
            plan = self.plan(category)
        except GatewayError as exc:
            res.error = str(exc)
            self._log(res, prompt)
            return res
        for route, model in plan:
            left = self.budget - (self.clock() - start)
            if left <= 0:
                res.attempts.append({"route": route, "model": model, "ok": False, "error": "budget exhausted"})
                break
            t0 = self.clock()
            try:
                text, answered, prov, tin, tout = self._attempt(route, model, system, prompt, max_tokens,
                                                                min(self.timeout, left))
                ms = int((self.clock() - t0) * 1000)
                res.attempts.append({"route": route, "model": model, "ok": True, "ms": ms})
                res.ok, res.text, res.route, res.model, res.provider = True, text, route, answered, prov
                res.tokens_in, res.tokens_out, res.latency_ms = tin, tout, ms
                break
            except Exception as exc:  # noqa: BLE001 - timeouts, 5xx, bad JSON: try the next route
                err = f"{type(exc).__name__}: {exc}" if isinstance(exc, GatewayError) else type(exc).__name__
                res.attempts.append({"route": route, "model": model, "ok": False, "error": err,
                                     "ms": int((self.clock() - t0) * 1000)})
                res.route = route
        if not res.ok:
            res.error = "skipped: " + "; ".join(f"{a['route']} {a.get('error')}" for a in res.attempts)
            res.latency_ms = int((self.clock() - start) * 1000)
        self._log(res, prompt + system, res.text)
        return res

    def submit(self, category: str, prompt: str, on_done: Callable[[AgentResult], None] | None = None,
               **kw: Any) -> threading.Thread:
        """Fire and forget: the core keeps trading; ``on_done`` (if any) gets the result."""
        def work() -> None:
            r = self.call(category, prompt, **kw)
            if on_done is not None:
                try:
                    on_done(r)
                except Exception:  # noqa: BLE001
                    pass
        t = threading.Thread(target=work, daemon=True)
        t.start()
        return t

    def _log(self, r: AgentResult, prompt: str, response: str = "") -> None:
        if self.mb is None:
            return
        with self.mb._lock, self.mb.db:
            self.mb.db.execute(
                "INSERT INTO agent_calls (time, category, route, provider, model, ok, latency_ms, tokens_in,"
                " tokens_out, prompt_hash, response_hash, error, attempts) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (self.now().isoformat(), r.category, r.route, r.provider, r.model, int(r.ok), r.latency_ms,
                 r.tokens_in, r.tokens_out, h(prompt), h(response) if response else None, r.error or None,
                 json.dumps(r.attempts)))
