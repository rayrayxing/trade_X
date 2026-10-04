import json

import pytest
import yaml

from tradex.agents.gateway import CATEGORIES, DEFAULT_CONFIG, Gateway, HttpResponse, h
from tradex.core.inbox import Mailbox
from tradex.core.ledger import Ledger

SECRETS = {"proxy_base_url": "http://proxy.local/v1", "proxy_api_key": "pk", "anthropic_api_key": "ak",
           "openai_api_key": "ok", "google_api_key": "gk"}


def secret(n):
    return SECRETS[n]


def oa(model, text="hi", headers=None):
    return HttpResponse(200, {"model": model, "choices": [{"message": {"content": text}}],
                              "usage": {"prompt_tokens": 7, "completion_tokens": 3}}, headers or {})


def an(model, text="hello"):
    return HttpResponse(200, {"model": model, "content": [{"type": "text", "text": text}],
                              "usage": {"input_tokens": 5, "output_tokens": 2}})


class Http:
    def __init__(self, handlers):
        self.h, self.calls = handlers, []

    def __call__(self, url, headers, body, timeout):
        self.calls.append((url, headers, body, timeout))
        for prefix, fn in self.h.items():
            if url.startswith(prefix):
                return fn(body)
        raise AssertionError(url)


def gw(tmp_path, http):
    Ledger(tmp_path / "l.db")
    mb = Mailbox(tmp_path / "l.db")
    return Gateway(mb, http=http, secret=secret), mb


def rows(mb):
    return mb.read("SELECT * FROM agent_calls ORDER BY id")


def test_default_config_all_categories_on_proxy():
    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text())
    assert set(cfg["categories"]) == set(CATEGORIES)
    assert all(c["route"] == "proxy" for c in cfg["categories"].values())
    assert cfg["timeout_s"] == 20


def test_records_actual_model_not_route(tmp_path):
    # asked for opus via the proxy, proxy answered with a different model/provider
    http = Http({"http://proxy.local": lambda b: oa("gemini-x", "ok", {"x-upstream-provider": "google"})})
    g, mb = gw(tmp_path, http)
    r = g.call("red_team", "the prompt", system="sys")
    assert r.ok and r.route == "proxy" and r.model == "gemini-x" and r.provider == "google"
    row = rows(mb)[0]
    assert (row["provider"], row["model"], row["route"]) == ("google", "gemini-x", "proxy")
    assert row["tokens_in"] == 7 and row["tokens_out"] == 3 and row["ok"] == 1
    assert row["prompt_hash"] == h("the promptsys") and row["response_hash"] == h("ok")
    assert "pk" not in json.dumps([dict(x) for x in rows(mb)])
    assert http.calls[0][0] == "http://proxy.local/v1/chat/completions" and http.calls[0][3] == 20


def test_primary_killed_fallback_answers(tmp_path):
    def dead(b):
        raise ConnectionError("proxy down")
    http = Http({"http://proxy.local": dead, "https://api.anthropic.com": lambda b: an("claude-opus-5-5-20261001")})
    g, mb = gw(tmp_path, http)
    r = g.call("critic", "p")
    assert r.ok and r.route == "anthropic" and r.provider == "anthropic" and r.model == "claude-opus-5-5-20261001"
    row = rows(mb)[0]
    assert row["route"] == "anthropic" and row["ok"] == 1
    att = json.loads(row["attempts"])
    assert [a["ok"] for a in att] == [False, True] and att[0]["route"] == "proxy"
    assert http.calls[1][1]["x-api-key"] == "ak"


def test_all_fail_is_skipped_and_logged_not_raised(tmp_path):
    def boom(b):
        raise TimeoutError()
    g, mb = gw(tmp_path, Http({"http://proxy.local": boom, "https://api.anthropic.com": boom}))
    r = g.call("scout_analyst", "p")
    assert not r.ok and r.error.startswith("skipped")
    row = rows(mb)[0]
    assert row["ok"] == 0 and "TimeoutError" in row["error"]


def test_http_error_and_missing_secret_fall_through(tmp_path):
    http = Http({"https://api.anthropic.com": lambda b: an("claude-sonnet-5-5")})
    s = dict(SECRETS)
    del s["proxy_api_key"]
    Ledger(tmp_path / "l.db")
    g = Gateway(Mailbox(tmp_path / "l.db"), http=http, secret=lambda n: s[n])
    r = g.call("researcher", "p")
    assert r.ok and r.route == "anthropic"
    http2 = Http({"http://proxy.local": lambda b: HttpResponse(500, {}), "https://api.anthropic.com": lambda b: an("m")})
    g2 = Gateway(None, http=http2, secret=secret)
    assert g2.call("researcher", "p").route == "anthropic"


def test_budget_stops_chain(tmp_path):
    t = [0.0]

    def slow(b):
        t[0] += 50
        raise TimeoutError()
    http = Http({"http://proxy.local": slow, "https://api.anthropic.com": lambda b: an("m")})
    g = Gateway(None, http=http, secret=secret, clock=lambda: t[0])
    r = g.call("critic", "p")
    assert not r.ok and "budget" in r.error


def test_per_category_switch_to_direct(tmp_path):
    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text())
    cfg["categories"]["red_team"] = {"route": "openai", "model": "m1", "fallbacks": [{"route": "google", "model": "g1"}]}
    http = Http({"https://api.openai.com": lambda b: HttpResponse(503, {}),
                 "https://generativelanguage.googleapis.com": lambda b: HttpResponse(
                     200, {"modelVersion": "g1-001", "candidates": [{"content": {"parts": [{"text": "yo"}]}}]})})
    r = Gateway(None, cfg, http=http, secret=secret).call("red_team", "p")
    assert r.ok and r.provider == "google" and r.model == "g1-001"
    assert [c[0].split("/")[2] for c in http.calls] == ["api.openai.com", "generativelanguage.googleapis.com"]


def test_submit_does_not_block_and_unknown_category(tmp_path):
    g = Gateway(None, http=Http({"http://proxy.local": lambda b: oa("claude-sonnet-5-5")}), secret=secret)
    got = []
    g.submit("critic", "p", on_done=got.append).join(2)
    assert got and got[0].ok
    assert not g.call("nope", "p").ok
