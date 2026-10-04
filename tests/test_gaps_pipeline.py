"""Adversarial tests for the pipeline gaps of the 4 Oct 2026 review: G6 to G11 and S5.

``known_gap`` tests reproduce a gap still open on the Phase 1 tip (strict xfail, see
``gapkit``); the rest pin behaviour that already holds.
"""
import json
import sqlite3
import sys

import pandas as pd
import pytest
import yaml

from gapkit import T0, known_gap, plan, raised, verdict
from conftest import simple_spec
from tradex.agents.gateway import DEFAULT_CONFIG, Gateway, HttpResponse, infer_provider
from tradex.backtest.validation import WalkForwardConfig, walk_forward
from tradex.core.inbox import Mailbox, ingest_inbox
from tradex.core.interfaces import ReplayClock
from tradex.core.ledger import Ledger
from tradex.core.loop import CoreConfig
from tradex.core.records import Health, Order, Veto
from tradex.core.replay import build_replay_core
from tradex.data.synthetic import synthetic_bars
from tradex.decision.ensemble import DEFAULT_HIT_RATE
from tradex.notify.telegram import TelegramService
from tradex.research.trials import TrialLedger
from tradex.strategy.spec import StrategySpec


# --- G6: the moomoo import broke CI ---------------------------------------------------------------

@known_gap("G6", "opend_check.collect imports RET_OK from moomoo at call time, so it fails wherever the SDK is not installed (CI)")
def test_g6_collect_needs_no_moomoo_sdk(monkeypatch):
    from test_opend_check import Q, T
    from tradex.data.opend_check import collect
    monkeypatch.setitem(sys.modules, "moomoo", None)          # None in sys.modules makes `import moomoo` raise ImportError
    assert not raised(ImportError, collect, Q(), T())


# --- G7: the 0.40 default win rate -----------------------------------------------------------------

def _paper_core(spec, bars):
    led = Ledger(":memory:", git_commit="t")
    return build_replay_core([spec], {"X": bars}, led, None, cfg=CoreConfig(mode="paper", simulation=False),
                             use_es=False), bars


def _votes(spec):
    bars = synthetic_bars(120, seed=2)
    core, bars = _paper_core(spec, bars)
    close_t = bars.index[-1] + pd.Timedelta(days=1)
    core.clock.set(close_t)
    try:
        return core._votes("X", spec.signal_tf, close_t)
    except Exception:  # noqa: BLE001 - blocking the vote with an error is also acceptable
        return []


def test_a_measured_hit_rate_is_used_as_the_votes_strength():
    spec = simple_spec(long="close > 0")
    spec.status = "paper"
    spec.stats = {"hit_rate": 0.57}
    votes = _votes(spec)
    assert votes and votes[0].strength == pytest.approx(0.57)


@known_gap("G7", "loop._votes falls back to DEFAULT_HIT_RATE (0.40) in paper/live for a strategy with no measured hit rate")
@pytest.mark.parametrize("stats", [{}, {"sharpe": 1.1}])
def test_g7_paper_never_votes_with_the_default_hit_rate(stats):
    spec = simple_spec(long="close > 0")
    spec.status = "paper"
    spec.stats = stats
    assert not [v for v in _votes(spec) if v.strength == DEFAULT_HIT_RATE]


# --- G8: the model gateway ---------------------------------------------------------------------------

SECRETS = {"proxy_base_url": "http://proxy.local/v1", "proxy_api_key": "pk", "anthropic_api_key": "ak"}


def _ok_http(url, headers, body, timeout):
    return HttpResponse(200, {"model": "claude-sonnet-5-5", "choices": [{"message": {"content": "analysis"}}],
                              "usage": {"prompt_tokens": 9, "completion_tokens": 4}}, {})


def _gateway(tmp_path):
    Ledger(tmp_path / "l.db")
    mb = Mailbox(tmp_path / "l.db")
    return Gateway(mb, http=_ok_http, secret=SECRETS.__getitem__), mb


@known_gap("G8", "config/models.yaml gives a 45 s total budget; the review asks for 20 s")
def test_g8_total_budget_is_at_most_20_seconds():
    assert yaml.safe_load(DEFAULT_CONFIG.read_text())["total_budget_s"] <= 20


@known_gap("G8", "Gateway.call logs after answering; a logging failure escapes and kills the submit() worker thread")
def test_g8_call_never_raises_even_when_the_call_log_cannot_be_written(tmp_path):
    g, mb = _gateway(tmp_path)
    mb.close()
    assert not raised(Exception, g.call, "critic", "check this plan")


def test_gateway_budget_stops_a_slow_chain_of_fallbacks(tmp_path):
    clock = {"t": 0.0}

    def slow(url, headers, body, timeout):
        clock["t"] += 19.0
        raise OSError("timeout")

    Ledger(tmp_path / "l.db")
    g = Gateway(Mailbox(tmp_path / "l.db"), http=slow, secret=SECRETS.__getitem__, clock=lambda: clock["t"])
    res = g.call("critic", "x")
    assert not res.ok and clock["t"] <= g.budget + 19.0


@known_gap("G8", "only hashes of the prompt and response are stored, so an agent call can never be replayed")
def test_g8_the_prompt_text_is_stored_for_replay(tmp_path):
    g, mb = _gateway(tmp_path)
    g.call("critic", "PROMPT-MARKER-7731 review the EUR_USD plan")
    stored = json.dumps([dict(r) for t in ("agent_calls", "jobs", "agent_inbox")
                         for r in mb.read(f"SELECT * FROM {t}")], default=str)
    assert "PROMPT-MARKER-7731" in stored


@known_gap("G8", "no DeepSeek route or provider detection")
def test_g8_deepseek_is_recognised_as_a_provider():
    assert infer_provider("deepseek-chat", {}, "proxy") == "deepseek"


def test_gateway_records_who_actually_answered_not_the_route(tmp_path):
    g, mb = _gateway(tmp_path)
    res = g.call("critic", "plan?")
    row = mb.read("SELECT * FROM agent_calls")[0]
    assert res.ok and row["provider"] == "anthropic" and row["model"] == "claude-sonnet-5-5"
    assert row["prompt_hash"] and row["response_hash"] and "analysis" not in json.dumps(dict(row))


# --- G9: Telegram -----------------------------------------------------------------------------------------

class _Fake:
    def __init__(self):
        self.sent = []

    def call(self, method, payload, timeout=15):
        if method == "getUpdates":
            return []
        self.sent.append(payload.get("text", ""))
        return {}


def _tg(path, **kw):
    f = _Fake()
    return TelegramService(path, token="x", chat_id=111, transport=f, **kw), f


@known_gap("G9", "TelegramService(backfill=True) is the default: the first start alerts every historical ledger row")
def test_g9_first_start_does_not_replay_the_whole_history(tmp_path):
    p = tmp_path / "l.db"
    led = Ledger(p, git_commit="t")
    for i in range(6):
        led.append(Veto(f"2026-03-02-{i:04d}", T0.isoformat(), "risk", f"old row {i}"))
        led.append(Health(T0.isoformat(), "feed", False, f"old fault {i}"))
    svc, fake = _tg(p)
    svc.send_alerts()
    assert fake.sent == []


def test_alerts_are_sent_once_across_restarts(tmp_path):
    p = tmp_path / "l.db"
    led = Ledger(p, git_commit="t")
    led.append(Veto("2026-03-02-0001", T0.isoformat(), "risk", "no room"))
    svc, fake = _tg(p, backfill=True)
    assert svc.send_alerts() == 1
    again, fake2 = _tg(p, backfill=True)
    assert again.send_alerts() == 0 and fake2.sent == []


@known_gap("G9", "alerts are rendered from order/fill/close rows only: no stop, no targets, no justification")
def test_g9_an_entry_alert_carries_stop_targets_and_why(tmp_path):
    p = tmp_path / "l.db"
    led = Ledger(p, git_commit="t")
    pl = plan(entry=1.1000, stop=1.0900)
    led.append(pl)
    led.append(verdict())
    led.append(Order(pl.decision_id, f"{pl.decision_id}-entry", T0.isoformat(), "EUR_USD", 1, 1000, "market", None,
                     "entry", "ensemble"))
    svc, fake = _tg(p, backfill=True)
    svc.send_alerts()
    text = " ".join(fake.sent)
    assert "1.09" in text and "1.12" in text and "trend" in text


# --- G10: agent inbox ---------------------------------------------------------------------------------------

def _inbox_ledger(tmp_path):
    path = tmp_path / "l.db"
    led = Ledger(path, git_commit="t")
    mb = Mailbox(path)
    mb.add_inbox(T0.isoformat(), "scout", "flag", "EUR_USD", {"note": "x"})
    return led, mb


def _pending(led) -> int:
    return led.db.execute("SELECT COUNT(*) FROM agent_inbox WHERE applied_at IS NULL").fetchone()[0]


def test_inbox_rows_are_applied_once_and_ignored_actions_do_not_crash(tmp_path):
    led, mb = _inbox_ledger(tmp_path)
    mb.add_inbox(T0.isoformat(), "evil", "place_order", "EUR_USD", {"qty": 1e9})
    mb.add_inbox(T0.isoformat(), "agent", "veto", "EUR_USD")
    boom = {"flag": lambda row: (_ for _ in ()).throw(RuntimeError("handler bug")), "veto": lambda row: "ok"}
    res = ingest_inbox(led, T0.isoformat(), boom)
    assert [r.applied for r in res] == [False, False, True]
    assert "not allowed" in res[1].result and "handler failed" in res[0].result
    assert ingest_inbox(led, T0.isoformat(), boom) == [] and _pending(led) == 0
    assert led.verify() == (True, None)


@known_gap("G10", "ingest marks the inbox row applied before the ledger row is written: a crash in between loses the request")
def test_g10_a_crash_while_recording_leaves_the_request_pending_for_retry(tmp_path):
    led, _ = _inbox_ledger(tmp_path)
    real = led.append

    def crash(rec):
        raise sqlite3.OperationalError("disk I/O error")
    led.append = crash
    try:
        ingest_inbox(led, T0.isoformat(), {"flag": lambda row: "flagged"})
    except sqlite3.Error:
        pass
    led.append = real
    assert _pending(led) == 1 or led.rows(kind="agent_output")


# --- G11 and S5: the deflated Sharpe and the trial ledger ------------------------------------------------------

@pytest.fixture(scope="module")
def _data():
    return {s: synthetic_bars(1500, seed=i) for i, s in enumerate(["NVDA", "AMD", "AAPL"])}


def _spec():
    return StrategySpec.load("strategies/seeds/stk-ema-pullback-swing.yaml")


def test_rerunning_the_same_search_keeps_the_trial_count(_data, tmp_path):
    led = TrialLedger(tmp_path / "t.sqlite")
    wf = WalkForwardConfig(n_folds=3, grid_points=2)
    a = walk_forward(_spec(), _data, wf=wf, trials=led)
    b = walk_forward(_spec(), _data, wf=wf, trials=led)
    assert a.n_trials == b.n_trials == led.count(_spec().id)
    assert b.oos["expected_max_sharpe_annual"] == pytest.approx(a.oos["expected_max_sharpe_annual"])


@known_gap("G11", "the DSR's trial-Sharpe variance comes from the current run only: re-running ONE parameter set zeroes the deflation")
@pytest.mark.parametrize("narrow", [WalkForwardConfig(n_folds=3, grid_points=1),
                                    WalkForwardConfig(n_folds=3, grid_points=3, max_trials=1)])
def test_g11_rerunning_one_parameter_set_cannot_switch_deflation_off(_data, tmp_path, narrow):
    led = TrialLedger(tmp_path / "t.sqlite")
    wide = walk_forward(_spec(), _data, wf=WalkForwardConfig(n_folds=3, grid_points=3), trials=led)
    assert wide.n_trials == 9 and wide.oos["expected_max_sharpe_annual"] > 0
    one = walk_forward(_spec(), _data, wf=narrow, trials=led)
    assert one.n_trials == 9                                      # N is kept (and passes today) ...
    assert one.oos["expected_max_sharpe_annual"] > 0              # ... but its variance term is not


@known_gap("S5", "a trial is its parameter hash only: the same parameters on different data are one trial")
def test_s5_trial_count_distinguishes_the_data(tmp_path):
    led = TrialLedger(tmp_path / "t.sqlite")
    led.record("s", 1, {"a": 1}, run_id="r1", data_key="stocks-2019-2023")
    led.record("s", 1, {"a": 1}, run_id="r2", data_key="stocks-2024")
    assert led.count("s") == 2


@known_gap("S5", "a strategy at version 3 with an empty trial ledger silently restarts N at its grid size")
def test_s5_empty_ledger_for_a_revised_strategy_is_not_silent(_data, tmp_path):
    spec = _spec()
    spec.version = 3
    rep = walk_forward(spec, _data, wf=WalkForwardConfig(n_folds=3, grid_points=2),
                       trials=TrialLedger(tmp_path / "fresh.sqlite"))
    assert any("trial" in w.lower() and "ledger" in w.lower() for w in rep.warnings)
