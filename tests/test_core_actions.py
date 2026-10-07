"""Agent inbox actions and Ray's commands applied by the core; faults as loud Health rows."""
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from test_spine import _frames, _two_family_specs
from tradex.agents.gateway import Gateway
from tradex.core.inbox import Mailbox
from tradex.core.ledger import Ledger
from tradex.core.loop import CoreConfig
from tradex.core.replay import build_replay_core, close_events, run_replay
from tradex.execution.guard import GuardedBroker, OrderGuard, ledger_verdicts
from tradex.notify.telegram import TelegramService
from tradex.runtime.config import RuntimeConfig
from tradex.runtime.runner import LiveRunner
from tradex.runtime.schedule import BarCloseScheduler

T = "2026-10-05T00:00:00+00:00"


class Core:
    """The replay core stepped one close at a time, ensemble behind the order guard."""

    def __init__(self, agents="shadow", mode="replay", ledger=None, wrap=None):
        frames = _frames()
        self.start = frames["AAA"].index[260]
        self.led = ledger or Ledger(":memory:", git_commit="t")
        self.core = build_replay_core(_two_family_specs(), frames, self.led, self.start,
                                      cfg=CoreConfig(agents_mode=agents, mode=mode))
        br = GuardedBroker(self.core.brokers["ensemble"], OrderGuard(ledger_verdicts(self.led), {"sim"}))
        self.core.brokers["ensemble"] = wrap(br) if wrap else br
        self.events = iter(close_events(self.core.data, self.core.tfs, self.start))

    def step(self):
        ts, tf = next(self.events)
        self.core.clock.set(ts)
        self.core.on_bar_close(tf, ts)
        return ts

    def until(self, cond, limit=230):
        for _ in range(limit):
            self.step()
            if cond():
                return
        raise AssertionError("condition never met")

    def ask(self, action, target, **body):
        return self.led.add_agent_inbox(T, "scout", action, target, body)

    def outputs(self):
        return [r for r in self.led.rows(kind="agent_output") if "inbox_id" in r["body"]]


def _pending(c):
    """An unfilled ensemble entry. Simulated market entries fill at the next open, so the
    test applies the inbox right after the close that placed it (an agent that answered
    inside the bar, before the venue filled)."""
    c.until(lambda: c.core.pending_entries())
    return c.core.pending_entries()[0]


def _fills(c, did):
    return [f for f in c.led.rows(kind="fill", decision_id=did) if f["book"] == "ensemble"]


def test_agents_default_to_shadow():
    assert CoreConfig().agents_mode == "shadow"
    assert RuntimeConfig.load().agents_mode == "shadow"


def test_runtime_config_rejects_unknown_agent_mode(tmp_path):
    p = tmp_path / "r.yaml"
    p.write_text("agents: {mode: yolo}\n")
    with pytest.raises(ValueError):
        RuntimeConfig.load(p)
    p.write_text("agents: {mode: active}\n")
    assert RuntimeConfig.load(p).agents_mode == "active"


def test_shadow_veto_records_what_it_would_do_and_applies_nothing():
    c = Core("shadow")
    did = _pending(c)
    c.ask("veto", did, reason="earnings")
    c.core.desk.ingest(c.core.clock.now())
    c.step()
    assert _fills(c, did)                                             # the trade went ahead
    assert not [v for v in c.led.rows(kind="veto", decision_id=did) if v["source"].startswith("agent")]
    out = c.outputs()[-1]
    assert out["body"]["applied"] is False and out["body"]["result"].startswith("shadow: would veto")


def test_active_veto_cancels_the_unfilled_entry():
    c = Core("active")
    did = _pending(c)
    c.ask("veto", did, reason="earnings")
    c.core.desk.ingest(c.core.clock.now())
    c.step()
    assert not _fills(c, did)
    v = [v for v in c.led.rows(kind="veto", decision_id=did)][-1]
    assert v["source"] == "agent:scout" and v["reason"] == "earnings"
    assert c.outputs()[-1]["body"]["applied"] is True


def test_active_shrink_cuts_size_and_never_increases():
    c = Core("active")
    did = _pending(c)
    q = c.core.entry_qty(did)
    c.ask("shrink", did, factor=1.5)
    c.ask("shrink", did, qty=q + 10)
    c.ask("shrink", did, factor=0.5)
    c.core.desk.ingest(c.core.clock.now())
    res = [o["body"] for o in c.outputs()[-3:]]
    assert [r["applied"] for r in res] == [False, False, True]
    assert "never increases" in res[0]["result"] and "never increases" in res[1]["result"]
    c.step()
    f = _fills(c, did)
    assert len(f) == 1 and f[0]["qty"] == q // 2


def _first_aaa_entry(led):
    return next(o for o in led.rows(kind="order") if o["book"] == "ensemble" and o["symbol"] == "AAA"
                and o["purpose"].startswith("entry"))


def test_a_standing_shrink_is_applied_once_not_twice():
    """0.5 must size about half of what the gate approved. It once also went into the gate's own risk
    budget, so the order came out at about a quarter."""
    frames = _frames()
    base_led = Ledger(":memory:", git_commit="t")
    run_replay(_two_family_specs(), frames, base_led, frames["AAA"].index[260], cfg=CoreConfig(agents_mode="active"))
    led = Ledger(":memory:", git_commit="t")
    led.add_agent_inbox(T, "scout", "shrink", "AAA", {"factor": 0.5, "until": "2100-01-01", "reason": "thin book"})
    run_replay(_two_family_specs(), frames, led, frames["AAA"].index[260], cfg=CoreConfig(agents_mode="active"))
    base, cut = _first_aaa_entry(base_led), _first_aaa_entry(led)
    approved = next(v for v in led.rows(kind="verdict") if v["decision_id"] == cut["decision_id"])
    base_v = next(v for v in base_led.rows(kind="verdict") if v["decision_id"] == base["decision_id"])
    assert approved["qty"] == base_v["qty"] == base["qty"]          # the gate's own size did not move
    assert cut["qty"] == float(int(base["qty"] * 0.5 + 1e-9)) and cut["purpose"].startswith("entry: agent shrink")
    assert approved["checks"]["kelly"]["size_factor"] == 1.0


def test_calendar_halving_reaches_the_gate_not_dropped_by_the_agent_factor():
    c = Core("active")
    c.core.calendar.check = lambda *a, **k: (None, 0.5, [])
    c.until(lambda: c.led.rows(kind="verdict"))
    v = c.led.rows(kind="verdict")[0]
    assert v["checks"]["kelly"]["size_factor"] == 0.5


def test_close_request_sends_a_reducing_exit_through_the_guard():
    c = Core("active")
    c.until(lambda: c.core.agent_positions())
    p = c.core.agent_positions()[0]
    c.ask("close", p.decision_id, reason="thesis broken")
    c.step()
    exits = [o for o in c.led.rows(kind="order", decision_id=p.decision_id) if o["purpose"].startswith("exit: agent")]
    assert len(exits) == 1 and exits[0]["side"] == -p.direction and exits[0]["qty"] == p.qty
    c.step()
    assert p.decision_id not in {q.decision_id for q in c.core.agent_positions()}
    assert c.led.rows(kind="close", decision_id=p.decision_id)


def test_shadow_close_and_flag_change_nothing():
    c = Core("shadow")
    c.until(lambda: c.core.agent_positions())
    p = c.core.agent_positions()[0]
    c.ask("close", p.decision_id)
    c.ask("flag", p.symbol, note="odd volume")
    c.step()
    assert not [o for o in c.led.rows(kind="order", decision_id=p.decision_id) if "agent" in o["purpose"]]
    close, flag = (o["body"] for o in c.outputs()[-2:])
    assert close["result"].startswith("shadow: would close") and not close["applied"]
    assert flag["result"] == "flagged for review"


@pytest.mark.parametrize("mode", ["shadow", "active"])
def test_standing_symbol_veto_blocks_plans_only_when_active(mode):
    frames = _frames()
    led = Ledger(":memory:", git_commit="t")
    led.add_agent_inbox(T, "scout", "veto", "AAA", {"until": "2100-01-01", "reason": "fraud probe"})
    run_replay(_two_family_specs(), frames, led, frames["AAA"].index[260], cfg=CoreConfig(agents_mode=mode))
    ens = [o for o in led.rows(kind="order") if o["book"] == "ensemble" and o["symbol"] == "AAA"]
    agent_vetoes = [v for v in led.rows(kind="veto") if v["source"] == "agent:scout"]
    would = [r for r in led.rows(kind="agent_output") if r["body"].get("shadow")]
    if mode == "active":
        assert not ens and agent_vetoes and not would
    else:
        assert ens and not agent_vetoes and would and would[0]["body"]["would"] == "veto"


def test_pause_resume_and_flatten_commands():
    c = Core()
    c.until(lambda: c.core.agent_positions())
    held = c.core.agent_positions()
    c.led.add_command(T, "telegram", "flatten")
    c.step()
    assert c.core.paused
    cmd = c.led.db.execute("SELECT result FROM commands").fetchone()["result"]
    assert f"exits sent for {len(held)} of {len(held)}" in cmd
    flat = [o for o in c.led.rows(kind="order") if o["purpose"] == "exit: flatten command"]
    assert {o["decision_id"] for o in flat} == {p.decision_id for p in held}
    for _ in range(5):
        c.step()
    assert not c.core.agent_positions()
    n = len([o for o in c.led.rows(kind="order") if o["book"] == "ensemble"])
    for _ in range(30):                                              # paused: no new ensemble entries
        c.step()
    assert len([o for o in c.led.rows(kind="order") if o["book"] == "ensemble"]) == n
    c.led.add_command(T, "telegram", "resume")
    c.step()
    assert not c.core.paused


# --- faults --------------------------------------------------------------------------------

class FlakyVenue:
    """A venue whose first order submission fails with a network error."""

    def __init__(self, inner):
        self.inner, self.failed = inner, False

    def place(self, req):
        if not self.failed:
            self.failed = True
            raise ConnectionError("venue unreachable")
        return self.inner.place(req)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def test_broker_error_in_paper_is_a_fault_not_a_crash():
    c = Core(mode="paper", wrap=FlakyVenue)
    c.until(lambda: c.core.brokers["ensemble"].failed)
    h = [r for r in c.led.rows(kind="health") if not r["ok"]]
    assert len(h) == 1 and h[0]["check"] == "broker" and "ConnectionError" in h[0]["detail"]
    assert any(v["source"] == "broker" for v in c.led.rows(kind="veto"))


def test_broker_error_in_replay_still_raises():
    c = Core(wrap=FlakyVenue)
    with pytest.raises(ConnectionError):
        c.until(lambda: False)


class FakeTelegram:
    def __init__(self):
        self.sent = []

    def call(self, method, payload, timeout=15):
        if method == "sendMessage":
            self.sent.append(payload)
        return [] if method == "getUpdates" else {}


def test_fault_in_core_reaches_telegram_as_one_loud_message(tmp_path):
    path = tmp_path / "ledger.db"
    c = Core(mode="paper", wrap=FlakyVenue, ledger=Ledger(path, git_commit="t"))
    c.until(lambda: c.core.brokers["ensemble"].failed)
    quiet = datetime(2026, 10, 5, 2, 0, tzinfo=timezone(timedelta(hours=8)))      # 02:00 Singapore
    fake = FakeTelegram()
    TelegramService(path, chat_id=1, transport=fake, now=lambda: quiet, backfill=True).send_alerts()
    loud = [m for m in fake.sent if not m["disable_notification"]]
    assert len(loud) == 1 and loud[0]["text"].startswith("FAULT broker:")
    assert all(m["disable_notification"] for m in fake.sent if m is not loud[0])   # trade alerts stay quiet


def test_scheduler_overrun_and_missed_closes_are_faults():
    c = Core()
    t = [0.0]

    def slow(tf, ts):
        t[0] += 90.0
    runner = LiveRunner(c.core, BarCloseScheduler(c.core.tfs, c.led, done_until=c.start), before_close=slow,
                        overrun=pd.Timedelta(seconds=60), timer=lambda: t[0])
    c.core.clock.set(c.start + pd.Timedelta(days=3, hours=1))
    runner.tick()
    h = [r for r in c.led.rows(kind="health") if r["check"] == "scheduler"]
    assert any("took 90.0s" in r["detail"] for r in h) and any("missed 2 D1 closes" in r["detail"] for r in h)
    assert not any(r["ok"] for r in h)


def test_agent_gateway_skip_is_not_a_fault(tmp_path):
    path = tmp_path / "ledger.db"
    led = Ledger(path, git_commit="t")

    def down(url, headers, body, timeout):
        raise TimeoutError()
    r = Gateway(Mailbox(path), http=down, secret=lambda n: "x").call("scout_analyst", "p")
    assert not r.ok
    assert not led.rows(kind="health")
