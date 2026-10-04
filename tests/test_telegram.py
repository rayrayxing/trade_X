from datetime import datetime, timedelta, timezone

import pytest

from tradex.core.ledger import Ledger
from tradex.core.records import Close, Fill, Health, Order, Veto
from tradex.notify.telegram import TelegramService, is_quiet

RAY = 111
T = "2026-10-05T00:00:00+00:00"


class Fake:
    def __init__(self):
        self.sent, self.updates, self.fail = [], [], False

    def call(self, method, payload, timeout=15):
        if method == "getUpdates":
            ups, self.updates = self.updates, []
            return ups
        if self.fail and method == "sendMessage":
            raise RuntimeError("down")
        self.sent.append((method, payload))
        return {}

    def texts(self):
        return [p["text"] for m, p in self.sent if m == "sendMessage"]


class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


def day(h, m=0):  # SGT wall clock -> utc datetime
    return datetime(2026, 10, 5, h, m, tzinfo=timezone(timedelta(hours=8))).astimezone(timezone.utc)


def mk(tmp_path, now=None):
    p = tmp_path / "l.db"
    led = Ledger(p)
    f = Fake()
    clock = Clock(now or day(12))
    return led, f, clock, TelegramService(p, token="x", chat_id=RAY, transport=f, now=clock)


def msg(text, chat=RAY, uid=None, n=1):
    return {"update_id": n, "message": {"chat": {"id": chat}, "from": {"id": uid or chat}, "text": text}}


def fill_ledger(led):
    led.append(Order("d1", "c1", T, "AAPL", 1, 10, "market", None, "entry"))
    led.append(Fill("d1", "c1", T, "AAPL", 1, 10, 100.0, 1.0, 0.5))
    led.append(Close("d1", T, "AAPL", 105.0, 10, 48.0, 1.2, "target"))
    led.append(Veto("d2", T, "calendar", "NFP blackout"))
    led.append(Health(T, "reconcile", False, "mismatch"))
    led.append(Health(T, "feed", True))


def test_one_alert_per_row_and_idempotent(tmp_path):
    led, f, clock, svc = mk(tmp_path)
    fill_ledger(led)
    assert svc.send_alerts() == 5
    t = f.texts()
    assert sum(x.startswith("ORDER") for x in t) == 1 and sum(x.startswith("CLOSE") for x in t) == 1
    assert any(x.startswith("FAULT") for x in t) and any(x.startswith("BLOCKED") for x in t)
    assert svc.send_alerts() == 0
    svc2 = TelegramService(tmp_path / "l.db", token="x", chat_id=RAY, transport=f, now=clock)
    assert svc2.send_alerts() == 0 and len(f.texts()) == 5       # restart: still no duplicates
    led.append(Order("d3", "c3", T, "MSFT", -1, 5, "market", None, "exit"))
    assert svc2.send_alerts() == 1


def test_failed_send_is_retried_not_lost(tmp_path):
    led, f, clock, svc = mk(tmp_path)
    led.append(Order("d1", "c1", T, "AAPL", 1, 10, "market", None, "entry"))
    f.fail = True
    assert svc.send_alerts() == 0
    f.fail = False
    assert svc.send_alerts() == 1 and svc.send_alerts() == 0


def test_quiet_hours_silent_but_faults_loud(tmp_path):
    assert is_quiet(day(23, 0)) and is_quiet(day(7, 29)) and not is_quiet(day(7, 30)) and not is_quiet(day(22, 59))
    led, f, clock, svc = mk(tmp_path, now=day(2))
    fill_ledger(led)
    svc.send_alerts()
    loud = {p["text"][:5]: p["disable_notification"] for m, p in f.sent}
    assert loud["FAULT"] is False and loud["ORDER"] is True and loud["CLOSE"] is True
    clock.t = day(12)
    led.append(Order("d9", "c9", T, "X", 1, 1, "market", None, "entry"))
    svc.send_alerts()
    assert f.sent[-1][1]["disable_notification"] is False


def test_commands_written_not_executed(tmp_path):
    led, f, clock, svc = mk(tmp_path)
    f.updates = [msg("/pause", n=1), msg("/resume@tradexbot", n=2), msg("/status", n=3)]
    assert svc.poll_once() == 3
    rows = list(led.db.execute("SELECT source, command FROM commands ORDER BY id"))
    assert [(r["source"], r["command"]) for r in rows] == [("telegram", "pause"), ("telegram", "resume")]
    assert led.rows() == [] and led.verify()[0]
    assert any("No snapshot" in t for t in f.texts())
    f.updates = [msg("/pause", n=3)]                                 # offset persisted: update 3 not replayed
    svc.poll_once()


def test_allowlist_drops_strangers(tmp_path):
    led, f, clock, svc = mk(tmp_path)
    f.updates = [msg("/flatten", chat=999), msg("/pause", chat=RAY, uid=999, n=2)]
    svc.poll_once()
    assert f.sent == [] and svc.dropped == 2
    assert list(led.db.execute("SELECT * FROM commands")) == []


def cb(data, chat=RAY, n=10):
    return {"update_id": n, "callback_query": {"id": "q", "data": data, "from": {"id": chat},
                                               "message": {"message_id": 5, "chat": {"id": chat}}}}


def test_flatten_needs_confirm_and_expires(tmp_path):
    led, f, clock, svc = mk(tmp_path)
    f.updates = [msg("/flatten")]
    svc.poll_once()
    assert list(led.db.execute("SELECT * FROM commands")) == []
    kb = f.sent[-1][1]["reply_markup"]["inline_keyboard"][0]
    confirm = kb[0]["callback_data"]
    f.updates = [cb(confirm, chat=999, n=11)]                        # stranger tapping: ignored
    svc.poll_once()
    assert list(led.db.execute("SELECT * FROM commands")) == []
    f.updates = [cb(confirm)]
    svc.poll_once()
    assert [r["command"] for r in led.db.execute("SELECT command FROM commands")] == ["flatten"]
    f.updates = [cb(confirm, n=12)]                                  # single use
    svc.poll_once()
    assert len(list(led.db.execute("SELECT * FROM commands"))) == 1

    f.updates = [msg("/flatten", n=13)]
    svc.poll_once()
    tok = f.sent[-1][1]["reply_markup"]["inline_keyboard"][0][0]["callback_data"]
    clock.t += timedelta(seconds=61)
    f.updates = [cb(tok, n=14)]
    svc.poll_once()
    assert len(list(led.db.execute("SELECT * FROM commands"))) == 1
    assert "Expired" in f.sent[-2][1]["text"]


def test_service_cannot_append_chain(tmp_path):
    led, f, clock, svc = mk(tmp_path)
    import sqlite3
    with pytest.raises(sqlite3.DatabaseError):
        svc.mb.db.execute("DELETE FROM events")


def test_approve_live_needs_confirm_and_counts_on_scorecard(tmp_path):
    from tradex.readiness_evidence import scorecard
    led, f, clock, svc = mk(tmp_path)
    f.updates = [msg("/approve_live")]
    svc.poll_once()
    assert list(led.db.execute("SELECT * FROM commands")) == []
    approve = f.sent[-1][1]["reply_markup"]["inline_keyboard"][0][0]["callback_data"]
    f.updates = [cb(approve.replace("approve_live", "flatten"), n=11)]   # a forged action is not this tap
    svc.poll_once()
    assert list(led.db.execute("SELECT * FROM commands")) == []
    f.updates = [msg("/approve_live", n=12)]
    svc.poll_once()
    approve = f.sent[-1][1]["reply_markup"]["inline_keyboard"][0][0]["callback_data"]
    f.updates = [cb(approve, n=13)]
    svc.poll_once()
    rows = [(r["source"], r["command"], r["args"]) for r in led.db.execute("SELECT * FROM commands")]
    assert rows == [("telegram", "go_live_approved", '{"by": "ray"}')]
    go = {o.id: o for o in scorecard(led, tmp_path / "none.json")}["ray_go_ahead"]
    assert go.value == 1
