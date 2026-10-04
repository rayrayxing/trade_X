import sqlite3

import pytest

from tradex.core.inbox import Mailbox, ingest_inbox
from tradex.core.ledger import Ledger
from tradex.core.records import Health
from tradex.readiness import evaluate, load_criteria, ready

T = "2026-10-05T00:00:00+00:00"


def test_mailbox_cannot_write_the_chain(tmp_path):
    p = tmp_path / "l.db"
    led = Ledger(p)
    led.append(Health(time=T, check="x", ok=True))
    mb = Mailbox(p)
    mb.add_command(T, "telegram", "pause")
    mb.add_inbox(T, "red_team", "veto", "AAPL")
    for sql in ("INSERT INTO events (kind,time,payload,git_commit,config_hash,run_id,prev_hash,hash)"
                " VALUES ('x','t','{}','g','c','r','p','h')",
                "UPDATE events SET payload='{}'", "DELETE FROM events", "DROP TABLE events"):
        with pytest.raises(sqlite3.DatabaseError):
            mb.db.execute(sql)
    assert not hasattr(mb, "append")
    ok, bad = led.verify()
    assert ok and len(led.rows()) == 1


def test_ingest_applies_whitelist_and_records(tmp_path):
    led = Ledger(tmp_path / "l.db")
    mb = Mailbox(tmp_path / "l.db")
    a = mb.add_inbox(T, "red_team", "veto", "2026-10-05-0001", {"why": "earnings"})
    b = mb.add_inbox(T, "rogue", "place_order", "AAPL")
    c = mb.add_inbox(T, "reviewer", "close", "AAPL")
    seen = []
    res = ingest_inbox(led, T, {"veto": lambda r: seen.append(r["target"]) or "vetoed",
                                "close": lambda r: 1 / 0})
    by = {r.id: r for r in res}
    assert by[a].applied and by[a].result == "vetoed" and seen == ["2026-10-05-0001"]
    assert not by[b].applied and "not allowed" in by[b].result
    assert not by[c].applied and "handler failed" in by[c].result
    assert ingest_inbox(led, T) == []  # idempotent
    outs = led.rows(kind="agent_output")
    assert len(outs) == 3 and outs[1]["action"] == "note"
    assert led.verify()[0]
    assert all(r["applied_at"] for r in led.db.execute("SELECT * FROM agent_inbox"))


def test_busy_timeout_everywhere(tmp_path):
    p = tmp_path / "l.db"
    led, ro, mb = Ledger(p), None, Mailbox(p)
    ro = Ledger(p, read_only=True)
    for db in (led.db, ro.db, mb.db):
        assert db.execute("PRAGMA busy_timeout").fetchone()[0] >= 5000


def test_readiness_unknown_without_real_evidence():
    crit = load_criteria()
    assert {c.id for c in crit} >= {"clean_reconcile_parity", "paper_trades_per_venue", "ray_go_ahead"}
    assert all(c.real_data_only for c in crit)
    out = evaluate(crit, {})
    assert all(o.status == "unknown" for o in out) and not ready(out)


def test_readiness_ignores_non_real_evidence():
    crit = load_criteria()
    ev = {"paper_trades_per_venue": [{"value": 500, "mode": "replay", "real_data": False},
                                      {"value": 500, "mode": "paper", "real_data": False}]}
    o = {x.id: x for x in evaluate(crit, ev)}["paper_trades_per_venue"]
    assert o.status == "unknown" and o.ignored == 2


def test_readiness_pass_fail_and_all_green():
    crit = load_criteria()
    real = lambda v: [{"value": v, "mode": "paper", "real_data": True}]  # noqa: E731
    good = {"clean_reconcile_parity": 14, "paper_trades_per_venue": 30, "costs_vs_model": 10,
            "stops_on_every_position": 0, "governor_tiers_tested": 1, "strategies_passing_real_data": 3,
            "forward_inside_backtest_range": 1, "ray_go_ahead": 1}
    assert ready(evaluate(crit, {k: real(v) for k, v in good.items()}))
    bad = dict(good, costs_vs_model=12)
    out = {o.id: o for o in evaluate(crit, {k: real(v) for k, v in bad.items()})}
    assert out["costs_vs_model"].status == "fail"
