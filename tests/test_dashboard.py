"""Dashboard: ledger fixture, plain HTTP tests (no browser), read-only guarantees, empty states."""
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from tradex.core.ledger import Ledger  # noqa: E402
from tradex.core.records import (AgentOutput, Close, Counterfactual, EquitySnapshot, ExitChange, Fill, Health,  # noqa: E402
                                 Order, TradePlan, Verdict, Veto, Vote)
from tradex.dashboard.app import CSP, create_app  # noqa: E402
from tradex.dashboard.views import Sources, Views, clean, run_kind  # noqa: E402

NOW = datetime(2026, 10, 6, 6, 0, tzinfo=timezone.utc)          # 14:00 in Singapore, same SGT day as the rows below


def iso(minutes_ago: float) -> str:
    return (NOW - timedelta(minutes=minutes_ago)).isoformat()


def plan(did, sym="EUR_USD", d=1, t=100, **kw):
    return TradePlan(decision_id=did, time=iso(t), symbol=sym, asset_class="forex" if "_" in sym else "stocks", direction=d,
                     entry_type="market", entry_price=1.10, stop=1.095, targets=[1.11, 1.12], max_bars=20,
                     invalidation="close below 1.095", families=["trend", "breakout"], strategies=["s-a", "s-b"], score=0.6,
                     p_target=0.45, p_source="base_rate", reward_risk=2.0, ev_r=0.31, cost_r=0.05, tf="H1", **kw)


def build(path: Path, run_id="paper-1"):
    led = Ledger(path, run_id=run_id, git_commit="abc123")
    led.set_config(iso(300), "config/risk/policy.yaml", {"book": {"heat_cap_pct": 3}})
    # winning forex trade
    led.append(Vote(time=iso(100), strategy_id="s-a", strategy_version=1, family="trend", symbol="EUR_USD", asset_class="forex",
                    direction=1, strength=0.6, entry_ref=1.10, stop=1.095, targets=[1.11], max_bars=20, knowable_at=iso(100),
                    decision_id="2026-10-06-0001", tf="H1"))
    led.append(plan("2026-10-06-0001"))
    led.append(Verdict("2026-10-06-0001", iso(99), "accepted", 20000, 100.0, 1.0, [], {"heat": {"now_usd": 0, "cap_usd": 300.0}}, "v1"))
    led.append(Order("2026-10-06-0001", "c1", iso(98), "EUR_USD", 1, 20000, "market", None, "entry"))
    led.append(Fill("2026-10-06-0001", "c1", iso(97), "EUR_USD", 1, 20000, 1.1001, 1.0, 0.5))
    led.append(ExitChange("2026-10-06-0001", iso(60), "stop", 1.095, 1.1, "breakeven"))
    led.append(AgentOutput(iso(70), "chart_reader", "proxy", "m", "veto", "EUR_USD",
                           {"shadow": True, "would": "veto", "reason": "<script>alert(1)</script>"}))
    led.append(Close("2026-10-06-0001", iso(30), "EUR_USD", 1.11, 20000, 180.0, 1.8, "target 1"))
    # blocked by calendar
    led.append(plan("2026-10-06-0002", "GBP_USD", t=50))
    led.append(Veto("2026-10-06-0002", iso(50), "calendar", "FOMC in 2h"))
    led.append(Counterfactual("2026-10-06-0002", iso(40), "calendar", iso(20), "stop", -1.0))
    # declined by risk, a stock
    led.append(plan("2026-10-06-0003", "NVDA", t=45))
    led.append(Verdict("2026-10-06-0003", iso(44), "rejected", 0, 0, 0, ["heat cap reached"], {}, ""))
    # open position (unclosed) + snapshots
    led.append(plan("2026-10-06-0004", "USD_JPY", d=-1, t=20))
    led.append(Verdict("2026-10-06-0004", iso(19), "accepted", 10000, 50.0, 0.5, [], {}, "v4"))
    led.append(Fill("2026-10-06-0004", "c4", iso(18), "USD_JPY", -1, 10000, 150.0, 1.0, 0.5))
    led.append(EquitySnapshot(time=(NOW - timedelta(days=1)).isoformat(), book="ensemble", equity_usd=10000.0, cash_usd=10000.0,
                              open_risk_usd=0.0, positions=[], exposure_by_currency={}, limits={"heat_cap_usd": 300.0, "tier": 0, "paused": False},
                              config_hash="x"))
    led.append(EquitySnapshot(time=iso(10), book="ensemble", equity_usd=10180.0, cash_usd=9000.0, open_risk_usd=50.0,
                              positions=[{"decision_id": "2026-10-06-0004", "symbol": "USD_JPY", "direction": -1, "qty": 10000,
                                          "entry": 150.0, "stop": 150.5, "target": 149.0, "account": "agent", "mark": 149.8}],
                              exposure_by_currency={"JPY": 66.7, "USD": -66.7}, limits={"heat_cap_usd": 300.0, "tier": 1, "paused": False},
                              config_hash="x"))
    led.append(Health(iso(5), "feed", False, "Oanda stream silent"))
    led.append(Health(iso(4), "scheduler", True, "ok"))
    led.add_command(iso(3), "telegram", "pause")
    led.db.execute("INSERT INTO agent_calls (time, category, route, provider, model, ok, latency_ms, tokens_in, tokens_out) "
                   "VALUES (?,?,?,?,?,?,?,?,?)", (iso(30), "chart_reader", "proxy", "anthropic", "claude-sonnet-5-5", 1, 900, 100, 50))
    led.db.commit()
    led.close()


@pytest.fixture
def ledger_path(tmp_path):
    p = tmp_path / "led.sqlite"
    build(p)
    return p


def sources(tmp_path, ledger, **kw):
    cfg = tmp_path / "cfg"
    cfg.mkdir(exist_ok=True)
    (cfg / "accounts.yaml").write_text("accounts:\n  - {name: oanda-practice, venue: oanda, mode: paper, environment: practice,"
                                       " asset_classes: [forex], secret: s, account_id: '123-456-789'}\n")
    (cfg / "runtime.yaml").write_text("agents:\n  mode: shadow\n")
    return Sources(ledger=Path(ledger), strategies_dir=tmp_path / "nostrat", reports_dir=tmp_path / "noreports",
                   bars_dir=tmp_path / "nobars", accounts_path=cfg / "accounts.yaml", runtime_config=cfg / "runtime.yaml",
                   readiness_path=Path(__file__).resolve().parents[1] / "config" / "gates" / "readiness.yaml",
                   state_dir=tmp_path / "state", **kw)


@pytest.fixture
def client(tmp_path, ledger_path):
    return TestClient(create_app(sources(tmp_path, ledger_path)))


@pytest.fixture
def empty_client(tmp_path):
    return TestClient(create_app(sources(tmp_path, tmp_path / "missing.sqlite")))


# --- shell and safety ---------------------------------------------------------------------------

def test_index_and_static_are_served_with_csp(client):
    r = client.get("/")
    assert r.status_code == 200 and "trade_X" in r.text
    assert r.headers["content-security-policy"] == CSP
    assert "lightweight-charts" in r.text and "cdnjs.cloudflare.com" in r.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/app.css").status_code == 200


def test_every_route_is_get_only(client):
    for route in client.app.routes:
        methods = getattr(route, "methods", None) or set()
        assert methods <= {"GET", "HEAD"}, (route.path, methods)
    assert client.post("/api/today").status_code == 405
    assert client.delete("/api/decisions/x").status_code == 405


def test_ledger_is_opened_read_only(tmp_path, ledger_path):
    led = Views(sources(tmp_path, ledger_path)).open()
    with pytest.raises(sqlite3.OperationalError):
        led.db.execute("DELETE FROM events")
    with pytest.raises(PermissionError):
        from tradex.core.records import Health as H
        led.append(H(iso(0), "x", True))
    led.close()


def test_frontend_never_uses_innerhtml():
    js = (Path(__file__).resolve().parents[1] / "tradex" / "dashboard" / "static" / "app.js").read_text()
    assert "innerHTML" not in js and "insertAdjacentHTML" not in js and "document.write" not in js


def test_clean_removes_nan_and_inf():
    assert clean({"a": float("inf"), "b": [float("nan"), 1.5]}) == {"a": None, "b": [None, 1.5]}


# --- empty states: no fake numbers --------------------------------------------------------------

def test_missing_ledger_shows_empty_states_everywhere(empty_client):
    for path in ("/api/today", "/api/positions", "/api/decisions", "/api/performance", "/api/strategies", "/api/system"):
        r = empty_client.get(path)
        assert r.status_code == 200, path
        assert r.json()["empty"] is True, path
    t = empty_client.get("/api/today").json()
    assert "equity" not in t and t["timeline"] == [] and "No ledger" in t["headline"]
    assert empty_client.get("/api/positions").json()["positions"] == []
    assert empty_client.get("/api/decisions/2026-10-06-0001").status_code == 404
    assert empty_client.get("/api/meta").json()["source"] == "none"
    r = empty_client.get("/api/readiness").json()
    assert r["ready"] is False and all(x["status"] == "unknown" for x in r["criteria"])


def test_empty_ledger_file_has_no_numbers(tmp_path):
    p = tmp_path / "e.sqlite"
    Ledger(p).close()
    c = TestClient(create_app(sources(tmp_path, p)))
    m = c.get("/api/meta").json()
    assert m["source"] == "empty" and m["empty"] is True
    assert c.get("/api/today").json()["empty"] is True
    perf = c.get("/api/performance").json()
    assert perf["empty"] is True and perf["trades"] == 0 and perf["equity"] == []
    r = c.get("/api/readiness").json()
    assert all(x["status"] == "unknown" and x["value"] is None for x in r["criteria"]) and r["ready"] is False


# --- views over the fixture ----------------------------------------------------------------------

def test_today_counts_and_headline(tmp_path, ledger_path):
    d = Views(sources(tmp_path, ledger_path)).today(NOW)
    assert d["empty"] is False
    assert d["counts"]["plan"] == 4 and d["counts"]["veto"] == 1 and d["counts"]["declined"] == 1
    assert d["closed_today"] == {"trades": 1, "net_pnl_usd": 180.0, "total_r": 1.8}
    assert d["equity"]["now"] == 10180.0 and d["equity"]["since_yesterday"] == 180.0
    assert d["risk"]["open_positions"] == 1 and d["risk"]["heat_cap_usd"] == 300.0
    assert any("feed" in a["text"] for a in d["attention"])           # failing health check is surfaced
    assert d["timeline"][0]["time"] >= d["timeline"][-1]["time"]       # newest first


def test_positions_exposure_and_lookback(tmp_path, ledger_path):
    v = Views(sources(tmp_path, ledger_path))
    d = v.positions()
    p = d["positions"][0]
    assert p["symbol_label"] == "USD/JPY" and p["side"] == "Sell" and p["has_stop"]
    assert p["unrealised_r"] == pytest.approx(0.4)                      # (150.0-149.8)/0.5 for a short
    assert d["heat"]["used_pct"] == pytest.approx(50 / 300 * 100)
    assert {e["currency"] for e in d["exposure"]} == {"JPY", "USD"}
    back = v.positions(at=(NOW - timedelta(hours=5)).isoformat())     # yesterday's snapshot: flat book
    assert back["empty"] is False and back["positions"] == [] and back["equity_usd"] == 10000.0 and back["looking_back"]
    assert v.positions(at=(NOW - timedelta(days=3)).isoformat())["empty"] is True   # before any snapshot: say so, show nothing


def test_decisions_list_and_filters(client):
    d = client.get("/api/decisions").json()
    by = {x["decision_id"]: x for x in d["decisions"]}
    assert by["2026-10-06-0001"]["outcome"] == "closed" and by["2026-10-06-0001"]["r_multiple"] == 1.8
    assert by["2026-10-06-0002"]["outcome"] == "blocked" and "FOMC" in by["2026-10-06-0002"]["why"]
    assert by["2026-10-06-0002"]["counterfactual_r"] == -1.0
    assert by["2026-10-06-0003"]["outcome"] == "declined" and "heat cap" in by["2026-10-06-0003"]["why"]
    assert by["2026-10-06-0004"]["outcome"] == "open"
    only = client.get("/api/decisions?outcome=blocked").json()["decisions"]
    assert [x["decision_id"] for x in only] == ["2026-10-06-0002"]
    assert client.get("/api/decisions?outcome=nothing").json()["empty"] is True


def test_decision_drawer_has_votes_checks_and_agent_opinions(client):
    d = client.get("/api/decisions/2026-10-06-0001").json()
    assert d["outcome"] == "closed" and len(d["votes"]) == 1 and d["votes"][0]["strategy_id"] == "s-a"
    assert d["checks"][0]["name"] == "heat" and d["verdict"]["outcome"] == "accepted"
    assert len(d["fills"]) == 1 and d["close"]["r_multiple"] == 1.8 and d["exit_changes"][0]["reason"] == "breakeven"
    assert d["agent_opinions"][0]["agent"] == "chart_reader" and d["agent_opinions"][0]["status"].startswith("shadow")
    assert d["git_commit"] == "abc123" and d["config_hash"]
    assert [t["kind"] for t in d["timeline"]][0] == "plan"
    blocked = client.get("/api/decisions/2026-10-06-0002").json()
    assert blocked["vetoes"][0]["source"] == "calendar" and blocked["counterfactual"]["r_multiple"] == -1.0


def test_performance_stats_and_small_sample_flag(client):
    d = client.get("/api/performance").json()
    assert d["stats"]["trades"] == 1 and d["stats"]["win_rate"] == 1.0 and d["stats"]["avg_r"] == 1.8
    assert d["stats"]["profit_factor"] is None                         # no losing trade: not infinite, just undefined
    assert d["enough"] is False and d["costs_usd"] == 3.0              # two fills x (fees + slippage)
    assert len(d["equity"]) == 2 and d["by_asset_class"][0]["asset_class"] == "forex"
    assert d["filters"][0]["blocked_by"] == "calendar" and d["filters"][0]["mean_r"] == -1.0
    assert sum(d["r_hist"]["counts"]) == 1


def test_strategies_forward_vs_backtest(tmp_path, ledger_path):
    reports = tmp_path / "reports" / "s-a"
    reports.mkdir(parents=True)
    (reports / "validation.json").write_text(json.dumps({
        "strategy_id": "s-a", "status": "validated", "reasons": [],
        "oos": {"trades": 150, "win_rate": 0.5, "profit_factor": float("inf"), "expectancy_r": 0.3, "dsr": 0.97}}))
    src = sources(tmp_path, ledger_path)
    src.reports_dir = tmp_path / "reports"
    led = Ledger(ledger_path, run_id="paper-1")
    led.append(Close("v1", iso(10), "EUR_USD", 1.1, 1, 10.0, 0.5, "target", book="virtual:s-a"))
    led.close()
    c = TestClient(create_app(src))
    d = c.get("/api/strategies").json()
    s = {x["id"]: x for x in d["strategies"]}
    assert s["s-a"]["forward"]["trades"] == 1 and s["s-a"]["backtest"]["trades"] == 150
    assert s["s-a"]["delta_avg_r"] == pytest.approx(0.2) and s["s-a"]["too_early"] is True
    assert s["s-a"]["backtest"]["profit_factor"] is None               # inf was sanitised for JSON
    assert s["s-a"]["in_ensemble"]["trades"] == 1 and s["s-b"]["in_ensemble"]["trades"] == 1


def test_seed_strategy_files_are_listed_with_empty_forward(tmp_path):
    src = sources(tmp_path, tmp_path / "none.sqlite")
    src.strategies_dir = Path(__file__).resolve().parents[1] / "strategies"
    d = Views(src).strategies()
    on_disk = len(list((src.strategies_dir).rglob("*.yaml")))
    assert len(d["strategies"]) == on_disk >= 8 and all(s["forward"]["trades"] == 0 and s["forward"]["avg_r"] is None for s in d["strategies"])


def test_system_hides_account_ids_and_reports_gateway(client, tmp_path):
    d = client.get("/api/system").json()
    assert "123-456-789" not in json.dumps(d)
    assert d["accounts"][0]["name"] == "oanda-practice"
    assert d["ledger"]["chain"]["ok"] is True and d["agents_mode"] == "shadow"
    assert d["gateway"]["calls_24h"] == 0 or d["gateway"]["answered_by"][0]["who"] == "anthropic / claude-sonnet-5-5"
    assert any(c["command"] == "pause" for c in d["commands"])
    assert {o["status"] for o in d["ops"]} == {"unknown"}              # no stamp files yet


def test_ops_stamps_drive_backup_status(tmp_path, ledger_path):
    src = sources(tmp_path, ledger_path)
    src.state_dir.mkdir()
    (src.state_dir / "backup_ok").write_text((datetime.now(timezone.utc) - timedelta(hours=2)).isoformat())
    (src.state_dir / "healthcheck_ok").write_text((datetime.now(timezone.utc) - timedelta(hours=3)).isoformat())
    ops = {o["name"]: o for o in Views(src).system()["ops"]}
    assert ops["backup_ok"]["status"] == "ok" and ops["healthcheck_ok"]["status"] == "stale"


def test_broken_chain_is_reported(tmp_path, ledger_path):
    db = sqlite3.connect(ledger_path)
    db.execute("UPDATE events SET payload=replace(payload,'target 1','tampered') WHERE kind='close'")
    db.commit()
    db.close()
    d = TestClient(create_app(sources(tmp_path, ledger_path))).get("/api/system").json()
    assert d["ledger"]["chain"]["ok"] is False and d["ledger"]["chain"]["bad_seq"] > 0


# --- run labelling and readiness -----------------------------------------------------------------------

def test_run_labels():
    assert run_kind("paper-2026-10-06") == "paper" and run_kind("live") == "live"
    assert run_kind("replay") == "replay" and run_kind("dry") == "replay" and run_kind("abc") == "unlabelled"


def test_replay_ledger_is_flagged_and_never_counts_as_readiness_evidence(tmp_path):
    p = tmp_path / "r.sqlite"
    build(p, run_id="replay")
    c = TestClient(create_app(sources(tmp_path, p)))
    m = c.get("/api/meta").json()
    assert m["source"] == "replay" and m["real"] is False and "replay" in m["banner"].lower()
    r = c.get("/api/readiness").json()
    assert r["ready"] is False and r["passing"] == 0
    paper = {x["id"]: x for x in r["criteria"]}["paper_trades_per_venue"]
    assert paper["status"] == "unknown" and paper["ignored"] == 1


def test_paper_ledger_feeds_readiness_honestly(client):
    r = client.get("/api/readiness").json()
    crit = {x["id"]: x for x in r["criteria"]}
    assert crit["paper_trades_per_venue"]["status"] == "fail" and crit["paper_trades_per_venue"]["value"] == 0   # moomoo has 0
    assert crit["stops_on_every_position"]["status"] == "pass"
    assert crit["governor_tiers_tested"]["status"] == "pass"                   # tier went 0 -> 1
    assert crit["costs_vs_model"]["status"] == "unknown"                      # nothing stores the modelled cost: stay unknown
    assert crit["ray_go_ahead"]["status"] == "fail" and r["ready"] is False
    assert r["total"] == 8


def test_missing_stop_is_a_readiness_failure(tmp_path):
    p = tmp_path / "ns.sqlite"
    led = Ledger(p, run_id="paper-1")
    led.append(EquitySnapshot(time=iso(1), book="ensemble", equity_usd=1.0, cash_usd=1.0, open_risk_usd=0.0,
                              positions=[{"decision_id": "d", "symbol": "EUR_USD", "direction": 1, "qty": 1, "entry": 1.0,
                                          "stop": None, "target": None, "account": "a", "mark": 1.0}],
                              exposure_by_currency={}, limits={"tier": 0}, config_hash="x"))
    led.close()
    r = Views(sources(tmp_path, p)).readiness()
    assert {x["id"]: x for x in r["criteria"]}["stops_on_every_position"]["status"] == "fail"
    pos = Views(sources(tmp_path, p)).positions()["positions"][0]
    assert pos["has_stop"] is False


# --- candles ----------------------------------------------------------------------------------------------

def test_bars_empty_state_still_returns_levels(client):
    d = client.get("/api/bars/EUR_USD?tf=H1&decision_id=2026-10-06-0001").json()
    assert d["empty"] is True and d["bars"] == [] and "tradex fetch" in d["reason"]
    assert [l["label"] for l in d["levels"]][:2] == ["entry", "stop"]


def test_bars_from_cached_csv(tmp_path, ledger_path):
    src = sources(tmp_path, ledger_path)
    (tmp_path / "bars" / "oanda").mkdir(parents=True)
    rows = ["timestamp,open,high,low,close,volume"] + [
        f"2026-10-0{d} 00:00:00+00:00,1.1,1.12,1.09,1.11,100" for d in range(1, 6)]
    (tmp_path / "bars" / "oanda" / "EUR_USD_D1.csv").write_text("\n".join(rows))
    src.bars_dir = tmp_path / "bars"
    d = TestClient(create_app(src)).get("/api/bars/EUR_USD?tf=D1").json()
    assert d["empty"] is False and len(d["bars"]) == 5 and d["bars"][0]["open"] == 1.1


def test_bars_reject_path_tricks(client):
    for sym in ("..%2F..%2Fetc%2Fpasswd", "a%2Fb", "x" * 40):
        d = client.get(f"/api/bars/{sym}").json() if client.get(f"/api/bars/{sym}").status_code == 200 else {"empty": True}
        assert d["empty"] is True
    assert client.get("/api/bars/EUR_USD?tf=../../x").json()["empty"] is True


# --- live refresh ----------------------------------------------------------------------------------------------

def test_stream_sends_a_change_event_with_a_token(client):
    with client.stream("GET", "/api/stream?once=true") as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        text = "".join(r.iter_text())
    assert text.startswith("event: change\ndata: ") and "\n\n" in text


def test_change_token_moves_when_the_ledger_grows(tmp_path, ledger_path):
    v = Views(sources(tmp_path, ledger_path))
    a = v.change_token()
    led = Ledger(ledger_path, run_id="paper-1")
    led.append(Health(iso(0), "x", True))
    led.close()
    assert v.change_token() != a
    assert Views(sources(tmp_path, tmp_path / "nope.sqlite")).change_token() == "none"
