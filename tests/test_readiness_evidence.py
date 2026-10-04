import json

from tradex.core.ledger import Ledger
from tradex.core.records import Close, EquitySnapshot, Fill, Health, TradePlan, Verdict
from tradex.readiness_evidence import collect, render, scorecard

T = "2026-10-05T00:00:00+00:00"


def by_id(outs):
    return {o.id: o for o in outs}


def gate(tmp_path, n_pass=3):
    rows = [{"strategy_id": f"s{i}", "result": "pass", "bars_used": 100} for i in range(n_pass)]
    rows += [{"strategy_id": "bad", "result": "fail", "bars_used": 100},
             {"strategy_id": "synth", "result": "pass", "bars_used": 100, "data_source": "synthetic"}]
    p = tmp_path / "gate.json"
    p.write_text(json.dumps({"rows": rows}))
    return p


def test_empty_ledger_is_unknown_with_reasons(tmp_path):
    outs = by_id(scorecard(Ledger(tmp_path / "l.db"), tmp_path / "missing.json"))
    for k in ("clean_reconcile_parity", "paper_trades_per_venue", "costs_vs_model", "stops_on_every_position",
              "governor_tiers_tested", "strategies_passing_real_data", "forward_inside_backtest_range"):
        assert outs[k].status == "unknown" and outs[k].detail, k
    assert outs["ray_go_ahead"].status == "fail" and outs["ray_go_ahead"].value == 0


def test_replay_evidence_is_ignored(tmp_path):
    led = Ledger(tmp_path / "l.db", run_id="replay")
    for d in range(1, 21):
        for c in ("reconcile", "parity"):
            led.append(Health(time=f"2026-09-{d:02d}T10:00:00+00:00", check=c, ok=True))
    o = by_id(scorecard(led, tmp_path / "x.json"))["clean_reconcile_parity"]
    assert o.status == "unknown" and o.ignored == 1


def test_clean_days_streak_breaks_on_failure_and_gap(tmp_path):
    led = Ledger(tmp_path / "l.db", run_id="paper")
    for d in range(1, 20):
        ok = d != 4
        for c in ("reconcile", "parity"):
            if d == 10 and c == "parity":
                continue
            led.append(Health(time=f"2026-09-{d:02d}T10:00:00+00:00", check=c, ok=ok))
    o = by_id(scorecard(led, tmp_path / "x.json"))["clean_reconcile_parity"]
    assert o.status == "fail" and o.value == 9  # day 10 has no parity row, so only 11..19 count
    led.append(Health(time="2026-09-20T10:00:00+00:00", check="reconcile", ok=False))
    assert by_id(scorecard(led, tmp_path / "x.json"))["clean_reconcile_parity"].value == 0


def trade(led, i, cls, fees, cost_r=0.1, risk=100.0, stop=1.0):
    d = f"2026-10-05-{i:04d}"
    led.append(TradePlan(decision_id=d, time=T, symbol="X", asset_class=cls, direction=1, entry_type="market",
                         entry_price=1, stop=stop, targets=[2], max_bars=5, invalidation="", families=[],
                         strategies=[], score=.5, p_target=.5, p_source="base_rate", reward_risk=2, ev_r=.2,
                         cost_r=cost_r))
    led.append(Verdict(decision_id=d, time=T, outcome="accepted", qty=1, risk_usd=risk, risk_pct=1, reasons=[],
                       checks={}))
    led.append(Fill(decision_id=d, client_order_id="c", time=T, symbol="X", side=1, qty=1, price=1, fees_usd=fees,
                    spread_slippage_usd=0))
    led.append(Close(decision_id=d, time=T, symbol="X", exit_price=2, qty=1, net_pnl_usd=1, r_multiple=1, reason="t"))


def test_trades_per_venue_and_cost_error(tmp_path):
    led = Ledger(tmp_path / "l.db", run_id="paper-1")
    for i in range(30):
        trade(led, i, "forex", 10.5)          # modelled 0.1*100 = 10 -> 5% off
    for i in range(30, 45):
        trade(led, i, "stocks", 10.0)
    o = by_id(scorecard(led, tmp_path / "x.json"))
    assert o["paper_trades_per_venue"].status == "fail" and o["paper_trades_per_venue"].value == 15
    assert o["paper_trades_per_venue"].evidence[-1]["closed_by_venue"] == {"oanda": 30, "moomoo": 15}
    assert o["costs_vs_model"].status == "pass" and round(o["costs_vs_model"].value, 2) == 3.33
    for i in range(45, 60):
        trade(led, i, "stocks", 10.0)
    assert by_id(scorecard(led, tmp_path / "x.json"))["paper_trades_per_venue"].status == "pass"


def snap(led, t, tier, stop=1.0):
    led.append(EquitySnapshot(time=t, book="ensemble", equity_usd=1, cash_usd=1, open_risk_usd=0,
                              positions=[{"symbol": "X", "stop": stop}], exposure_by_currency={},
                              limits={"tier": tier}, config_hash="c"))


def test_stops_and_tiers(tmp_path):
    led = Ledger(tmp_path / "l.db", run_id="paper")
    snap(led, T, 2)
    snap(led, "2026-10-06T00:00:00+00:00", 1)
    o = by_id(scorecard(led, tmp_path / "x.json"))
    assert o["stops_on_every_position"].status == "pass" and o["governor_tiers_tested"].value == 1
    snap(led, "2026-10-07T00:00:00+00:00", 1, stop=None)
    led.append(Health(time=T, check="stop_guard", ok=False))
    assert by_id(scorecard(led, tmp_path / "x.json"))["stops_on_every_position"].value == 2


def test_strategies_from_gate_file_and_go_ahead(tmp_path):
    led = Ledger(tmp_path / "l.db")
    o = by_id(scorecard(led, gate(tmp_path)))
    assert o["strategies_passing_real_data"].status == "pass" and o["strategies_passing_real_data"].value == 3
    assert by_id(scorecard(led, gate(tmp_path, 2)))["strategies_passing_real_data"].status == "fail"
    led.add_command(T, "dashboard", "go_live_approved", {"by": "ray"})
    led.add_command(T, "red_team", "go_live_approved", {"by": "ray"})
    led.add_command(T, "telegram", "go_live_approved", {"by": "someone"})
    led.add_command(T, "telegram", "pause", {"by": "ray"})
    assert by_id(scorecard(led, gate(tmp_path)))["ray_go_ahead"].value == 0
    led.add_command(T, "telegram", "go_live_approved", {"by": "ray"})
    led.add_command(T, "cli", "go_live_approved", {"by": "ray"})
    assert by_id(scorecard(led, gate(tmp_path)))["ray_go_ahead"].value == 2


def test_render_and_cli(tmp_path, capsys):
    from tradex.cli import main
    Ledger(tmp_path / "l.db").close()
    text = render(scorecard(Ledger(tmp_path / "l.db"), tmp_path / "x.json"))
    assert "UNKNOWN" in text and "NOT READY" in text
    assert main(["readiness", "--ledger", str(tmp_path / "l.db"), "--gate-results", str(tmp_path / "x.json")]) == 1
    assert "ray_go_ahead" in capsys.readouterr().out
