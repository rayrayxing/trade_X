import pandas as pd
import pytest

from pfx import H, T0, plan
from tradex.core.interfaces import OrderRequest
from tradex.core.ledger import Ledger
from tradex.core.records import Fill
from tradex.costs.models import MoomooStockCosts, OandaFxCosts, RateMissing
from tradex.execution.sim import SimBroker
from tradex.profit.costcal import (CalibratedCosts, CostCalConfig, calibrate_costs, calibrated_costs,
                                   calibration_report)

DAY = pd.Timedelta(days=1)


def stock_ledger(n, ss_mult=1.0, fee_mult=1.0, book="ensemble", model=None, start=0):
    """Plans and entry fills as a venue booked them: the model's own numbers scaled by what the venue really charged."""
    model = model or MoomooStockCosts()
    led = Ledger(":memory:", git_commit="t")
    for i in range(start, start + n):
        t = T0 + i * H
        price = 100.0 + i % 7
        pl = plan(decision_id=f"2026-03-{1 + i % 28:02d}-{i:04d}", symbol="NVDA", entry=price, stop=price - 2,
                  targets=(price + 3,), t=t, book=book)
        pl.asset_class = "stocks"
        led.append(pl)
        qty = 10.0 + i % 5
        _, unit = model.fill("NVDA", 1, price, t)
        fees = sum(model.order_fees("NVDA", 1, qty, price, t).values())
        led.append(Fill(pl.decision_id, pl.decision_id + "-e", t.isoformat(), "NVDA", 1, qty, price + 0.01,
                        fees * fee_mult, (unit["spread"] + unit["slippage"]) * qty * ss_mult, book))
    return led


COSTS = {"stocks": MoomooStockCosts()}


def test_too_few_fills_leaves_the_model_alone():
    cal = calibrate_costs(stock_ledger(12, ss_mult=3.0), COSTS)["stocks"]
    assert not cal.calibrated and (cal.spread_mult, cal.fee_mult) == (1.0, 1.0)
    assert "12 usable fills, need 30" in cal.status
    assert calibrated_costs(COSTS, {"stocks": cal})["stocks"] is COSTS["stocks"]


def test_a_venue_twice_as_expensive_as_the_model_is_learned_and_shrunk_by_sample_size():
    cal = calibrate_costs(stock_ledger(40, ss_mult=2.0, fee_mult=1.5), COSTS)["stocks"]
    assert cal.calibrated and cal.spread_ratio == pytest.approx(2.0, rel=1e-3) and cal.fee_ratio == pytest.approx(1.5)
    assert cal.spread_mult == pytest.approx(1 + 1.0 * 40 / 60, rel=1e-3) and cal.fee_mult == pytest.approx(1 + 0.5 * 40 / 60)
    big = calibrate_costs(stock_ledger(600, ss_mult=2.0), COSTS)["stocks"]
    assert big.spread_mult > cal.spread_mult and big.spread_mult < 2.0
    assert cal.spread_ci[0] <= 2.0 + 1e-3 and cal.spread_ci[1] >= 2.0 - 1e-3


def test_the_model_only_gets_more_pessimistic_by_default():
    cheap = calibrate_costs(stock_ledger(60, ss_mult=0.5), COSTS)["stocks"]
    assert cheap.spread_ratio == pytest.approx(0.5, rel=1e-3) and cheap.spread_mult == 1.0
    lowered = calibrate_costs(stock_ledger(60, ss_mult=0.5), COSTS, cfg=CostCalConfig(floor=0.5))["stocks"]
    assert 0.5 < lowered.spread_mult < 1.0
    capped = calibrate_costs(stock_ledger(600, ss_mult=30.0), COSTS, cfg=CostCalConfig(cap=3.0))["stocks"]
    assert capped.spread_mult == 3.0


def test_virtual_books_do_not_calibrate_the_model_against_itself():
    cal = calibrate_costs(stock_ledger(60, ss_mult=4.0, book="virtual:x"), COSTS)["stocks"]
    assert cal.n_fills == 0 and not cal.calibrated


def test_calibration_is_against_the_base_model_so_applying_it_does_not_feed_back():
    led = stock_ledger(60, ss_mult=2.0)
    first = calibrate_costs(led, COSTS)
    applied = calibrated_costs(COSTS, first)
    again = calibrate_costs(led, applied)
    assert again["stocks"].spread_mult == pytest.approx(first["stocks"].spread_mult)
    assert CalibratedCosts(applied["stocks"], 3.0).base is COSTS["stocks"]


def test_calibrated_model_charges_what_the_ledger_says():
    base = MoomooStockCosts()
    cc = CalibratedCosts(base, spread_mult=2.0, fee_mult=1.5)
    px0, u0 = base.fill("NVDA", 1, 100.0, T0)
    px1, u1 = cc.fill("NVDA", 1, 100.0, T0)
    assert px1 - 100.0 == pytest.approx(2.0 * (px0 - 100.0)) and u1["spread"] == pytest.approx(2.0 * u0["spread"])
    px_s, _ = cc.fill("NVDA", -1, 100.0, T0)
    assert px_s < 100.0
    assert sum(cc.order_fees("NVDA", 1, 10, 100.0, T0).values()) == pytest.approx(
        1.5 * sum(base.order_fees("NVDA", 1, 10, 100.0, T0).values()))
    assert cc.asset_class == "stocks" and cc.default_borrow_rate_annual == base.default_borrow_rate_annual
    fx = CalibratedCosts(OandaFxCosts(), 1.2)
    assert fx.annual_rate("EUR_USD", 1, T0) == OandaFxCosts().annual_rate("EUR_USD", 1, T0)


def test_a_simulated_venue_with_the_calibrated_model_books_the_learned_cost():
    cc = CalibratedCosts(MoomooStockCosts(), 2.0)
    fills = {}
    for name, costs in (("base", MoomooStockCosts()), ("cal", cc)):
        br = SimBroker(10_000, {"stocks": costs}, bar=DAY)
        br.place(OrderRequest("e", "d", "X", "stocks", 1, 10, stop_loss=90.0, take_profit=120.0))
        br.on_bar("X", T0, 100, 101, 99, 100)
        fills[name] = br.fills()[0]
    assert fills["cal"].spread_slippage_usd == pytest.approx(2 * fills["base"].spread_slippage_usd)


def fx_ledger(n):
    model = OandaFxCosts()
    led = Ledger(":memory:", git_commit="t")
    for i in range(n):
        t = T0 + i * H
        pl = plan(decision_id=f"2026-03-{1 + i % 28:02d}-{i:04d}", symbol="EUR_USD", entry=1.10, stop=1.09,
                  targets=(1.12,), t=t)
        led.append(pl)
        _, unit = model.fill("EUR_USD", 1, 1.10, t)
        led.append(Fill(pl.decision_id, pl.decision_id + "-e", t.isoformat(), "EUR_USD", 1, 1000.0, 1.1001, 0.0,
                        (unit["spread"] + unit["slippage"]) * 1000.0 * 1.5, "ensemble"))
    return led


def test_forex_needs_a_rate_function_and_is_skipped_without_one():
    costs = {"forex": OandaFxCosts()}
    led = fx_ledger(40)
    none = calibrate_costs(led, costs)["forex"]
    assert none.n_fills == 0 and none.skipped == {"no_rate_function": 40} and not none.calibrated
    ok = calibrate_costs(led, costs, rate_fn=lambda ccy, ts: 1.0)["forex"]
    assert ok.calibrated and ok.spread_ratio == pytest.approx(1.5) and ok.fee_ratio is None

    def missing(ccy, ts):
        raise RateMissing(ccy)
    gone = calibrate_costs(led, costs, rate_fn=missing)["forex"]
    assert gone.skipped == {"rate_missing": 40}


def test_report_rows_and_entry_shortfall_diagnostic():
    cal = calibrate_costs(stock_ledger(40, ss_mult=1.0), COSTS)
    row = calibration_report(cal)[0]
    assert row["asset_class"] == "stocks" and row["fills"] == 40 and row["calibrated"]
    assert cal["stocks"].entry_shortfall_bps == pytest.approx(1e4 * 0.01 / 103, rel=0.1)
