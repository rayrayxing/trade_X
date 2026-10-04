import pandas as pd
import pytest

from tradex.costs.models import (MoomooStockCosts, OandaFxCosts, PolicyRates, next_rollover, overnight_days,
                                 rollover_days)


def ny(s):
    return pd.Timestamp(s, tz="America/New_York").tz_convert("UTC")


def test_moomoo_fees_buy_and_sell():
    c = MoomooStockCosts()
    buy = c.order_fees("NVDA", +1, 100, 120.0, ny("2026-01-05 10:00"))
    assert buy["platform"] == pytest.approx(0.99 * 1.09) and buy["commission"] == 0
    assert buy["settlement"] == pytest.approx(0.30) and buy["cat"] == pytest.approx(0.0003)
    assert "sec" not in buy
    sell = c.order_fees("NVDA", -1, 100, 120.0, ny("2026-01-05 15:00"))
    assert sell["sec"] == pytest.approx(20.60e-6 * 100 * 120)
    assert sell["finra_taf"] == pytest.approx(0.0195)
    penny = c.order_fees("XYZ", -1, 1000, 0.2, ny("2026-01-05 15:00"))
    assert penny["settlement"] == pytest.approx(2.0)           # capped at 1% of order value
    small = c.order_fees("XYZ", -1, 10, 0.5, ny("2026-01-05 15:00"))
    assert small["sec"] == 0.01 and small["finra_taf"] == 0.01 # regulatory minimums
    assert c.order_fees("XYZ", -1, 100_000, 5.0, ny("2026-01-05 15:00"))["finra_taf"] == 9.79


def test_stock_fill_pays_spread_and_slippage():
    c = MoomooStockCosts(default_half_spread_bps=1, slippage_bps=2)
    px, unit = c.fill("SPY", +1, 100.0, ny("2026-01-05 10:00"))
    assert px == pytest.approx(100.03)
    px, _ = c.fill("SPY", -1, 100.0, ny("2026-01-05 10:00"))
    assert px == pytest.approx(99.97)


def test_short_borrow_and_margin_interest_accrue_per_night():
    c = MoomooStockCosts(default_borrow_rate_annual=0.036, margin_rate_annual=0.072)
    h = c.holding_cost("TSLA", -1, 10, 200.0, ny("2026-01-05 10:00"), ny("2026-01-07 10:00"), 1.0, 0.0)
    assert h["borrow"] == pytest.approx(10 * 200 * 0.036 * 2 / 360)
    h = c.holding_cost("TSLA", +1, 10, 200.0, ny("2026-01-05 10:00"), ny("2026-01-06 10:00"), 1.0, 1000.0)
    assert h == {"margin_interest": pytest.approx(1000 * 0.072 / 360)}
    assert c.holding_cost("TSLA", -1, 10, 200.0, ny("2026-01-05 10:00"), ny("2026-01-05 15:00"), 1.0, 0.0) == {}
    assert overnight_days(ny("2026-01-09 15:00"), ny("2026-01-12 10:00")) == 3


@pytest.mark.parametrize("t0,t1,days", [
    ("2026-01-05 16:00", "2026-01-06 18:00", 2),   # Mon 17:00 and Tue 17:00
    ("2026-01-05 16:00", "2026-01-05 17:00", 1),   # exactly at the cut counts
    ("2026-01-05 17:00", "2026-01-05 18:00", 0),   # just after the cut
    ("2026-01-08 18:00", "2026-01-12 10:00", 3),   # Thu evening to Mon: Friday charges 3
    ("2026-01-09 18:00", "2026-01-12 10:00", 0),   # Fri evening to Mon: weekend already charged
    ("2026-01-07 16:00", "2026-01-07 18:00", 1),   # Wednesday is a single day at Oanda
])
def test_rollover_days(t0, t1, days):
    assert rollover_days(ny(t0), ny(t1)) == days


def test_next_rollover_skips_weekend():
    cut, days = next_rollover(ny("2026-01-09 18:00"))  # Friday after the cut
    assert cut == ny("2026-01-12 17:00") and days == 1
    cut, days = next_rollover(ny("2026-01-09 10:00"))
    assert cut == ny("2026-01-09 17:00") and days == 3


def test_fx_financing_sign_follows_carry():
    c = OandaFxCosts()
    t0, t1 = ny("2024-01-08 16:00"), ny("2024-01-08 18:00")
    # Early 2024: USD ~5.375%, JPY -0.10%. Long USD/JPY earns carry minus the admin fee.
    long_rate = c.annual_rate("USD_JPY", +1, t1)
    assert long_rate == pytest.approx(0.05375 + 0.0010 - 0.025)
    long_cost = c.holding_cost("USD_JPY", +1, 10_000, 145.0, t0, t1, base_to_usd=1.0)
    short_cost = c.holding_cost("USD_JPY", -1, 10_000, 145.0, t0, t1, base_to_usd=1.0)
    assert long_cost["financing"] < 0 < short_cost["financing"]
    assert short_cost["financing"] == pytest.approx(10_000 * (0.05475 + 0.025) / 365)


def test_fx_fill_uses_pip_size():
    c = OandaFxCosts(slippage_pips=0.0)
    px, unit = c.fill("USD_JPY", +1, 150.0, ny("2026-01-05 10:00"))
    assert px == pytest.approx(150.0 + 0.7 * 0.01)
    px, unit = c.fill("EUR_USD", -1, 1.1, ny("2026-01-05 10:00"))
    assert px == pytest.approx(1.1 - 0.7 * 0.0001)


def test_policy_rates_step_function():
    r = PolicyRates()
    assert r.rate("USD", pd.Timestamp("2021-06-01", tz="UTC")) == pytest.approx(0.00125)
    assert r.rate("JPY", pd.Timestamp("2024-08-15", tz="UTC")) == pytest.approx(0.0025)


def test_moomoo_option_fees_match_brief():
    from tradex.costs.models import MoomooOptionCosts, model_for
    c = model_for("options")
    assert isinstance(c, MoomooOptionCosts)
    t = ny("2026-01-05 10:00")
    buy1 = sum(c.order_fees("SPY", +1, 1, 1.00, t).values())
    assert buy1 == pytest.approx((1.99 + 0.99) * 1.09 + 0.013 + 0.02 + 0.0003)   # minimums bind
    assert buy1 == pytest.approx(3.28, abs=0.01)
    sell1 = c.order_fees("SPY", -1, 1, 1.00, t)
    assert sell1["sec"] == pytest.approx(0.0000206 * 100) and sell1["finra_taf"] == pytest.approx(0.00279)
    assert buy1 + sum(sell1.values()) == pytest.approx(6.56, abs=0.01)           # one-contract round trip
    ten = sum(c.order_fees("SPY", +1, 10, 1.00, t).values())
    assert ten / 10 == pytest.approx(1.07, abs=0.005)
    assert c.order_fees("SPY", +1, 5000, 1.00, t)["occ"] == 55.0                  # OCC cap per trade
