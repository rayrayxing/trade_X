"""Multi-venue book equity in USD and the per-venue margin headroom check."""
import pandas as pd
import pytest

from test_spine import ZERO_COST, _frames, _plan, _two_family_specs
from tradex.core.interfaces import AccountInfo, OrderRequest, ReplayClock
from tradex.core.ledger import Ledger
from tradex.core.replay import build_replay_core, close_events
from tradex.execution.sim import SimBroker
from tradex.risk.gate import BookState, RiskGate
from tradex.runtime.fx import MarketDataRates, MissingRate, SeriesRates
from tradex.runtime.market import BarStore
from tradex.runtime.venues import MultiVenueBook

T0 = pd.Timestamp("2026-03-02 21:00", tz="UTC")
DAY = pd.Timedelta(days=1)


class _YenAccount(SimBroker):
    """A venue whose account is kept in JPY."""

    def account(self):
        a = super().account()
        return AccountInfo("acct-jpy", "fake", "JPY", a.equity * 150, a.cash * 150, a.margin_used * 150,
                           a.buying_power * 150)


def test_book_equity_sums_venues_in_usd_at_live_rates():
    clock = ReplayClock(T0)
    usdjpy = pd.DataFrame({"open": 160.0, "high": 160.0, "low": 160.0, "close": 160.0, "volume": 1.0},
                          index=pd.DatetimeIndex([T0 - pd.Timedelta(hours=1)]))
    rates = MarketDataRates(BarStore("H1", {"USD_JPY": usdjpy}, clock), mode="paper")
    book = MultiVenueBook({"stocks": SimBroker(10_000, ZERO_COST, bar=DAY),
                           "forex": _YenAccount(10_000, ZERO_COST, bar=DAY)}, rates, clock)
    assert book.account().equity == pytest.approx(10_000 + 10_000 * 150 / 160)
    assert book.account().currency == "USD"
    clock.set(T0 + pd.Timedelta(days=2))                       # the USD_JPY bar is now stale
    with pytest.raises(MissingRate):
        book.account()
    with pytest.raises(MissingRate):
        MultiVenueBook({"forex": _YenAccount(10_000, ZERO_COST)}, SeriesRates({}, "paper"), clock).account()


def test_orders_route_to_the_venue_of_their_asset_class():
    fx, stk = SimBroker(10_000, ZERO_COST, bar=DAY), SimBroker(10_000, ZERO_COST, bar=DAY)
    book = MultiVenueBook({"forex": fx, "stocks": stk}, SeriesRates(), ReplayClock(T0))
    book.place(OrderRequest("a", "da", "EUR_USD", "forex", 1, 1000, stop_loss=1.09))
    book.place(OrderRequest("b", "db", "X", "stocks", 1, 10, stop_loss=95.0))
    assert set(fx.orders) == {"a"} and set(stk.orders) == {"b"}
    book.on_bar("EUR_USD", T0, 1.1, 1.1, 1.1, 1.1)
    book.on_bar("X", T0, 100, 100, 100, 100)
    assert {p.symbol for p in book.positions()} == {"EUR_USD", "X"}
    assert [f.client_order_id for f in book.fills()] == ["a", "b"]
    assert book.order_status("a").status == "filled" and book.order_status("zz").status == "unknown"
    book.amend_stop("db", 97.0)
    assert stk.positions()[0].stop == 97.0 and fx.positions()[0].stop == 1.09


def test_gate_caps_size_at_the_venue_free_margin():
    gate = RiskGate.from_policy()
    plan = _plan("X", "stocks", entry=100.0, stop=99.0, t1=103.0, p=0.6)
    free = gate.review(plan, BookState(10_000, []), 1.0)
    capped = gate.review(plan, BookState(10_000, [], margin_max_qty=7.0, margin={"venue": "sim"}), 1.0)
    assert free.qty > 7 and "margin" not in free.checks
    assert capped.qty == 7 and "size cut by margin limit" in capped.reasons
    assert capped.checks["margin"] == {"venue": "sim", "max_qty": 7.0}


def test_core_sizes_within_the_stock_venue_headroom():
    frames = _frames()
    start = frames["AAA"].index[260]
    clock = ReplayClock()
    led = Ledger(":memory:")
    core = build_replay_core(_two_family_specs(), frames, led, start, clock=clock)
    tight = SimBroker(10_000, core.costs, bar=core.bar, rate_fn=core.rates.usd_per_unit)
    tight.margin_rates = {"stocks": 4.0, "forex": 0.05}       # a venue that wants 4x the notional free
    core.brokers["ensemble"] = MultiVenueBook({"stocks": tight}, core.rates, clock)
    core.cfg.margin_rates = {"stocks": 4.0, "forex": 0.05}
    for ts, tf in close_events(core.data, core.tfs, start):
        clock.set(ts)
        core.on_bar_close(tf, ts)
    ens = [v for v in led.rows(kind="verdict") if v["decision_id"] in {p["decision_id"] for p in led.rows(kind="plan", book="ensemble")}]
    assert ens and all("margin" in v["checks"] for v in ens)
    accepted = [v for v in ens if v["outcome"] == "accepted"]
    assert accepted and all(v["qty"] <= v["checks"]["margin"]["max_qty"] + 1e-9 for v in accepted)
    assert any("margin" in " ".join(v["reasons"]) for v in ens)
