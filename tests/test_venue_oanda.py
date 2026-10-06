"""Oanda practice adapter on a fake HTTP layer (no network)."""

import pandas as pd
import pytest

from tradex.core.interfaces import OrderRequest
from tradex.core.records import Verdict
from tradex.costs.models import RateMissing
from tradex.data.oanda import LiveHostRefused
from tradex.execution.accounts import AgentAccount
from tradex.execution.errors import ClientIdCollision, OrderRejected, OrderUncertain, VenueUnavailable, WrongEnvironment
from tradex.execution.guard import OrderGuard, OrderRefused
from tradex.execution.oanda import OandaPracticeAdapter

ACC = "101-000-TEST"
BASE = f"https://api-fxpractice.oanda.com/v3/accounts/{ACC}"
T0 = pd.Timestamp("2026-10-05 12:00", tz="UTC")
INSTR = {"name": "EUR_USD", "displayPrecision": 5, "tradeUnitsPrecision": 0, "pipLocation": -4}


class FakeHttp:
    """Routes (method, path-without-query) -> handler(body, query) -> (status, json)."""

    def __init__(self):
        self.calls = []
        self.routes = {}
        self.on("GET", "/instruments", lambda b, q: (200, {"instruments": [INSTR]}))
        self.on("GET", "/summary", lambda b, q: (200, {"account": {"currency": "USD", "NAV": "1000.5", "balance": "1000",
                                                                    "marginUsed": "20", "marginAvailable": "980.5"}}))

    def on(self, method, path, fn):
        self.routes[(method, path)] = fn

    def __call__(self, method, url, headers, body):
        assert url.startswith("https://api-fxpractice.oanda.com/")
        assert headers["Authorization"] == "Bearer tok"
        path, _, query = url.partition("?")
        path = path.removeprefix(BASE)
        self.calls.append((method, path, body))
        fn = self.routes.get((method, path))
        if fn is None:
            return 404, {"errorMessage": "not found"}
        return fn(body, query)

    def posts(self):
        return [c for c in self.calls if c[0] == "POST"]


def _guard(qty=1000):
    v = Verdict("2026-10-05-0001", T0.isoformat(), "accepted", qty, 10, 0.5, [], {}, verdict_id="v1")
    return OrderGuard(lambda did, vid: v if vid == "v1" else None, {ACC})


def _adapter(http=None, **kw):
    http = http or FakeHttp()
    http.routes.setdefault(("GET", "/openTrades"), lambda b, q: (200, {"trades": []}))
    return OandaPracticeAdapter(ACC, kw.pop("guard", _guard()), token="tok", http=http, sleep=lambda s: None,
                                fills_from=T0, **kw), http


def _entry(side=1, **kw):
    base = dict(stop_loss=1.09, take_profit=1.12, verdict_id="v1")
    base.update(kw)
    return OrderRequest("2026-10-05-0001-entry", "2026-10-05-0001", "EUR_USD", "forex", side, 1000, **base)


def test_bracket_body_has_stop_target_signed_units_and_client_ids():
    a, _ = _adapter()
    o = a.order_body(_entry(side=-1, stop_loss=1.123456, take_profit=1.08))["order"]
    assert o["type"] == "MARKET" and o["timeInForce"] == "FOK" and o["positionFill"] == "OPEN_ONLY"
    assert o["units"] == "-1000" and o["instrument"] == "EUR_USD"
    assert o["stopLossOnFill"] == {"price": "1.12346", "timeInForce": "GTC"}
    assert o["takeProfitOnFill"] == {"price": "1.08000", "timeInForce": "GTC"}
    assert o["clientExtensions"]["id"] == "2026-10-05-0001-entry" and o["clientExtensions"]["tag"] == "tradex"
    assert o["tradeClientExtensions"] == {"id": "2026-10-05-0001", "tag": "tradex"}
    assert "priceBound" not in o
    ex = a.order_body(OrderRequest("x1", "2026-10-05-0001", "EUR_USD", "forex", 1, 500, purpose="exit"))["order"]
    assert ex["positionFill"] == "REDUCE_ONLY" and ex["units"] == "500" and "stopLossOnFill" not in ex


def test_price_bound_comes_from_the_live_quote_or_blocks():
    a, http = _adapter(price_bound_pips=2)
    http.on("GET", "/pricing", lambda b, q: (200, {"prices": [{"bids": [{"price": "1.10000"}], "asks": [{"price": "1.10010"}]}]}))
    assert a.order_body(_entry())["order"]["priceBound"] == "1.10030"
    http.on("GET", "/pricing", lambda b, q: (200, {"prices": []}))
    with pytest.raises(RateMissing):
        a.order_body(_entry())


def test_place_goes_through_the_guard_and_posts_once():
    a, http = _adapter()
    http.on("POST", "/orders", lambda b, q: (201, {"orderFillTransaction": {"id": "5"}}))
    assert a.place(_entry()) == "2026-10-05-0001-entry"
    assert len(http.posts()) == 1 and http.posts()[0][2]["order"]["stopLossOnFill"]["price"] == "1.09000"
    with pytest.raises(OrderRefused, match="no risk-gate verdict"):
        a.place(_entry(verdict_id=""))
    with pytest.raises(OrderRefused, match="exceeds verdict"):
        a.place(OrderRequest("big", "2026-10-05-0001", "EUR_USD", "forex", 1, 5000, verdict_id="v1"))
    assert len(http.posts()) == 1                                      # refused orders never reach Oanda


def test_existing_client_id_is_not_submitted_again():
    a, http = _adapter()
    http.on("GET", "/orders/@2026-10-05-0001-entry",
            lambda b, q: (200, {"order": {"instrument": "EUR_USD", "units": "1000", "state": "FILLED"}}))
    http.on("POST", "/orders", lambda b, q: pytest.fail("double submit"))
    assert a.place(_entry()) == "2026-10-05-0001-entry"
    http.on("GET", "/orders/@2026-10-05-0001-entry",
            lambda b, q: (200, {"order": {"instrument": "GBP_USD", "units": "1000", "state": "FILLED"}}))
    with pytest.raises(ClientIdCollision):
        a.place(_entry())


def test_timeout_then_lookup_finds_the_order_so_no_second_post():
    a, http = _adapter()
    state = {"landed": False}

    def post(b, q):
        state["landed"] = True
        raise VenueUnavailable("timeout")
    http.on("POST", "/orders", post)
    http.on("GET", "/orders/@2026-10-05-0001-entry",
            lambda b, q: (200, {"order": {"instrument": "EUR_USD", "units": "1000", "state": "FILLED"}})
            if state["landed"] else (404, {}))
    assert a.place(_entry()) == "2026-10-05-0001-entry"
    assert len(http.posts()) == 1


def test_timeout_and_not_found_resubmits_once_and_lookup_failure_never_resubmits():
    a, http = _adapter()
    n = {"post": 0}

    def post(b, q):
        n["post"] += 1
        if n["post"] == 1:
            raise VenueUnavailable("timeout")
        return 201, {"orderFillTransaction": {"id": "6"}}
    http.on("POST", "/orders", post)
    assert a.place(_entry()) and n["post"] == 2

    b, http2 = _adapter()
    calls = {"get": 0}

    def get(bd, q):
        calls["get"] += 1
        if calls["get"] == 1:
            return 404, {}
        raise VenueUnavailable("down")
    http2.on("GET", "/orders/@2026-10-05-0001-entry", get)
    http2.on("POST", "/orders", lambda bd, q: (_ for _ in ()).throw(VenueUnavailable("timeout")))
    with pytest.raises(OrderUncertain):
        b.place(_entry())
    assert len(http2.posts()) == 1


def test_cancel_on_create_is_a_rejection():
    a, http = _adapter()
    http.on("POST", "/orders", lambda b, q: (201, {"orderCancelTransaction": {"reason": "INSUFFICIENT_MARGIN"}}))
    with pytest.raises(OrderRejected, match="INSUFFICIENT_MARGIN"):
        a.place(_entry())


def test_live_host_and_non_practice_accounts_are_refused():
    with pytest.raises(LiveHostRefused):
        OandaPracticeAdapter(ACC, _guard(), token="tok", http=FakeHttp(), host="api-fxtrade.oanda.com")
    live = AgentAccount("x", "oanda", "paper", ["forex"], ACC, environment="live")
    with pytest.raises(WrongEnvironment):
        OandaPracticeAdapter(ACC, _guard(), token="tok", http=FakeHttp(), account=live)
    with pytest.raises(WrongEnvironment):
        OandaPracticeAdapter("", _guard(), token="tok", http=FakeHttp())
    a, http = _adapter()
    http.on("GET", "/transactions", lambda b, q: (200, {"pages": [f"https://api-fxtrade.oanda.com/v3/accounts/{ACC}/x"],
                                                        "lastTransactionID": "9"}))
    with pytest.raises(LiveHostRefused):
        a.fills()


FILL_OPEN = {"type": "ORDER_FILL", "id": "10", "time": "2026-10-05T12:00:01.000000000Z", "orderID": "9",
             "clientOrderID": "2026-10-05-0001-entry", "instrument": "EUR_USD", "units": "1000", "price": "1.10000",
             "reason": "MARKET_ORDER", "pl": "0", "financing": "0", "commission": "0", "halfSpreadCost": "0.0500",
             "tradeOpened": {"tradeID": "11", "units": "1000", "price": "1.10000", "halfSpreadCost": "0.0500",
                             "clientExtensions": {"id": "2026-10-05-0001", "tag": "tradex"}}}
FINANCE = {"type": "DAILY_FINANCING", "id": "12", "time": "2026-10-05T21:00:00.000000000Z", "financing": "-0.0300",
           "positionFinancings": [{"instrument": "EUR_USD", "financing": "-0.0300",
                                   "openTradeFinancings": [{"tradeID": "11", "financing": "-0.0300"}]}]}
FILL_STOP = {"type": "ORDER_FILL", "id": "13", "time": "2026-10-06T08:00:00.000000000Z", "orderID": "14",
             "instrument": "EUR_USD", "units": "-1000", "price": "1.09000", "reason": "STOP_LOSS_ORDER",
             "pl": "-10.0000", "financing": "0", "commission": "0", "halfSpreadCost": "0.0500",
             "tradesClosed": [{"tradeID": "11", "units": "-1000", "price": "1.09000", "realizedPL": "-10.0000",
                               "financing": "0.0000", "halfSpreadCost": "0.0500"}]}


def test_fills_and_financing_parsed_from_transactions():
    a, http = _adapter()
    http.on("GET", "/transactions", lambda b, q: (200, {"pages": [f"{BASE}/transactions/idrange?from=1&to=12"],
                                                        "lastTransactionID": "12"}))
    http.on("GET", "/transactions/idrange", lambda b, q: (200, {"transactions": [FINANCE, FILL_OPEN]}))
    http.on("GET", "/transactions/sinceid", lambda b, q: (200, {"transactions": [FILL_STOP], "lastTransactionID": "13"})
            if "id=12" in q else (200, {"transactions": [], "lastTransactionID": "13"}))
    http.on("GET", "/trades/11", lambda b, q: (200, {"trade": {"id": "11", "state": "CLOSED", "realizedPL": "-10.0000",
                                                               "financing": "-0.0300", "clientExtensions":
                                                               {"id": "2026-10-05-0001", "tag": "tradex"}}}))
    first = a.fills()
    assert [(f.reason, f.decision_id, f.side, f.qty, f.price) for f in first] == \
        [("entry", "2026-10-05-0001", 1, 1000, 1.1)]
    assert first[0].spread_slippage_usd == pytest.approx(0.05) and first[0].fill_id == "oanda-10-11"
    fin = a.financing()
    assert [(x.decision_id, x.amount_usd, x.symbol) for x in fin] == [("2026-10-05-0001", -0.03, "EUR_USD")]
    allf = a.fills()
    stop = allf[-1]
    assert (stop.reason, stop.side, stop.position_closed, stop.client_order_id) == \
        ("stop", -1, True, "2026-10-05-0001-stop")
    assert stop.net_pnl_usd == pytest.approx(-10.03)                    # realised P&L plus the trade's financing
    assert a.fills(since=pd.Timestamp("2026-10-06", tz="UTC")) == [stop]
    assert len({f.fill_id for f in allf}) == 2


def test_non_usd_account_needs_a_rate():
    a, http = _adapter()
    http.on("GET", "/summary", lambda b, q: (200, {"account": {"currency": "SGD", "NAV": "1", "balance": "1",
                                                                "marginUsed": "0", "marginAvailable": "1"}}))
    with pytest.raises(RateMissing):
        a.parse_fill(FILL_OPEN)
    b, http2 = _adapter(to_usd=lambda ccy: 0.75)
    http2.routes[("GET", "/summary")] = http.routes[("GET", "/summary")]
    assert b.parse_fill(FILL_OPEN)[0].spread_slippage_usd == pytest.approx(0.0375)


def test_account_positions_status_cancel_and_amend():
    a, http = _adapter()
    acct = a.account()
    assert (acct.currency, acct.equity, acct.cash, acct.margin_used, acct.buying_power) == ("USD", 1000.5, 1000, 20, 980.5)
    http.on("GET", "/openTrades", lambda b, q: (200, {"trades": [
        {"id": "11", "instrument": "EUR_USD", "price": "1.1", "openTime": "2026-10-05T12:00:01Z", "currentUnits": "-1000",
         "clientExtensions": {"id": "2026-10-05-0001", "tag": "tradex"}, "stopLossOrder": {"price": "1.11"},
         "takeProfitOrder": {"price": "1.08"}},
        {"id": "20", "instrument": "USD_JPY", "price": "150", "openTime": "2026-10-05T12:00:01Z", "currentUnits": "10"}]}))
    ps = a.positions()
    assert [(p.decision_id, p.direction, p.qty, p.stop, p.take_profit) for p in ps] == \
        [("2026-10-05-0001", -1, 1000, 1.11, 1.08)]
    assert [p.account for p in a.positions(None)] == ["agent", "ray"]
    http.on("GET", "/orders/@c1", lambda b, q: (200, {"order": {"state": "FILLED", "fillingTransactionID": "10"}}))
    http.on("GET", "/transactions/10", lambda b, q: (200, {"transaction": FILL_OPEN}))
    st = a.order_status("c1")
    assert (st.status, st.filled_qty, st.avg_price) == ("filled", 1000, 1.1)
    assert a.order_status("nope").status == "unknown"
    http.on("GET", "/orders/@c2", lambda b, q: (200, {"order": {"state": "CANCELLED", "cancellingTransactionID": "15"}}))
    http.on("GET", "/transactions/15", lambda b, q: (200, {"transaction": {"reason": "CLIENT_REQUEST"}}))
    assert a.order_status("c2").status == "cancelled"
    assert a.cancel("nope") is False
    http.on("PUT", "/orders/@c3/cancel", lambda b, q: (200, {"orderCancelTransaction": {"id": "16"}}))
    assert a.cancel("c3") is True
    http.on("GET", "/trades/@2026-10-05-0001", lambda b, q: (200, {"trade": {
        "instrument": "EUR_USD", "clientExtensions": {"id": "2026-10-05-0001", "tag": "tradex"}}}))
    http.on("PUT", "/trades/@2026-10-05-0001/orders", lambda b, q: (200, {}))
    a.amend_stop("2026-10-05-0001", 1.105)
    assert http.calls[-1] == ("PUT", "/trades/@2026-10-05-0001/orders",
                              {"stopLoss": {"price": "1.10500", "timeInForce": "GTC"}})


def test_outages_retry_then_raise_typed():
    a, http = _adapter()
    http.on("GET", "/summary", lambda b, q: (503, {}))
    with pytest.raises(VenueUnavailable):
        a.account()
    assert sum(1 for c in http.calls if c[1] == "/summary") == 3
