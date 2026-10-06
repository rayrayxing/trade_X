"""moomoo SIMULATE adapter and the stop guardian on a fake OpenD trade context (no network)."""
import itertools

import pandas as pd
import pytest

from tradex.core.interfaces import OrderRequest
from tradex.core.records import Verdict
from tradex.execution.accounts import load_accounts
from tradex.execution.errors import ClientIdCollision, OrderRejected, VenueError, WrongEnvironment
from tradex.execution.guard import OrderGuard, OrderRefused
from tradex.execution.guardian import StopGuardian
from tradex.execution.moomoo import MoomooSimAdapter, auto_resolver, pick_sim_account

SIM_ID, REAL_ID = 1111, 9999
T0 = pd.Timestamp("2026-10-05 14:00", tz="UTC")
DID = "2026-10-05-0001"


class FakeCtx:
    """Just enough of OpenSecTradeContext. Records every call's trd_env."""

    def __init__(self, accounts=None):
        self.accounts = accounts if accounts is not None else [
            {"acc_id": REAL_ID, "trd_env": "REAL", "trdmarket_auth": ["HK", "US"], "acc_status": "ACTIVE"},
            {"acc_id": SIM_ID, "trd_env": "SIMULATE", "trdmarket_auth": ["US"], "acc_status": "ACTIVE"}]
        self.orders: list[dict] = []
        self.envs: list[str] = []
        self.placed: list[dict] = []
        self.timeout_next_place = False
        self.ids = itertools.count(100)

    def _seen(self, kw):
        self.envs.append(kw.get("trd_env"))
        assert kw.get("acc_id") == SIM_ID

    def get_acc_list(self):
        return 0, pd.DataFrame(self.accounts)

    def place_order(self, price, qty, code, trd_side, **kw):
        self._seen(kw)
        o = {"order_id": str(next(self.ids)), "code": code, "trd_side": trd_side, "order_type": kw["order_type"],
             "order_status": "SUBMITTED", "qty": qty, "price": price, "dealt_qty": 0.0, "dealt_avg_price": 0.0,
             "remark": kw.get("remark"), "updated_time": "2026-10-05 10:00:00.000", "create_time": "2026-10-05 10:00:00"}
        self.orders.append(o)
        self.placed.append(o)
        if self.timeout_next_place:
            self.timeout_next_place = False
            raise TimeoutError("socket")
        return 0, pd.DataFrame([{"order_id": o["order_id"]}])

    def order_list_query(self, order_id="", code="", **kw):
        self._seen(kw)
        rows = [o for o in self.orders if (not order_id or o["order_id"] == order_id) and (not code or o["code"] == code)]
        return 0, pd.DataFrame(rows, columns=["order_id", "code", "trd_side", "order_type", "order_status", "qty",
                                              "price", "dealt_qty", "dealt_avg_price", "remark", "updated_time",
                                              "create_time"])

    def modify_order(self, op, order_id, qty, price, **kw):
        self._seen(kw)
        o = next(o for o in self.orders if o["order_id"] == order_id)
        if op == "CANCEL":
            o["order_status"] = "CANCELLED_ALL"
        else:
            o["price"], o["qty"] = price, qty
        return 0, pd.DataFrame([{"order_id": order_id}])

    def position_list_query(self, **kw):
        self._seen(kw)
        pos: dict[str, float] = {}
        for o in self.orders:
            s = 1 if o["trd_side"] == "BUY" else -1
            pos[o["code"]] = pos.get(o["code"], 0.0) + s * o["dealt_qty"]
        rows = [{"code": c, "qty": abs(q), "position_side": "LONG" if q > 0 else "SHORT", "cost_price": 100.0}
                for c, q in pos.items() if q]
        return 0, pd.DataFrame(rows, columns=["code", "qty", "position_side", "cost_price"])

    def accinfo_query(self, **kw):
        self._seen(kw)
        return 0, pd.DataFrame([{"total_assets": 1_000_000.0, "cash": 990_000.0, "power": 2_000_000.0,
                                 "initial_margin": 10_000.0, "currency": "N/A"}])

    def fill(self, remark, px, qty=None):
        o = next(o for o in self.orders if o["remark"] == remark)
        o["dealt_qty"], o["dealt_avg_price"] = qty if qty is not None else o["qty"], px
        o["order_status"] = "FILLED_ALL" if o["dealt_qty"] >= o["qty"] else "FILLED_PART"
        o["updated_time"] = "2026-10-05 10:01:00.000"


def _guard(qty=10):
    v = Verdict(DID, T0.isoformat(), "accepted", qty, 10, 0.5, [], {}, verdict_id="v1")
    return OrderGuard(lambda did, vid: v if vid == "v1" else None, {str(SIM_ID)})


def _adapter(ctx=None, **kw):
    ctx = ctx or FakeCtx()
    return MoomooSimAdapter(str(SIM_ID), _guard(), ctx=ctx, clock=lambda: T0, **kw), ctx


def _entry(**kw):
    base = dict(stop_loss=95.0, take_profit=110.0, verdict_id="v1")
    base.update(kw)
    return OrderRequest(f"{DID}-entry", DID, "AAPL", "stocks", 1, 10, **base)


def test_real_env_and_real_account_are_refused():
    with pytest.raises(WrongEnvironment):
        MoomooSimAdapter(str(SIM_ID), _guard(), ctx=FakeCtx(), trd_env="REAL")
    a, ctx = _adapter()
    with pytest.raises(WrongEnvironment):
        a._call("accinfo_query", trd_env="REAL")
    a.trd_env = "REAL"                                       # even if someone flips it later
    with pytest.raises(WrongEnvironment):
        a.account()
    b = MoomooSimAdapter(str(REAL_ID), OrderGuard(lambda d, v: None, {str(REAL_ID)}), ctx=FakeCtx())
    with pytest.raises(WrongEnvironment, match="not a US SIMULATE"):
        b.account()
    assert set(ctx.envs) <= {"SIMULATE"}


def test_auto_select_picks_the_single_us_simulate_account():
    assert pick_sim_account(FakeCtx()) == str(SIM_ID)
    two = FakeCtx(FakeCtx().accounts + [{"acc_id": 2, "trd_env": "SIMULATE", "trdmarket_auth": ["US"]}])
    none = FakeCtx([{"acc_id": REAL_ID, "trd_env": "REAL", "trdmarket_auth": ["US"]},
                    {"acc_id": 3, "trd_env": "SIMULATE", "trdmarket_auth": ["HK"]}])
    for ctx in (two, none):
        with pytest.raises(WrongEnvironment):
            pick_sim_account(ctx)
    accs = load_accounts(resolve=lambda n: None, auto={"moomoo": auto_resolver(FakeCtx())})
    by = {a.venue: a for a in accs}
    assert by["moomoo"].account_id == str(SIM_ID) and by["oanda"].account_id == ""


def test_accounts_refuse_a_non_paper_environment(tmp_path):
    f = tmp_path / "a.yaml"
    f.write_text("accounts:\n  - {name: m, venue: moomoo, mode: paper, environment: REAL, asset_classes: [stocks]}\n")
    with pytest.raises(ValueError, match="not allowed"):
        load_accounts(f, resolve=lambda n: None)


def test_market_entry_goes_through_guard_with_remark_and_no_stop_order():
    a, ctx = _adapter()
    with pytest.raises(OrderRefused, match="no risk-gate verdict"):
        a.place(_entry(verdict_id=""))
    assert not ctx.placed
    assert a.place(_entry()) == f"{DID}-entry"
    (o,) = ctx.placed
    assert (o["order_type"], o["trd_side"], o["qty"], o["remark"], o["code"]) == ("MARKET", "BUY", 10, f"{DID}-entry", "US.AAPL")
    assert a.decisions[DID]["stop"] == 95.0 and a.stop_levels() == {}      # watched once filled
    for bad in (dict(order_type="stop"), dict(order_type="stop_limit")):
        with pytest.raises(OrderRejected):
            a.place(OrderRequest(f"{DID}-x", DID, "AAPL", "stocks", 1, 1, verdict_id="v1", **bad))
    assert set(ctx.envs) == {"SIMULATE"}


def test_resubmit_and_timeout_never_double_submit():
    a, ctx = _adapter()
    a.place(_entry())
    a.place(_entry())                                        # same client ID: found by remark, not sent again
    assert len(ctx.placed) == 1
    b, ctx2 = _adapter()
    ctx2.timeout_next_place = True                           # order lands, reply is lost
    assert b.place(_entry()) == f"{DID}-entry"
    assert len(ctx2.placed) == 1
    ctx3 = FakeCtx()
    ctx3.orders.append({"order_id": "1", "code": "US.MSFT", "trd_side": "BUY", "qty": 10, "remark": f"{DID}-entry",
                        "order_status": "FILLED_ALL", "dealt_qty": 10, "dealt_avg_price": 1})
    c, _ = _adapter(ctx3)
    with pytest.raises(ClientIdCollision):
        c.place(_entry())


def test_fills_positions_status_cancel_and_amend():
    a, ctx = _adapter()
    a.place(_entry())
    assert a.order_status(f"{DID}-entry").status == "pending"
    ctx.fill(f"{DID}-entry", 100.0, qty=4)
    ctx.fill(f"{DID}-entry", 101.5, qty=10)                  # cumulative avg 101.5 -> second slice at 102.5
    fs = a.fills()
    assert [(f.reason, f.qty) for f in fs] == [("entry", 10)]
    ctx2 = FakeCtx()
    b = MoomooSimAdapter(str(SIM_ID), _guard(), ctx=ctx2, clock=lambda: T0)
    b.place(_entry())
    ctx2.fill(f"{DID}-entry", 100.0, qty=4)
    b.fills()
    ctx2.fill(f"{DID}-entry", 101.5, qty=10)
    fs = b.fills()
    assert [(f.qty, round(f.price, 4)) for f in fs] == [(4, 100.0), (6, 102.5)]
    assert fs[0].time == pd.Timestamp("2026-10-05 14:01", tz="UTC")       # New York 10:01 -> UTC
    assert b.order_status(f"{DID}-entry").status == "filled"
    (p,) = b.positions()
    assert (p.decision_id, p.qty, p.direction, p.stop) == (DID, 10, 1, 95.0)
    assert round(p.entry_price, 4) == 101.5
    b.amend_stop(DID, 99.0)
    assert b.positions()[0].stop == 99.0
    acct = b.account()
    assert (acct.currency, acct.equity, acct.buying_power) == ("USD", 1_000_000.0, 990_000.0)
    lim = OrderRequest("2026-10-05-0002-entry", "2026-10-05-0002", "MSFT", "stocks", 1, 1, "limit", 50.0)
    b._submit(lim)
    b.amend_limit(lim.client_order_id, 51.0)
    assert ctx2.orders[-1]["price"] == 51.0
    assert b.cancel(lim.client_order_id) is True and b.cancel(lim.client_order_id) is False
    b.fills()
    assert "2026-10-05-0002" not in b.decisions                       # unfilled entry leaves no position behind


def test_positions_report_unknown_holdings_as_rays_and_mismatch_is_loud():
    a, ctx = _adapter()
    ctx.orders.append({"order_id": "1", "code": "US.TSLA", "trd_side": "BUY", "qty": 3, "remark": "manual",
                       "order_status": "FILLED_ALL", "dealt_qty": 3, "dealt_avg_price": 1})
    assert a.positions() == []
    (r,) = a.positions(None)
    assert (r.account, r.symbol, r.qty) == ("ray", "TSLA", 3)
    a.place(_entry())
    ctx.fill(f"{DID}-entry", 100.0)
    a.fills()
    ctx.orders[-1]["dealt_qty"] = 0                           # venue lost the shares
    with pytest.raises(VenueError, match="reconcile"):
        a.positions()


def test_state_survives_a_restart(tmp_path):
    path = tmp_path / "moomoo.json"
    a, ctx = _adapter(state_path=path)
    a.place(_entry())
    ctx.fill(f"{DID}-entry", 100.0)
    a.fills()
    b = MoomooSimAdapter(str(SIM_ID), _guard(), ctx=ctx, clock=lambda: T0, state_path=path)
    assert b.stop_levels()[DID]["stop"] == 95.0 and b.positions()[0].qty == 10


def _filled_position():
    a, ctx = _adapter()
    a.place(_entry())
    ctx.fill(f"{DID}-entry", 100.0)
    a.fills()
    return a, ctx


def test_guardian_fires_once_through_the_guard_and_closes():
    a, ctx = _filled_position()
    faults = []
    g = StopGuardian(a, lambda syms: {"AAPL": 96.0}, lambda c, m: faults.append(m), clock=lambda: T0)
    assert g.tick() == []                                    # above the stop
    g.marks = lambda syms: {"AAPL": 94.5}
    assert g.tick() == [f"{DID}-stop"]
    assert g.tick() == [] and g.tick() == []                 # at most once per position
    exits = [o for o in ctx.placed if o["remark"] == f"{DID}-stop"]
    assert len(exits) == 1 and exits[0]["trd_side"] == "SELL" and exits[0]["order_type"] == "MARKET"
    ctx.fill(f"{DID}-stop", 94.4)
    (f,) = [f for f in a.fills() if f.reason == "stop"]
    assert f.position_closed and f.net_pnl_usd == pytest.approx(-56.0)
    assert a.positions() == [] and a.stop_levels() == {}
    assert not faults and g.alive(T0 + pd.Timedelta(seconds=90)) and not g.alive(T0 + pd.Timedelta(minutes=5))


def test_guardian_target_rearms_after_a_rejected_exit_and_never_guesses_a_mark():
    a, ctx = _filled_position()
    faults = []
    g = StopGuardian(a, lambda syms: {}, lambda c, m: faults.append(m), interval_s=300, clock=lambda: T0)
    assert g.interval_s == 60                                 # pings at least every minute
    assert g.tick() == [] and "no live mark" in faults[0] and len(ctx.placed) == 1
    g.marks = lambda syms: {"AAPL": 111.0}
    assert g.tick() == [f"{DID}-target"]
    next(o for o in ctx.orders if o["remark"] == f"{DID}-target")["order_status"] = "FAILED"
    assert g.tick() == [f"{DID}-target-1"]                    # rejected exit re-arms with a new client ID
    assert g.tick() == []


def test_guardian_exit_is_still_guarded():
    a, ctx = _filled_position()
    faults = []
    a.guard = OrderGuard(lambda d, v: None, {"someone-else"})
    g = StopGuardian(a, lambda syms: {"AAPL": 90.0}, lambda c, m: faults.append(m), clock=lambda: T0)
    assert g.tick() == [] and "OrderRefused" in faults[0]
    assert not any(o["remark"].endswith("-stop") for o in ctx.placed)
