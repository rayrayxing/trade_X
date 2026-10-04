"""Broker protocol, the order guard and the agent account list."""
import pandas as pd
import pytest

from tradex.core.interfaces import BrokerPosition, OrderRequest
from tradex.core.ledger import Ledger
from tradex.core.records import Verdict
from tradex.execution.accounts import agent_account_ids, load_accounts
from tradex.execution.guard import GuardedBroker, OrderGuard, OrderRefused, VenueAdapter, ledger_verdicts
from tradex.execution.sim import SimBroker
from test_spine import ZERO_COST, _frames, _two_family_specs

T0 = pd.Timestamp("2026-03-02 21:00", tz="UTC")
DAY = pd.Timedelta(days=1)


def _entry(qty=10, verdict_id="d1-v", did="d1", **kw):
    return OrderRequest(f"{did}-entry", did, "X", "stocks", 1, qty, stop_loss=95.0, take_profit=110.0,
                        verdict_id=verdict_id, **kw)


# --- simulated broker satisfies the full protocol -------------------------------------------

def test_sim_reports_fills_status_and_account():
    br = SimBroker(10_000, ZERO_COST, bar=DAY, account_id="sim-1")
    br.place(_entry())
    assert br.order_status("d1-entry").status == "pending" and br.order_status("nope").status == "unknown"
    br.on_bar("X", T0, 100, 101, 99, 100)
    st = br.order_status("d1-entry")
    assert (st.status, st.filled_qty, st.avg_price) == ("filled", 10, 100)
    br.on_bar("X", T0 + DAY, 100, 111, 94, 105)                           # stop
    all_fills = br.fills()
    assert [f.reason for f in all_fills] == ["entry", "stop"]
    assert len({f.fill_id for f in all_fills}) == 2
    assert br.fills(since=T0 + DAY) == all_fills[1:]
    stop = all_fills[1]
    assert stop.position_closed and stop.net_pnl_usd == pytest.approx(-50)
    acct = br.account()
    assert (acct.account_id, acct.currency, acct.margin_used) == ("sim-1", "USD", 0.0)
    assert acct.equity == pytest.approx(9_950) and acct.buying_power == pytest.approx(9_950)


def test_sim_margin_and_cancel_and_expiry():
    br = SimBroker(10_000, ZERO_COST, bar=DAY)
    br.place(_entry())
    br.on_bar("X", T0, 100, 101, 99, 100)
    assert br.account().margin_used == pytest.approx(1_000)               # stocks: cash account, 100%
    br.place(OrderRequest("lim", "d2", "X", "stocks", 1, 1, "limit", 50.0, purpose="entry"))
    br.place(OrderRequest("lim2", "d3", "X", "stocks", 1, 1, "limit", 50.0, purpose="entry"))
    assert br.cancel("lim2") and br.order_status("lim2").status == "cancelled"
    br.on_bar("X", T0 + DAY, 100, 101, 99, 100)
    assert br.order_status("lim").status == "expired"


def test_sim_waits_for_a_missing_fx_rate_instead_of_guessing():
    def rate(ccy, ts):
        raise LookupError(ccy)
    br = SimBroker(10_000, ZERO_COST, bar=DAY, rate_fn=rate)
    br.place(OrderRequest("e", "d", "EUR_JPY", "forex", 1, 1000, stop_loss=150.0))
    br.on_bar("EUR_JPY", T0, 160, 161, 159, 160)
    assert br.order_status("e").status == "pending" and not br.positions()


# --- order guard ---------------------------------------------------------------------------

def _guard(verdicts=None, accounts=("acct-1",)):
    vs = {v.verdict_id: v for v in (verdicts or [Verdict("d1", T0.isoformat(), "accepted", 10, 50, 0.5, [], {},
                                                         verdict_id="d1-v")])}
    return OrderGuard(lambda did, vid: vs.get(vid), accounts)


def test_guard_passes_an_order_within_its_verdict():
    _guard().check(_entry(qty=10), "acct-1", [])


def test_guard_refuses_order_without_verdict():
    with pytest.raises(OrderRefused, match="no risk-gate verdict"):
        _guard().check(_entry(verdict_id=""), "acct-1", [])
    with pytest.raises(OrderRefused, match="does not exist"):
        _guard().check(_entry(verdict_id="d9-v"), "acct-1", [])


def test_guard_refuses_qty_over_verdict_and_rejected_verdicts():
    with pytest.raises(OrderRefused, match="exceeds verdict size"):
        _guard().check(_entry(qty=10.5), "acct-1", [])
    rejected = Verdict("d1", T0.isoformat(), "rejected", 0, 0, 0, ["no room"], {}, verdict_id="d1-v")
    with pytest.raises(OrderRefused, match="not accepted"):
        _guard([rejected]).check(_entry(qty=1), "acct-1", [])
    other = Verdict("d2", T0.isoformat(), "accepted", 10, 50, 0.5, [], {}, verdict_id="d1-v")
    with pytest.raises(OrderRefused, match="another decision"):
        _guard([other]).check(_entry(qty=1), "acct-1", [])


def test_guard_refuses_unknown_or_unresolved_account_and_rays_account():
    with pytest.raises(OrderRefused, match="not listed as agent-owned"):
        _guard().check(_entry(), "acct-2", [])
    with pytest.raises(OrderRefused, match="not listed as agent-owned"):
        _guard(accounts=("",)).check(_entry(), "", [])
    with pytest.raises(OrderRefused, match="Ray's own account"):
        _guard().check(_entry(account="ray"), "acct-1", [])


def test_guard_lets_exits_reduce_positions_only():
    pos = [BrokerPosition("d1", "X", "stocks", 1, 10, 100.0, T0, 95.0, None)]
    ex = lambda q, side=-1, did="d1": OrderRequest(f"{did}-x", did, "X", "stocks", side, q, purpose="exit")  # noqa: E731
    _guard().check(ex(10), "acct-1", pos)
    for bad in (ex(11), ex(5, side=1), ex(5, did="d7")):
        with pytest.raises(OrderRefused):
            _guard().check(bad, "acct-1", pos)


class _FakeVenue(VenueAdapter):
    venue = "fake"

    def __init__(self, guard):
        super().__init__("acct-1", guard)
        self.sim = SimBroker(10_000, ZERO_COST, bar=DAY, account_id="acct-1")

    def _submit(self, req):
        return self.sim.place(req)

    def cancel(self, coid): return self.sim.cancel(coid)
    def amend_stop(self, did, stop): self.sim.amend_stop(did, stop)
    def positions(self, account="agent"): return self.sim.positions(account)
    def fills(self, since=None): return self.sim.fills(since)
    def order_status(self, coid): return self.sim.order_status(coid)
    def account(self): return self.sim.account()


def test_venue_adapters_cannot_submit_around_the_guard():
    v = _FakeVenue(_guard())
    with pytest.raises(OrderRefused):
        v.place(_entry(qty=11))
    assert not v.sim.orders
    assert v.place(_entry(qty=10)) == "d1-entry"
    with pytest.raises(TypeError):
        _FakeVenue(None)
    g = GuardedBroker(SimBroker(10_000, ZERO_COST, bar=DAY, account_id="acct-9"), _guard())
    with pytest.raises(OrderRefused):
        g.place(_entry())


def test_every_replay_entry_order_cites_a_verdict_the_guard_accepts():
    from tradex.core.replay import run_replay
    frames = _frames()
    led = Ledger(":memory:")
    run_replay(_two_family_specs(), frames, led, frames["AAA"].index[260])
    guard = OrderGuard(ledger_verdicts(led), {"sim"})
    entries = [o for o in led.rows(kind="order") if o["purpose"] == "entry"]
    assert entries
    for o in entries:
        req = OrderRequest(o["client_order_id"], o["decision_id"], o["symbol"], "stocks", o["side"], o["qty"],
                           verdict_id=f"{o['decision_id']}-v")
        guard.check(req, "sim", [])


# --- accounts ------------------------------------------------------------------------------

def test_accounts_file_is_paper_only_and_holds_no_ids():
    accs = load_accounts(resolve=lambda name: None)
    assert {a.venue for a in accs} == {"oanda", "moomoo"}
    assert all(a.mode == "paper" and not a.account_id for a in accs)
    assert {a.secret for a in accs} == {"oanda_account_id", "moomoo_sim_account_id"}
    assert agent_account_ids(accs) == set()                               # unresolved: nothing is tradable


def test_accounts_resolve_ids_from_secrets_and_refuse_live(tmp_path):
    accs = load_accounts(resolve=lambda name: {"oanda_account_id": "acct-from-keychain"}.get(name))
    assert agent_account_ids(accs) == {"acct-from-keychain"}
    f = tmp_path / "a.yaml"
    f.write_text("accounts:\n  - {name: real, venue: oanda, mode: live, asset_classes: [forex]}\n")
    with pytest.raises(ValueError, match="not allowed"):
        load_accounts(f, resolve=lambda n: None)


def test_accounts_file_is_protected():
    from pathlib import Path
    assert "config/accounts.yaml" in Path("config/protected_paths.txt").read_text().split()
