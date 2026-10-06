"""Venue financing (Oanda daily financing) becomes Financing ledger rows, once per transaction."""
from dataclasses import dataclass

import pandas as pd

from test_spine import ZERO_COST, _frames, _two_family_specs
from tradex.core.interfaces import AccountInfo, ReplayClock
from tradex.core.ledger import Ledger
from tradex.core.loop import CoreConfig
from tradex.core.records import RECORD_TYPES, Financing
from tradex.core.replay import build_replay_core
from tradex.execution.sim import SimBroker
from tradex.notify.telegram import render
from tradex.runtime.fx import MissingRate, SeriesRates
from tradex.runtime.venues import MultiVenueBook

T0 = pd.Timestamp("2026-10-05 21:00", tz="UTC")


@dataclass
class _Charge:                       # the shape of tradex.execution.oanda.Financing
    fill_id: str
    decision_id: str
    time: pd.Timestamp
    symbol: str
    amount_usd: float


class _SgdVenue(SimBroker):
    """An Oanda-like venue: account kept in SGD, daily financing listed by ``financing()``."""
    def __init__(self):
        super().__init__(10_000, ZERO_COST)
        self.venue = "oanda"
        self.charges: list = []
        self.fail: Exception | None = None

    def account(self):
        a = super().account()
        return AccountInfo("101-xxx", "oanda", "SGD", a.equity, a.cash, a.margin_used, a.buying_power)

    def financing(self, since=None):
        if self.fail:
            raise self.fail
        return [c for c in self.charges if since is None or c.time >= since]


def _core(led, venue):
    frames = _frames()
    clock = ReplayClock(T0)
    core = build_replay_core(_two_family_specs(), frames, led, frames["AAA"].index[260], clock=clock,
                             cfg=CoreConfig(mode="paper"), use_es=False)
    core.brokers["ensemble"] = MultiVenueBook({"forex": venue}, SeriesRates({}, "paper"), clock,
                                              accounts={"forex": "oanda-practice"})
    return core


def test_financing_rows_are_written_once_per_transaction_and_survive_a_restart(tmp_path):
    led = Ledger(tmp_path / "l.sqlite", git_commit="t")
    venue = _SgdVenue()
    venue.charges = [_Charge("oanda-501-77", "2026-10-05-0003", T0 - pd.Timedelta(hours=1), "EUR_USD", -0.42)]
    core = _core(led, venue)
    core._begin(T0)
    core._begin(T0 + pd.Timedelta(hours=1))                  # the same charge again: not written twice
    rows = led.rows(kind="financing")
    assert len(rows) == 1
    r = rows[0]
    assert (r["venue"], r["account"], r["currency"], r["amount_usd"], r["txn_id"], r["book"]) == \
        ("oanda", "oanda-practice", "SGD", -0.42, "oanda-501-77", "ensemble")
    assert r["amount"] is None and r["decision_id"] == "2026-10-05-0003" and "101-xxx" not in str(r)
    assert led.why("2026-10-05-0003")[0]["kind"] == "financing"
    venue.charges.append(_Charge("oanda-502-77", "2026-10-05-0003", T0 + pd.Timedelta(hours=2), "EUR_USD", -0.40))
    core2 = _core(led, venue)                                 # restart on the same ledger
    core2._begin(T0 + pd.Timedelta(hours=3))
    assert [r["txn_id"] for r in led.rows(kind="financing")] == ["oanda-501-77", "oanda-502-77"]
    assert render("financing", led.rows(kind="financing")[0]) is None    # no Telegram alert
    assert isinstance(Financing.from_dict(rows[0]), RECORD_TYPES["financing"])


def test_account_currency_amount_is_kept_when_the_venue_reports_it():
    @dataclass
    class _Full(_Charge):
        amount: float = 0.0
        currency: str = ""
    led = Ledger(":memory:", git_commit="t")
    venue = _SgdVenue()
    venue.charges = [_Full("oanda-9-1", "d", T0, "USD_JPY", -0.75, amount=-1.0, currency="SGD")]
    _core(led, venue)._begin(T0)
    r = led.rows(kind="financing")[0]
    assert (r["amount"], r["currency"], r["amount_usd"]) == (-1.0, "SGD", -0.75)


def test_missing_rate_for_financing_is_a_fault_not_a_crash():
    led = Ledger(":memory:", git_commit="t")
    venue = _SgdVenue()
    venue.fail = MissingRate("no fresh Oanda quote to convert SGD to USD")
    _core(led, venue)._begin(T0)
    assert not led.rows(kind="financing")
    h = [x for x in led.rows(kind="health") if not x["ok"]]
    assert h and h[0]["check"] == "broker" and "SGD" in h[0]["detail"]
