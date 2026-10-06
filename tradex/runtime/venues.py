"""A book spread over several venue accounts, and the per-venue margin headroom check.

The live ensemble book trades forex at Oanda and stocks at moomoo. ``MultiVenueBook``
routes each order to the venue for its asset class and presents one Broker to the core.
Its equity is the sum of the venue account equities converted to USD with rates from a
rate source at the current time; a missing rate raises (paper/live never guess).

Venues that book financing (``financing(since)``: Oanda's daily financing) are read
through ``financing_records``, which turns their entries into ledger ``Financing`` rows
labelled with the venue, the agent account's name and the account currency.
"""
from __future__ import annotations

import pandas as pd

from tradex.core.interfaces import AccountInfo, Broker, BrokerFill, BrokerPosition, Clock, OrderRequest, OrderStatus
from tradex.core.records import Financing
from tradex.runtime.fx import RateSource


class MultiVenueBook:
    def __init__(self, venues: dict[str, Broker], rates: RateSource, clock: Clock,
                 accounts: dict[str, str] | None = None):
        self.venues = venues                                  # asset class -> venue broker
        self.rates, self.clock = rates, clock
        self.accounts = dict(accounts or {})                  # asset class -> agent account name (no IDs)
        self._ccy: dict[str, str] = {}
        self._by_order: dict[str, str] = {}
        self._by_decision: dict[str, str] = {}

    @property
    def simulated(self) -> bool:
        return any(getattr(v, "simulated", False) for v in self.venues.values())

    def on_bar(self, symbol, ts, o, h, l, c):
        """Feed simulated venues (tests and dry runs); real venues fill on their own."""
        out = []
        for v in self.venues.values():
            if getattr(v, "simulated", False):
                out += v.on_bar(symbol, ts, o, h, l, c)
        return out

    def venue_for(self, asset_class: str) -> Broker:
        try:
            return self.venues[asset_class]
        except KeyError:
            raise KeyError(f"no venue for {asset_class}") from None

    def place(self, req: OrderRequest) -> str:
        coid = self.venue_for(req.asset_class).place(req)
        self._by_order[req.client_order_id] = req.asset_class
        self._by_decision.setdefault(req.decision_id, req.asset_class)
        return coid

    def cancel(self, client_order_id: str) -> bool:
        ac = self._by_order.get(client_order_id)
        return self.venues[ac].cancel(client_order_id) if ac else False

    def amend_stop(self, decision_id: str, stop: float) -> None:
        ac = self._by_decision.get(decision_id)
        if ac is not None:
            self.venues[ac].amend_stop(decision_id, stop)

    def positions(self, account: str | None = "agent") -> list[BrokerPosition]:
        return [p for v in self.venues.values() for p in v.positions(account)]

    def fills(self, since: pd.Timestamp | None = None) -> list[BrokerFill]:
        out = [f for v in self.venues.values() for f in v.fills(since)]
        return sorted(out, key=lambda f: f.time)

    def financing_records(self, since: pd.Timestamp | None = None) -> list[Financing]:
        """Financing entries of every venue that books them, as ledger rows (book unset)."""
        out = []
        for ac, v in self.venues.items():
            fn = getattr(v, "financing", None)
            if fn is None:
                continue
            items = fn(since)
            if not items:
                continue
            if ac not in self._ccy:
                self._ccy[ac] = v.account().currency
            name = getattr(v, "venue", "") or type(v).__name__
            out += [financing_record(x, name, self.accounts.get(ac, name), self._ccy[ac]) for x in items]
        return sorted(out, key=lambda r: r.time)

    def order_status(self, client_order_id: str) -> OrderStatus:
        ac = self._by_order.get(client_order_id)
        return self.venues[ac].order_status(client_order_id) if ac else OrderStatus(client_order_id, "unknown")

    def account(self) -> AccountInfo:
        """All venue accounts as one, in USD at the rates of now."""
        now = self.clock.now()
        accts = [v.account() for v in self.venues.values()]
        r = [self.rates.usd_per_unit(a.currency, now) for a in accts]
        return AccountInfo("+".join(a.account_id for a in accts), "multi", "USD",
                           sum(a.equity * x for a, x in zip(accts, r)), sum(a.cash * x for a, x in zip(accts, r)),
                           sum(a.margin_used * x for a, x in zip(accts, r)),
                           sum(a.buying_power * x for a, x in zip(accts, r)))


def financing_record(x, venue: str, account: str, currency: str) -> Financing:
    """A venue adapter's financing entry (``fill_id``/``txn_id``, ``decision_id``, ``time``,
    ``symbol``, ``amount_usd``, optional ``amount`` in the account currency) as a ledger row."""
    amount = getattr(x, "amount", None)
    t = pd.Timestamp(x.time)
    return Financing(time=t.isoformat(), venue=venue, account=account, symbol=x.symbol, decision_id=x.decision_id,
                     amount=None if amount is None else float(amount),
                     currency=getattr(x, "currency", None) or currency, amount_usd=round(float(x.amount_usd), 6),
                     txn_id=str(getattr(x, "txn_id", None) or x.fill_id))


def margin_max_qty(venue: Broker, rates: RateSource, ts: pd.Timestamp, margin_rate: float,
                   unit_notional_usd: float) -> tuple[float, dict]:
    """Largest quantity the venue's free margin can carry, and the numbers behind it (no
    account ID: those stay in the Keychain, not the ledger)."""
    acct = venue.account()
    free = acct.buying_power * rates.usd_per_unit(acct.currency, ts)
    per_unit = margin_rate * unit_notional_usd
    q = free / per_unit if per_unit > 0 else float("inf")
    return max(0.0, q), {"venue": acct.venue, "free_margin_usd": round(free, 2), "margin_rate": margin_rate}
