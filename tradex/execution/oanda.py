"""Oanda v20 practice venue adapter (protected path).

Practice host only: every URL, including the page URLs Oanda hands back, passes
``tradex.data.oanda.check_host``, so this module cannot reach a live account, and the
adapter refuses an account whose mode is not ``paper`` / environment not ``practice``.

Orders (only through ``VenueAdapter.place``, i.e. behind the order guard):

- entries are MARKET (FOK, ``positionFill=OPEN_ONLY``, optional ``priceBound``) or LIMIT
  (GTC) orders with ``stopLossOnFill`` / ``takeProfitOnFill`` from the request, so the stop
  rests at Oanda from the moment of the fill; units are signed;
- exits are MARKET ``REDUCE_ONLY`` orders, so an exit can never open or flip a position;
- ``clientExtensions.id`` is the client order ID and ``tradeClientExtensions.id`` the
  decision ID (tag ``tradex``), so trades map back to decisions after a restart;
- idempotency: before every POST the order is looked up by ``@client_order_id``; after a
  timeout it is looked up again and only resubmitted when Oanda says it does not exist
  (Oanda also rejects a second order with the same client ID). A client ID that exists
  for a different instrument or size raises ``ClientIdCollision``, never "already done".

Fills come from the transaction list (``/transactions`` pages on the first poll, then
``/transactions/sinceid``): ORDER_FILL entries, stop-loss and take-profit fills, closes,
margin closeouts. DAILY_FINANCING is parsed into ``Financing`` records (``financing()``)
and is part of the net P&L of the closing fill (Oanda's trade ``financing`` total), so
carry reaches the ledger with the close. Amounts are in the account currency and are
converted to USD with ``to_usd``; a non-USD account without a rate raises, never guesses.

Errors are typed (``tradex.execution.errors``); the core turns them into Health faults.
HTTP is injected (``http``) so tests run without a network.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from tradex.core.interfaces import AccountInfo, BrokerFill, BrokerPosition, OrderRequest, OrderStatus
from tradex.costs.models import RateMissing
from tradex.data.oanda import REST_HOST, check_host
from tradex.data.opend import RateLimiter
from tradex.execution.accounts import AgentAccount
from tradex.execution.errors import (ClientIdCollision, OrderRejected, OrderUncertain, VenueAuthError, VenueError,
                                     VenueUnavailable, WrongEnvironment)
from tradex.execution.guard import OrderGuard, VenueAdapter

TAG = "tradex"
# (status, json body). Raises VenueUnavailable on timeouts / connection errors.
Http = Callable[[str, str, dict, dict | None], tuple[int, dict]]

STOP_REASONS = {"STOP_LOSS_ORDER": "stop", "GUARANTEED_STOP_LOSS_ORDER": "stop", "TRAILING_STOP_LOSS_ORDER": "stop",
                "TAKE_PROFIT_ORDER": "target", "MARKET_ORDER_MARGIN_CLOSEOUT": "margin_closeout"}


def _http(method: str, url: str, headers: dict, body: dict | None, timeout: float = 10.0) -> tuple[int, dict]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(check_host(url), data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        try:
            payload = json.loads(e.read().decode() or "{}")
        except ValueError:
            payload = {}
        return e.code, payload
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise VenueUnavailable(f"oanda {method} unreachable: {type(e).__name__}") from None   # no URL: it holds the account ID


def _f(x) -> float:
    return float(x) if x not in (None, "") else 0.0


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s).tz_convert("UTC")


@dataclass
class Financing:
    """Financing booked on one open trade (DAILY_FINANCING), in USD; negative is a cost."""
    fill_id: str
    decision_id: str
    time: pd.Timestamp
    symbol: str
    amount_usd: float


class OandaPracticeAdapter(VenueAdapter):
    venue = "oanda"

    def __init__(self, account_id: str, guard: OrderGuard, *, token: str | None = None,
                 account: AgentAccount | None = None, http: Http | None = None, host: str = REST_HOST,
                 to_usd: Callable[[str], float] | None = None, price_bound_pips: float | None = None,
                 fills_from: pd.Timestamp | None = None, limiter: RateLimiter | None = None,
                 sleep: Callable[[float], None] = time.sleep, retries: int = 3):
        if account is not None and (account.venue != "oanda" or account.mode != "paper"
                                    or account.environment != "practice"):
            raise WrongEnvironment(f"account {account.name!r} is not an Oanda paper/practice account")
        check_host(f"https://{host}/")                       # LiveHostRefused for anything but practice
        if not account_id:
            raise WrongEnvironment("no Oanda practice account ID (run trade-x setup)")
        super().__init__(account_id, guard)
        self._token = token
        self._http = http or _http
        self.base = f"https://{host}/v3/accounts/{account_id}"
        self.to_usd = to_usd
        self.price_bound_pips = price_bound_pips
        self.fills_from = fills_from if fills_from is not None else pd.Timestamp.now(tz="UTC")
        self.limiter = limiter or RateLimiter(n=20, window_s=1.0)   # Oanda allows 100/s per connection
        self.sleep, self.retries = sleep, retries
        self._instruments: dict[str, dict] = {}
        self._currency: str | None = None
        self._last_txn: str | None = None
        self._fills: list[BrokerFill] = []
        self._financing: list[Financing] = []
        self._trade_did: dict[str, str] = {}                 # Oanda trade ID -> decision ID
        self._reported: dict[str, float] = {}                # trade ID -> net P&L already reported on partials

    # --- HTTP ---------------------------------------------------------------------------

    def _headers(self) -> dict:
        if self._token is None:
            from tradex import secrets
            self._token = secrets.get("oanda_token")
        return {"Authorization": f"Bearer {self._token}", "Content-Type": "application/json",
                "Accept-Datetime-Format": "RFC3339"}

    def _call(self, method: str, url: str, body: dict | None = None, *, retry: bool = True,
              ok404: bool = False) -> dict | None:
        """One request; GETs and other idempotent calls retry on outages with backoff."""
        check_host(url)
        tries = self.retries if retry else 1
        for i in range(tries):
            self.limiter.wait()
            try:
                status, payload = self._http(method, url, self._headers(), body)
            except VenueUnavailable:
                if i + 1 >= tries:
                    raise
                self.sleep(2 ** i)
                continue
            if status == 429 or status >= 500:
                if i + 1 >= tries:
                    raise VenueUnavailable(f"oanda {method}: HTTP {status}")
                self.sleep(2 ** i)
                continue
            if status in (401, 403):
                raise VenueAuthError(f"oanda {method}: HTTP {status} (token or account permissions)")
            if status == 404 and ok404:
                return None
            if status >= 400:
                code = payload.get("errorCode") or (payload.get("orderRejectTransaction") or {}).get("rejectReason")
                msg = f"oanda {method}: HTTP {status} {code or ''} {payload.get('errorMessage', '')}".strip()
                raise (OrderRejected if method != "GET" else VenueError)(msg)
            return payload
        raise VenueUnavailable(f"oanda {method}: retries exhausted")    # pragma: no cover

    def _get(self, path: str, **q) -> dict:
        qs = f"?{urllib.parse.urlencode(q)}" if q else ""
        return self._call("GET", f"{self.base}{path}{qs}")

    # --- instruments and formatting -------------------------------------------------------

    def instrument(self, symbol: str) -> dict:
        if symbol not in self._instruments:
            got = self._get("/instruments", instruments=symbol).get("instruments", [])
            if not got:
                raise OrderRejected(f"{symbol}: not tradeable on this Oanda account")
            self._instruments[symbol] = got[0]
        return self._instruments[symbol]

    def _px(self, symbol: str, price: float) -> str:
        return f"{price:.{int(self.instrument(symbol).get('displayPrecision', 5))}f}"

    def _units(self, req: OrderRequest) -> str:
        prec = int(self.instrument(req.symbol).get("tradeUnitsPrecision", 0))
        q = round(req.qty, prec)
        if q <= 0:
            raise OrderRejected(f"{req.client_order_id}: qty {req.qty} rounds to zero units")
        return f"{req.side * q:.{prec}f}"

    # --- orders -------------------------------------------------------------------------

    def order_body(self, req: OrderRequest) -> dict:
        """The v20 order for ``req`` (public so tests can check it without a network)."""
        o: dict = {"instrument": req.symbol, "units": self._units(req),
                   "clientExtensions": {"id": req.client_order_id, "tag": TAG, "comment": req.purpose}}
        if req.purpose == "exit":
            o.update(type="MARKET", timeInForce="FOK", positionFill="REDUCE_ONLY")
            return {"order": o}
        if req.order_type == "limit":
            if req.limit_price is None:
                raise OrderRejected(f"{req.client_order_id}: limit order without a price")
            o.update(type="LIMIT", price=self._px(req.symbol, req.limit_price), timeInForce="GTC")
        else:
            o.update(type="MARKET", timeInForce="FOK")
            if self.price_bound_pips is not None:
                o["priceBound"] = self._price_bound(req)
        o["positionFill"] = "OPEN_ONLY"
        o["tradeClientExtensions"] = {"id": req.decision_id, "tag": TAG}
        if req.stop_loss is not None:
            o["stopLossOnFill"] = {"price": self._px(req.symbol, req.stop_loss), "timeInForce": "GTC"}
        if req.take_profit is not None:
            o["takeProfitOnFill"] = {"price": self._px(req.symbol, req.take_profit), "timeInForce": "GTC"}
        return {"order": o}

    def _price_bound(self, req: OrderRequest) -> str:
        """Worst acceptable fill: the live touch plus ``price_bound_pips`` (no quote, no order)."""
        prices = self._get("/pricing", instruments=req.symbol).get("prices", [])
        if not prices or not prices[0].get("bids") or not prices[0].get("asks"):
            raise RateMissing(f"no live Oanda price for {req.symbol}: priceBound unknown")
        p = prices[0]
        touch = _f(p["asks"][0]["price"]) if req.side > 0 else _f(p["bids"][0]["price"])
        pip = 10 ** int(self.instrument(req.symbol).get("pipLocation", -4))
        return self._px(req.symbol, touch + req.side * self.price_bound_pips * pip)

    def _lookup(self, coid: str) -> dict | None:
        got = self._call("GET", f"{self.base}/orders/@{urllib.parse.quote(coid, safe='')}", ok404=True)
        return None if got is None else got.get("order")

    def _same(self, order: dict, body: dict) -> bool:
        o = body["order"]
        return order.get("instrument") == o["instrument"] and _f(order.get("units")) == _f(o["units"])

    def _submit(self, req: OrderRequest) -> str:
        body = self.order_body(req)
        found = self._lookup(req.client_order_id)
        if found is not None:
            if not self._same(found, body):
                raise ClientIdCollision(f"{req.client_order_id}: client ID already used for another order")
            if found.get("state") == "CANCELLED":
                raise OrderRejected(f"{req.client_order_id}: this client ID was already cancelled at Oanda; "
                                    "a new attempt needs a new client ID")
            return req.client_order_id                     # idempotent: already at Oanda
        for _ in range(2):
            try:
                got = self._call("POST", f"{self.base}/orders", body, retry=False)
                if "orderCancelTransaction" in got and "orderFillTransaction" not in got:
                    why = got["orderCancelTransaction"].get("reason", "")
                    raise OrderRejected(f"{req.client_order_id}: Oanda cancelled the order on creation ({why})")
                return req.client_order_id
            except VenueUnavailable:
                try:
                    found = self._lookup(req.client_order_id)
                except VenueError as exc:
                    raise OrderUncertain(f"{req.client_order_id}: submit timed out and lookup failed "
                                         f"({type(exc).__name__}); not resubmitted") from None
                if found is not None:
                    return req.client_order_id
        raise VenueUnavailable(f"{req.client_order_id}: Oanda did not take the order after 2 attempts")

    def cancel(self, client_order_id: str) -> bool:
        got = self._call("PUT", f"{self.base}/orders/@{urllib.parse.quote(client_order_id, safe='')}/cancel",
                         ok404=True)
        return got is not None and "orderCancelTransaction" in got

    def amend_stop(self, decision_id: str, stop: float) -> None:
        """Replace the trade's dependent stop-loss order (the trade is found by its client ID)."""
        trade = self._trade(f"@{decision_id}")
        if trade is None or (trade.get("clientExtensions") or {}).get("tag") != TAG:
            raise VenueError(f"{decision_id}: no open agent trade to amend")
        self._call("PUT", f"{self.base}/trades/@{urllib.parse.quote(decision_id, safe='')}/orders",
                   {"stopLoss": {"price": self._px(trade["instrument"], stop), "timeInForce": "GTC"}}, retry=False)

    def order_status(self, client_order_id: str) -> OrderStatus:
        o = self._lookup(client_order_id)
        if o is None:
            return OrderStatus(client_order_id, "unknown")
        state = o.get("state")
        if state == "PENDING":
            return OrderStatus(client_order_id, "pending")
        if state in ("FILLED", "TRIGGERED"):
            tx = self._get(f"/transactions/{o['fillingTransactionID']}").get("transaction", {}) \
                if o.get("fillingTransactionID") else {}
            return OrderStatus(client_order_id, "filled", abs(_f(tx.get("units"))), _f(tx.get("price")) or None)
        if state == "CANCELLED":
            tx = self._get(f"/transactions/{o['cancellingTransactionID']}").get("transaction", {}) \
                if o.get("cancellingTransactionID") else {}
            reason = tx.get("reason", "")
            if reason == "CLIENT_REQUEST":
                return OrderStatus(client_order_id, "cancelled")
            return OrderStatus(client_order_id, "expired" if reason == "TIME_IN_FORCE_EXPIRED" else "rejected")
        return OrderStatus(client_order_id, "unknown")

    # --- positions and account --------------------------------------------------------------

    def _trade(self, spec: str) -> dict | None:
        got = self._call("GET", f"{self.base}/trades/{urllib.parse.quote(spec, safe='@')}", ok404=True)
        return None if got is None else got.get("trade")

    def positions(self, account: str | None = "agent") -> list[BrokerPosition]:
        out = []
        for t in self._get("/openTrades").get("trades", []):
            ext = t.get("clientExtensions") or {}
            mine = ext.get("tag") == TAG and bool(ext.get("id"))
            owner = "agent" if mine else "ray"                 # anything not opened by tradex is never traded
            if account is not None and owner != account:
                continue
            did = ext["id"] if mine else f"oanda-trade-{t['id']}"
            self._trade_did[t["id"]] = did
            units = _f(t.get("currentUnits"))
            sl, tp = t.get("stopLossOrder") or {}, t.get("takeProfitOrder") or {}
            out.append(BrokerPosition(did, t["instrument"], "forex", 1 if units > 0 else -1, abs(units),
                                      _f(t.get("price")), _ts(t["openTime"]),
                                      _f(sl["price"]) if sl.get("price") else None,
                                      _f(tp["price"]) if tp.get("price") else None, account=owner,
                                      meta={"venue": "oanda", "unrealized_pl": _f(t.get("unrealizedPL")),
                                            "financing": _f(t.get("financing"))}))
        return out

    def account(self) -> AccountInfo:
        a = self._get("/summary").get("account", {})
        self._currency = a.get("currency") or self._currency
        if not self._currency:
            raise VenueError("oanda account summary has no currency")
        return AccountInfo(self.account_id, self.venue, self._currency, _f(a.get("NAV")), _f(a.get("balance")),
                           _f(a.get("marginUsed")), _f(a.get("marginAvailable")))

    def _usd(self, amount: float) -> float:
        if self._currency is None:
            self.account()
        if self._currency == "USD" or amount == 0.0:
            return amount
        if self.to_usd is None:
            raise RateMissing(f"no rate to convert Oanda {self._currency} amounts to USD")
        return amount * self.to_usd(self._currency)

    # --- transactions: fills and financing --------------------------------------------------

    def _sync(self) -> None:
        if self._last_txn is None:
            got = self._get("/transactions", **{"from": self.fills_from.isoformat(),
                                                "to": pd.Timestamp.now(tz="UTC").isoformat(),
                                                "type": "ORDER_FILL,DAILY_FINANCING"})
            txns = []
            for page in got.get("pages", []):
                txns += self._call("GET", page).get("transactions", [])      # page URLs pass check_host too
            last = got.get("lastTransactionID")
        else:
            got = self._get("/transactions/sinceid", id=self._last_txn)
            txns, last = got.get("transactions", []), got.get("lastTransactionID")
        for tx in sorted(txns, key=lambda t: int(t["id"])):
            if tx.get("type") == "ORDER_FILL":
                self._fills += self.parse_fill(tx)
            elif tx.get("type") == "DAILY_FINANCING":
                self._financing += self.parse_financing(tx)
        if last:
            self._last_txn = str(last)
        elif txns:
            self._last_txn = str(max(int(t["id"]) for t in txns))

    def _did(self, trade_id: str) -> str:
        if trade_id not in self._trade_did:
            t = self._trade(trade_id) or {}
            ext = t.get("clientExtensions") or {}
            self._trade_did[trade_id] = ext["id"] if ext.get("tag") == TAG and ext.get("id") else f"oanda-trade-{trade_id}"
        return self._trade_did[trade_id]

    def parse_fill(self, tx: dict) -> list[BrokerFill]:
        """One BrokerFill per trade the ORDER_FILL opened, reduced or closed."""
        t, sym, reason = _ts(tx["time"]), tx["instrument"], tx.get("reason", "")
        fee = _f(tx.get("commission")) + _f(tx.get("guaranteedExecutionFee"))
        units_total = abs(_f(tx.get("units"))) or 1.0
        coid = tx.get("clientOrderID") or ""
        pieces = []
        if tx.get("tradeOpened"):
            pieces.append(("open", tx["tradeOpened"]))
        pieces += [("close", c) for c in tx.get("tradesClosed") or []]
        if tx.get("tradeReduced"):
            pieces.append(("reduce", tx["tradeReduced"]))
        out = []
        for kind, p in pieces:
            u = _f(p.get("units"))
            q = abs(u)
            share = q / units_total
            side = 1 if u > 0 else -1
            if kind == "open":
                ext = p.get("clientExtensions") or {}
                did = ext["id"] if ext.get("tag") == TAG and ext.get("id") else f"oanda-trade-{p['tradeID']}"
                self._trade_did[p["tradeID"]] = did
                out.append(BrokerFill(coid or f"{did}-entry", did, t, sym, side, q, _f(p.get("price") or tx.get("price")),
                                      self._usd(fee * share), self._usd(abs(_f(p.get("halfSpreadCost")))), "entry",
                                      fill_id=f"oanda-{tx['id']}-{p['tradeID']}"))
                continue
            did = self._did(p["tradeID"])
            why = STOP_REASONS.get(reason, "exit")
            closed = kind == "close"
            piece_net = _f(p.get("realizedPL")) + _f(p.get("financing")) - fee * share
            if closed:
                trade = self._trade(p["tradeID"]) or {}
                if trade.get("state") == "CLOSED" and "realizedPL" in trade:
                    # the trade's own totals include every daily financing charge it paid
                    total = _f(trade["realizedPL"]) + _f(trade.get("financing")) - fee * share
                    piece_net = total - self._reported.pop(p["tradeID"], 0.0)
            else:
                self._reported[p["tradeID"]] = self._reported.get(p["tradeID"], 0.0) + piece_net
            out.append(BrokerFill(coid or f"{did}-{why}", did, t, sym, side, q, _f(p.get("price") or tx.get("price")),
                                  self._usd(fee * share), self._usd(abs(_f(p.get("halfSpreadCost")))), why,
                                  fill_id=f"oanda-{tx['id']}-{p['tradeID']}", net_pnl_usd=self._usd(piece_net),
                                  position_closed=closed))
        return out

    def parse_financing(self, tx: dict) -> list[Financing]:
        t, out = _ts(tx["time"]), []
        for pf in tx.get("positionFinancings") or []:
            for tf in pf.get("openTradeFinancings") or []:
                out.append(Financing(f"oanda-{tx['id']}-{tf['tradeID']}", self._did(tf["tradeID"]), t,
                                     pf.get("instrument", ""), self._usd(_f(tf.get("financing")))))
        return out

    def fills(self, since: pd.Timestamp | None = None) -> list[BrokerFill]:
        self._sync()
        return [f for f in self._fills if since is None or f.time >= since]

    def financing(self, since: pd.Timestamp | None = None) -> list[Financing]:
        """Daily financing per open trade, for the ledger (already inside the close's net P&L)."""
        self._sync()
        return [f for f in self._financing if since is None or f.time >= since]


def from_accounts(accounts: list[AgentAccount], guard: OrderGuard, **kw) -> OandaPracticeAdapter:
    """The adapter for the single Oanda practice account in config/accounts.yaml."""
    acc = [a for a in accounts if a.venue == "oanda"]
    if len(acc) != 1:
        raise WrongEnvironment(f"expected one Oanda account in config/accounts.yaml, found {len(acc)}")
    return OandaPracticeAdapter(acc[0].account_id, guard, account=acc[0], **kw)
