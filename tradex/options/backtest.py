"""Options backtester: long calls, long puts and naked short calls on injected chain data.

It follows the stock engine's discipline (``tradex.backtest.engine``):

- Signals on the underlying use closed bars only (the same ``compute_signals``); an entry
  decided at a bar's close fills at the NEXT bar's open, from the chain the provider holds
  for that open (``chains.chain(underlying, ts, "open")``).
- Exits are decided on the information of a bar's close (its OHLC and the closing quote) and
  fill at the next bar's open quote, so nothing is filled at a price that was not on screen.
  Premium stops fill no better than the stop level: a long stop fills at the lower of the open
  bid and the stop, a naked call's buy-to-close at the higher of the open ask and the stop. A
  gap past the stop is therefore paid in full, which is what the 20% gap sizing prepares for.
- A position that reaches its expiry bar settles at that bar's close (``settle_expiry``); a
  naked call that gets that far is a failed stop and is reported as such.
- Buys fill at the ask and sells at the bid plus the cost model's slippage, never at the mid.
  Fees come from ``MoomooOptionCosts`` (US$0.65 commission + US$0.30 platform per contract,
  9% GST on both, plus the regulatory pass-throughs).
- Sizing is premium-at-risk for longs and the gap-stressed loss for naked calls
  (``tradex.options.sizing``); naked calls also respect an initial-margin limit and the
  earnings / short-interest rules (``tradex.options.rules``) both at entry and while held.

Limits to keep in mind: premium stops and underlying stops are checked once per bar (a
spike that recovers inside the bar is not seen); the margin estimate is Reg-T style, not the
broker's; American early exercise is only handled through the ex-dividend rule; nothing here
fits the chain's IV surface, it consumes what the provider recorded.

Bars follow ``tradex.timeframes``: stamped with open time in UTC. Daily bars stamped at
00:00 UTC are read as that calendar date; other stamps are read in New York time.
"""
from __future__ import annotations

import datetime as dt
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from tradex.backtest import metrics
from tradex.costs.models import MoomooOptionCosts, MoomooStockCosts, model_for
from tradex.options.contract import NY, OptionPosition, OptionQuote
from tradex.options.payoff import early_assignment_risk, settle_expiry
from tradex.options.providers import (DividendProvider, OptionDataProvider, UnderlyingRiskProvider,
                                      require_real_option_data)
from tradex.options.rules import NakedCallPolicy, blocker_kind, naked_call_blockers, stop_problem
from tradex.options.sizing import (GAP_PCT_FLOOR, GapVol, MarginEstimator, RegTMarginEstimator, effective_naked_stop,
                                   long_premium_at_risk, order_fee_usd, size_long, size_naked_call,
                                   stop_premium_from_multiple)
from tradex.options.spec import OptionStrategySpec
from tradex.strategy.spec import SignalFrame, compute_signals
from tradex.timeframes import duration

BACKTEST_MODES = ("backtest", "replay")


@dataclass
class OptionBacktestConfig:
    initial_equity: float = 10_000.0
    risk_pct: float = 1.0                  # equity % at risk per trade, capped by the spec's cap_risk_pct
    max_positions: int = 5
    max_heat_pct: float = 12.0             # summed risk of open positions, % of equity
    max_premium_pct: float = 30.0          # premium paid on open long options, % of equity
    max_margin_pct: float = 50.0           # initial margin of open naked calls, % of equity
    allow_naked: bool = True
    gap_pct: float = GAP_PCT_FLOOR         # raised to the spec's value if that is larger; never below 20%
    rate: float = 0.0                      # risk-free rate used only to locate a naked call's stop level
    div_yield: float = 0.0
    long_itm_at_expiry: str = "sell_at_intrinsic"
    exercise_fee_per_contract: float | None = None     # None = unknown: not charged, and flagged
    assignment_fee_per_contract: float | None = None
    margin: MarginEstimator | None = None               # default RegTMarginEstimator (an estimate)
    policy: NakedCallPolicy | None = None               # default: thresholds from config/risk/policy.yaml
    stock_costs: MoomooStockCosts | None = None         # prices the share leg when a naked call is assigned
    mode: str = "backtest"
    start: pd.Timestamp | str | None = None
    end: pd.Timestamp | str | None = None


@dataclass
class OptionTrade:
    strategy_id: str
    symbol: str                            # the underlying
    contract: str                          # moomoo code
    structure: str
    direction: int                         # +1 long option, -1 short option
    entry_time: pd.Timestamp
    entry_price: float                     # premium per share, the fill
    exit_time: pd.Timestamp
    exit_price: float
    qty: int                               # contracts
    notional_usd: float                    # premium x multiplier x contracts
    risk_usd: float                        # premium at risk, or gap-stressed loss for a naked call
    gross_pnl: float                       # from fills, so spread and slippage are inside
    spread_slippage: float
    fees: float
    financing: float
    borrow: float
    margin_interest: float
    net_pnl: float
    r_multiple: float
    bars_held: int
    exit_reason: str
    entry_underlying: float
    exit_underlying: float
    entry_delta: float | None
    entry_iv: float | None
    dte_entry: int
    fee_unknown: bool = False
    pdt_violation: bool = False


@dataclass
class OptionBacktestResult:
    strategy_id: str
    params: dict
    trades: pd.DataFrame
    equity: pd.Series
    warnings: list[str]
    skipped: dict[str, int]
    config: OptionBacktestConfig

    def summary(self) -> dict:
        return metrics.summarize(self.equity, self.trades) | {"skipped": dict(self.skipped)}


# --- time helpers ------------------------------------------------------------------------

def bar_date(ts: pd.Timestamp) -> dt.date:
    """Calendar date of a bar: a 00:00 UTC stamp is that date, any other stamp is read in New York."""
    u = ts.tz_convert("UTC")
    if u.hour == 0 and u.minute == 0:
        return u.date()
    return u.tz_convert(NY).date()


def bar_moments(ts: pd.Timestamp, bar_td: pd.Timedelta) -> tuple[pd.Timestamp, pd.Timestamp]:
    """(open moment, close moment) of a bar. Daily and weekly bars run 09:30 to 16:00 New York."""
    if bar_td >= pd.Timedelta(days=1):
        d = bar_date(ts)
        last = d + dt.timedelta(days=0 if bar_td <= pd.Timedelta(days=1) else 4)
        o = pd.Timestamp(dt.datetime.combine(d, dt.time(9, 30))).tz_localize(NY).tz_convert("UTC")
        c = pd.Timestamp(dt.datetime.combine(last, dt.time(16, 0))).tz_localize(NY).tz_convert("UTC")
        return o, c
    return ts.tz_convert("UTC"), ts.tz_convert("UTC") + bar_td


# --- contract selection ------------------------------------------------------------------

# The order select_contract applies its filters in; the deepest one that rejected anything is the
# binding constraint (contracts that got furthest before failing).
_STAGES = ["no_contracts_of_right", "dte", "no_two_sided_quote", "min_bid", "spread", "liquidity", "no_iv", "iv_band",
           "no_delta", "strike_rule"]


@dataclass
class Selection:
    quote: OptionQuote | None
    rejected: Counter = field(default_factory=Counter)

    @property
    def binding(self) -> str:
        """The filter that stopped the most advanced candidates ('none' when nothing was rejected)."""
        hit = [k for k in _STAGES if self.rejected.get(k)]
        return hit[-1] if hit else "none"


def select_contract(chain, spec: OptionStrategySpec, spot: float, today: dt.date) -> Selection:
    """Pick the contract the spec's rules ask for from ``chain`` (quotes at the entry moment).

    Filters in order: right, days to expiry, a usable two-sided quote with an ask, the liquidity
    limits, the implied-volatility band, then the strike rule. Among those left, the contract
    nearest the strike target wins; ties go to higher open interest. ``rejected`` counts why
    contracts dropped out, so a run that never trades can say why.
    """
    o, rej = spec.option, Counter()
    best, best_key = None, None
    for q in chain:
        c = q.contract
        if c.right is not spec.right:
            continue
        dte = (c.expiry - today).days
        if not (o.dte.min <= dte <= o.dte.max):
            rej["dte"] += 1
            continue
        if not q.two_sided or q.ask is None or q.ask <= 0:
            rej["no_two_sided_quote"] += 1
            continue
        if q.bid < o.liquidity.min_bid:
            rej["min_bid"] += 1
            continue
        if q.spread_pct is None or q.spread_pct > o.liquidity.max_spread_pct:
            rej["spread"] += 1
            continue
        if (q.open_interest or 0) < o.liquidity.min_open_interest or (q.volume or 0) < o.liquidity.min_volume:
            rej["liquidity"] += 1
            continue
        iv = q.iv
        if (o.iv.min is not None or o.iv.max is not None) and iv is None:
            rej["no_iv"] += 1
            continue
        if (o.iv.min is not None and iv < o.iv.min) or (o.iv.max is not None and iv > o.iv.max):
            rej["iv_band"] += 1
            continue
        if o.strike.by == "delta":
            if q.delta is None:
                rej["no_delta"] += 1
                continue
            dist = abs(abs(q.delta) - o.strike.target)
        else:
            dist = abs(c.moneyness(spot) - o.strike.target)
        if dist > o.strike.tolerance:
            rej["strike_rule"] += 1
            continue
        key = (dist, -(q.open_interest or 0), c.expiry, c.strike)
        if best_key is None or key < best_key:
            best, best_key = q, key
    if best is None and not rej:
        rej["no_contracts_of_right"] += 1
    return Selection(best, rej)


# --- execution prices ------------------------------------------------------------------------

def exec_price(q: OptionQuote, side: int, costs: MoomooOptionCosts) -> tuple[float, float] | None:
    """(fill premium per share, spread+slippage paid per share) for a market order on ``q``.

    Buys fill at the ask and sells at the bid, each moved against us by the cost model's slippage
    (a share of the mid). Without a two-sided quote the model's half spread is applied to the last
    price instead; with nothing usable the order cannot fill (None).
    """
    if q.two_sided and (q.ask > 0 or side < 0):
        mid = 0.5 * (q.bid + q.ask)
        base = q.ask if side > 0 else q.bid
        fill = max(base + side * costs.slippage_pct * mid * costs.stress, 0.0)
        return fill, abs(fill - mid)
    mid = q.mid
    if mid is None or mid <= 0:
        return None
    fill, unit = costs.fill(q.contract.moomoo_code, side, mid, q.ts)
    fill = max(fill, 0.0)
    return fill, unit["spread"] + unit["slippage"]


@dataclass
class _Held:
    pos: OptionPosition
    structure: str
    risk_usd: float
    cash_in: float                         # +credit or -debit at entry, before fees
    entry_fees: float
    spread_slip: float
    dte_entry: int
    last_mark: float | None = None


def _policy_for_spec(base: NakedCallPolicy, spec: OptionStrategySpec) -> NakedCallPolicy:
    """The policy with the spec's naked-call limits applied: a spec may tighten them, never relax."""
    n = spec.option.naked
    out = NakedCallPolicy(**{k: getattr(base, k) for k in NakedCallPolicy.__dataclass_fields__})
    out.gap_pct = max(base.gap_pct, n.gap_pct, GAP_PCT_FLOOR)
    out.earnings_buffer_days = max(base.earnings_buffer_days, n.earnings_buffer_days)
    out.max_short_interest_pct_float = min(base.max_short_interest_pct_float, n.max_short_interest_pct_float)
    out.max_days_to_cover = min(base.max_days_to_cover, n.max_days_to_cover)
    return out


class _Run:
    """State and steps of one backtest run (kept in a class so each step is testable by reading)."""

    def __init__(self, spec, chains, risk, dividends, costs, cfg):
        self.spec, self.chains, self.risk, self.dividends, self.costs, self.cfg = spec, chains, risk, dividends, costs, cfg
        self.stock_costs = cfg.stock_costs or MoomooStockCosts()
        self.estimator = cfg.margin or RegTMarginEstimator()
        self.policy = _policy_for_spec(cfg.policy or NakedCallPolicy.from_policy(), spec)
        self.gap_pct = max(cfg.gap_pct, self.policy.gap_pct)
        self.bar_td = duration(spec.signal_tf)
        self.bar_days = max(self.bar_td / pd.Timedelta(days=1), 1.0)
        self.view = 1 if spec.entry_key == "long" else -1
        self.cash = cfg.initial_equity
        self.held: dict[str, _Held] = {}
        self.trades: list[OptionTrade] = []
        self.skipped: Counter = Counter()
        self.warnings: list[str] = []
        self.last_underlying: dict[str, float] = {}

    # --- book state ---------------------------------------------------------------------
    def equity(self) -> float:
        return self.cash + sum(h.pos.market_value(h.last_mark if h.last_mark is not None else h.pos.entry_premium)
                               for h in self.held.values())

    def margin_used(self) -> float:
        tot = 0.0
        for sym, h in self.held.items():
            if h.pos.contracts < 0:
                mark = h.last_mark if h.last_mark is not None else h.pos.entry_premium
                tot += self.estimator.initial_margin(h.pos.contract, h.pos.contracts, mark,
                                                     self.last_underlying.get(sym, h.pos.entry_underlying or h.pos.contract.strike))
        return tot

    def premium_outlay(self) -> float:
        return sum(h.pos.entry_premium * h.pos.shares for h in self.held.values() if h.pos.contracts > 0)

    def heat(self) -> float:
        return sum(h.risk_usd for h in self.held.values())

    # --- closing --------------------------------------------------------------------------
    def _record(self, sym: str, ts: pd.Timestamp, exit_premium: float, exit_underlying: float, reason: str,
                close_cash: float, exit_fees: float, exit_slip: float, fee_unknown: bool = False) -> None:
        h = self.held.pop(sym)
        p = h.pos
        self.cash += close_cash - exit_fees
        gross = h.cash_in + close_cash
        net = gross - h.entry_fees - exit_fees
        self.trades.append(OptionTrade(
            strategy_id=self.spec.id, symbol=sym, contract=p.contract.moomoo_code, structure=h.structure,
            direction=p.direction, entry_time=p.entry_time, entry_price=p.entry_premium, exit_time=ts,
            exit_price=exit_premium, qty=p.qty, notional_usd=p.entry_premium * p.contract.multiplier * p.qty,
            risk_usd=h.risk_usd, gross_pnl=gross, spread_slippage=h.spread_slip + exit_slip,
            fees=h.entry_fees + exit_fees, financing=0.0, borrow=0.0, margin_interest=0.0, net_pnl=net,
            r_multiple=net / h.risk_usd if h.risk_usd else 0.0, bars_held=p.bars_held, exit_reason=reason,
            entry_underlying=p.entry_underlying if p.entry_underlying is not None else float("nan"),
            exit_underlying=exit_underlying, entry_delta=p.entry_delta, entry_iv=p.entry_iv,
            dte_entry=h.dte_entry, fee_unknown=fee_unknown))

    def close_at_quote(self, sym: str, when: pd.Timestamp, q: OptionQuote, reason: str, exit_underlying: float) -> bool:
        """Close at the quote's bid/ask (plus slippage). False when the quote cannot fill the order."""
        p = self.held[sym].pos
        side = -p.direction
        px = exec_price(q, side, self.costs)
        if px is None:
            return False
        fill, unit = px
        if reason.startswith("stop_premium") and p.stop_premium is not None:
            # a stop fills at its level or worse, never better
            fill = min(fill, p.stop_premium) if p.contracts > 0 else max(fill, p.stop_premium)
        fees = order_fee_usd(self.costs, side, p.qty, fill, p.contract.moomoo_code, when)
        self._record(sym, when, fill, exit_underlying, reason, fill * p.shares, fees,
                     unit * p.qty * p.contract.multiplier)
        return True

    def settle_at_expiry(self, sym: str, when: pd.Timestamp, spot: float) -> None:
        h = self.held[sym]
        p = h.pos
        cfg = self.cfg
        intrinsic = p.contract.intrinsic(spot)
        sell_fees = order_fee_usd(self.costs, -1, p.qty, intrinsic, p.contract.moomoo_code, when) if p.is_long else None
        st = settle_expiry(p, spot, cfg.long_itm_at_expiry, sell_fees, cfg.exercise_fee_per_contract,
                           cfg.assignment_fee_per_contract)
        fees, slip = st.fees_usd, 0.0
        if st.kind == "expired_worthless":
            close_cash = 0.0
        elif st.kind == "sold_at_intrinsic":
            close_cash = intrinsic * p.shares
        else:
            # shares change hands at the strike, then are cleared at the close through the stock cost model
            sh = st.shares_delivered
            fill, unit = self.stock_costs.fill(p.contract.underlying, -1 if sh > 0 else 1, spot, when)
            fees += sum(self.stock_costs.order_fees(p.contract.underlying, -1 if sh > 0 else 1, abs(sh), fill, when).values())
            slip = (unit["spread"] + unit["slippage"]) * abs(sh)
            close_cash = st.stock_cash_usd + sh * fill
        unknown = st.fee_unknown and st.kind != "expired_worthless"
        if unknown:
            self.warnings.append(f"{sym}: {st.kind} fee not configured, not charged")
        if p.is_naked_call and st.kind != "expired_worthless":
            self.warnings.append(f"{sym}: naked call reached expiry in the money ({st.kind}): the stop did not hold")
        reason = {"expired_worthless": "expired_worthless", "sold_at_intrinsic": "expiry_itm",
                  "exercised": "exercised", "assigned": "assigned"}[st.kind]
        self._record(sym, when, intrinsic, spot, reason, close_cash, fees, slip, fee_unknown=unknown)

    def close_all(self, ts_last: pd.Timestamp) -> None:
        """Close what is open at the last bar's closing quote (reason ``end_of_data``)."""
        _, close_t = bar_moments(ts_last, self.bar_td)
        for sym in list(self.held):
            p = self.held[sym].pos
            q = self.chains.quote(p.contract, ts_last, "close")
            spot = self.last_underlying.get(sym, p.entry_underlying or 0.0)
            if q is not None and self.close_at_quote(sym, close_t, q, "end_of_data", spot):
                continue
            mark = self.held[sym].last_mark if self.held[sym].last_mark is not None else p.entry_premium
            self.warnings.append(f"{sym}: no closing quote at the end of data, closed at the last mark without costs")
            self._record(sym, close_t, mark, spot, "end_of_data", mark * p.shares, 0.0, 0.0, fee_unknown=True)

    # --- exits ------------------------------------------------------------------------------
    def exit_reason(self, sym: str, q: OptionQuote | None, today: dt.date, close_t: pd.Timestamp, high: float,
                    low: float, close: float, signal_exit: bool) -> str | None:
        """Why the position should be closed at the next open, decided from this bar's close."""
        hd = self.held[sym]
        p, spec = hd.pos, self.spec
        o = spec.option
        two = q is not None and q.two_sided
        if p.is_naked_call:
            ref = q.ask if two else (q.mid if q is not None else None)
            if ref is None:
                self.skipped["naked_stop_unobserved:no_close_quote"] += 1
            elif p.stop_premium is not None and ref >= p.stop_premium:
                return "stop_premium"
            if o.naked.use_underlying_stop and p.stop_underlying is not None and high >= p.stop_underlying:
                return "stop_underlying"
            horizon = today + dt.timedelta(days=self.policy.earnings_buffer_days)
            blockers = naked_call_blockers(self.policy, sym, close_t, horizon, self.risk)
            if blockers:
                return {"earnings": "earnings_window", "squeeze": "squeeze_risk"}.get(blocker_kind(blockers), "risk_data_unknown")
            mark = q.mid if q is not None and q.mid is not None else hd.last_mark
            if mark is not None:
                div = self.dividends.next_ex_dividend(sym, close_t) if self.dividends is not None else None
                days = (div.ex_date - today).days if div is not None else None
                if early_assignment_risk(p, close, mark, div.amount if div else None, days):
                    return "early_assignment_risk"
        else:
            if p.stop_premium is not None and two and q.bid <= p.stop_premium:
                return "stop_premium"
            if p.stop_underlying is not None and ((self.view > 0 and low <= p.stop_underlying)
                                                  or (self.view < 0 and high >= p.stop_underlying)):
                return "stop_underlying"
            tp = p.meta.get("target_premium")
            if tp is not None and two and q.bid >= tp:
                return "target_premium"
        if p.target_underlying is not None and ((self.view > 0 and high >= p.target_underlying)
                                                or (self.view < 0 and low <= p.target_underlying)):
            return "target_underlying"
        if (p.contract.expiry - today).days <= o.exit.close_dte:
            return "dte_close"
        if p.max_bars is not None and p.bars_held >= p.max_bars:
            return "time"
        if signal_exit:
            return "signal_exit"
        return None

    # --- entries ----------------------------------------------------------------------------
    def try_enter(self, sym: str, ts: pd.Timestamp, open_t: pd.Timestamp, today: dt.date, bar_open: float,
                  prev_atr: float) -> None:
        spec, cfg, costs = self.spec, self.cfg, self.costs
        sk = self.skipped
        if not (prev_atr > 0):
            sk["entry_skipped:no_atr"] += 1
            return
        if len(self.held) >= cfg.max_positions:
            sk["entry_skipped:max_positions"] += 1
            return
        chain = self.chains.chain(sym, ts, "open")
        if not chain:
            sk["entry_skipped:no_chain"] += 1
            return
        spot = next((qq.underlying_price for qq in chain if qq.underlying_price), None) or bar_open
        sel = select_contract(chain, spec, spot, today)
        if sel.quote is None:
            sk[f"entry_skipped:no_contract:{sel.binding}"] += 1
            return
        q = sel.quote
        c = q.contract
        side = spec.side
        px = exec_price(q, side, costs)
        if px is None:
            sk["entry_skipped:unfillable_quote"] += 1
            return
        fill, unit = px
        if fill <= 0:
            sk["entry_skipped:zero_price"] += 1
            return
        eq = self.equity()
        budget = eq * min(cfg.risk_pct, spec.cap_risk_pct) / 100.0
        stop_dist = spec.base.exit.stop_atr * prev_atr
        stop_u = spot - self.view * stop_dist
        target_u = spot + self.view * spec.base.exit.target_r * stop_dist
        meta: dict = {}
        if not spec.is_naked:
            n = size_long(budget, fill, costs, c.multiplier).contracts
            room = eq * cfg.max_premium_pct / 100.0 - self.premium_outlay()
            while n > 0 and (n * fill * c.multiplier > room or
                             n * fill * c.multiplier + order_fee_usd(costs, +1, n, fill, c.moomoo_code, ts) > self.cash):
                n -= 1
            if n < 1:
                sk["entry_skipped:no_room_long"] += 1
                return
            risk_usd = long_premium_at_risk(fill, n, costs, c.multiplier).max_loss_usd
            ps = spec.option.exit.premium_stop_pct
            stop_prem = fill * (1.0 - ps) if ps is not None else None
            if spec.option.exit.premium_target_mult is not None:
                meta["target_premium"] = fill * (1.0 + spec.option.exit.premium_target_mult)
            stop_under = stop_u
        else:
            credit = fill
            stop_prem = stop_premium_from_multiple(credit, spec.option.naked.stop_premium_mult)
            why = stop_problem(stop_prem, credit, self.policy)
            if why:
                sk["entry_blocked:stop"] += 1
                return
            horizon = min(c.expiry, today + dt.timedelta(days=math.ceil(spec.base.exit.max_bars * self.bar_days * 7 / 5) + 1))
            blockers = naked_call_blockers(self.policy, sym, open_t, horizon, self.risk)
            if blockers:
                sk["entry_blocked:" + blocker_kind(blockers)] += 1
                return
            use_u = spec.option.naked.use_underlying_stop
            eff = effective_naked_stop(c, credit, stop_prem, spot, q.iv, c.years_to_expiry(open_t),
                                       stop_u if use_u else None, cfg.rate, cfg.div_yield)
            if eff is None:
                sk["entry_blocked:stop_level_unlocatable"] += 1
                return
            s_stop, p_stop = eff
            free_margin = eq * cfg.max_margin_pct / 100.0 - self.margin_used()
            vol = GapVol(q.iv, c.years_to_expiry(open_t), cfg.rate, cfg.div_yield)
            size = size_naked_call(budget, c, credit, p_stop, s_stop, spot, self.gap_pct, costs, self.estimator,
                                   free_margin, vol)
            n = size.contracts
            if n < 1:
                sk["entry_skipped:no_room_naked"] += 1
                return
            risk_usd = size.risk_usd
            stop_under = stop_u if use_u else None
            meta.update(underlying_at_stop=s_stop, gap_pct=self.gap_pct, margin_usd=size.margin_usd)
        if self.heat() + risk_usd > eq * cfg.max_heat_pct / 100.0 + 1e-9:
            sk["entry_skipped:heat"] += 1
            return
        fees = order_fee_usd(costs, side, n, fill, c.moomoo_code, ts)
        pos = OptionPosition(c, side * n, fill, open_t, stop_premium=stop_prem, stop_underlying=stop_under,
                             target_underlying=target_u, entry_underlying=spot, entry_delta=q.delta, entry_iv=q.iv,
                             max_bars=spec.base.exit.max_bars, meta=meta)
        cash_in = -fill * pos.shares                                   # debit for a long, credit for a short
        self.cash += cash_in - fees
        self.held[sym] = _Held(pos, spec.structure, risk_usd, cash_in, fees, unit * n * c.multiplier,
                               (c.expiry - today).days, q.mid)

    # --- driver -----------------------------------------------------------------------------
    def on_bar(self, sym: str, s: dict, i: int, ts: pd.Timestamp, pending_entry: set, pending_exit: dict) -> None:
        o, h_, l_, c = s["o"][i], s["h"][i], s["l"][i], s["c"][i]
        open_t, close_t = bar_moments(ts, self.bar_td)
        today = bar_date(ts)

        # a. exits decided at the previous close fill at this open
        if sym in pending_exit:
            if sym in self.held:
                q = self.chains.quote(self.held[sym].pos.contract, ts, "open")
                if q is not None and self.close_at_quote(sym, open_t, q, pending_exit[sym], o):
                    pending_exit.pop(sym)
                else:
                    self.skipped["exit_delayed:no_fillable_quote_at_open"] += 1
            else:
                pending_exit.pop(sym, None)

        # b. entries decided at the previous close fill at this open
        if sym in pending_entry and sym not in self.held:
            pending_entry.discard(sym)
            prev_atr = s["atr"][i - 1] if i > 0 else float("nan")
            self.try_enter(sym, ts, open_t, today, o, prev_atr)

        # c. observations at the close
        if sym in self.held:
            hd = self.held[sym]
            p = hd.pos
            q = self.chains.quote(p.contract, ts, "close")
            if q is not None and q.mid is not None:
                hd.last_mark = q.mid
            else:
                self.skipped["mark_missing_at_close"] += 1
            p.bars_held += 1
            if today >= p.contract.expiry:
                self.settle_at_expiry(sym, close_t, c)
                pending_exit.pop(sym, None)
            elif sym not in pending_exit:
                reason = self.exit_reason(sym, q, today, close_t, h_, l_, c, bool(s["exit"][i]))
                if reason:
                    pending_exit[sym] = reason
        self.last_underlying[sym] = c

        # d. new entry signals at this close
        if sym not in self.held and s["trad"][i] and s["entry"][i] and (self.cfg.allow_naked or not self.spec.is_naked):
            pending_entry.add(sym)
        else:
            pending_entry.discard(sym)


def run_option_backtest(
    spec: OptionStrategySpec,
    data: dict[str, pd.DataFrame],
    chains: OptionDataProvider,
    risk: UnderlyingRiskProvider | None = None,
    dividends: DividendProvider | None = None,
    costs: MoomooOptionCosts | None = None,
    cfg: OptionBacktestConfig | None = None,
    tradable: dict[str, pd.Series] | None = None,
    filter_ctx: dict | None = None,
    params: dict | None = None,
    precomputed: dict[str, SignalFrame] | None = None,
) -> OptionBacktestResult:
    """Backtest ``spec`` over the underlying bars in ``data`` using ``chains`` for every option price.

    ``risk`` (earnings calendar and short interest) is required for a naked-call spec: without it
    every entry is blocked and the result says so. ``filter_ctx`` feeds the spec's entry filters
    (for example ``{"earnings": [...]}`` for ``no_earnings_3d``).
    """
    cfg = cfg or OptionBacktestConfig()
    if cfg.mode not in BACKTEST_MODES:
        raise ValueError(f"the options backtester runs in {BACKTEST_MODES}, not {cfg.mode!r}")
    require_real_option_data(cfg.mode, chains)
    if cfg.gap_pct < GAP_PCT_FLOOR - 1e-12:
        raise ValueError(f"gap_pct {cfg.gap_pct:.2f} is below the {GAP_PCT_FLOOR:.0%} floor")
    if params:
        spec = spec.with_params(params)
    errs = spec.validate()
    if errs:
        raise ValueError("options spec failed its schema check:\n  " + "\n  ".join(errs))
    run = _Run(spec, chains, risk, dividends, costs or model_for("options"), cfg)
    start = pd.Timestamp(cfg.start, tz="UTC") if isinstance(cfg.start, str) else cfg.start
    end = pd.Timestamp(cfg.end, tz="UTC") if isinstance(cfg.end, str) else cfg.end
    if spec.is_naked and risk is None:
        run.warnings.append("naked_call: no UnderlyingRiskProvider injected, so every entry is blocked")
    if spec.is_naked and dividends is None:
        run.warnings.append("naked_call: no DividendProvider injected, ex-dividend early-assignment risk is not checked")

    S: dict[str, dict] = {}
    for sym, bars in data.items():
        if bars.empty:
            continue
        sig = precomputed[sym] if precomputed and sym in precomputed else compute_signals(spec.base, bars, filter_ctx)
        run.warnings += [f"{sym}: {w}" for w in sig.warnings]
        trad = np.ones(len(bars), dtype=bool)
        if tradable is not None:
            m = tradable.get(sym)
            trad = np.zeros(len(bars), dtype=bool) if m is None else \
                m.reindex(bars.index, method="ffill").fillna(False).astype(bool).to_numpy()
        long_view = spec.entry_key == "long"
        S[sym] = dict(o=bars["open"].to_numpy(float), h=bars["high"].to_numpy(float), l=bars["low"].to_numpy(float),
                      c=bars["close"].to_numpy(float), idx=bars.index, atr=sig.atr.to_numpy(float), trad=trad,
                      entry=(sig.long_entry if long_view else sig.short_entry).to_numpy(bool),
                      exit=(sig.long_exit if long_view else sig.short_exit).to_numpy(bool))

    timeline: dict[pd.Timestamp, list[tuple[str, int]]] = defaultdict(list)
    for sym, s in S.items():
        for i, ts in enumerate(s["idx"]):
            if (start is None or ts >= start) and (end is None or ts < end):
                timeline[ts].append((sym, i))

    pending_entry: set[str] = set()
    pending_exit: dict[str, str] = {}
    eq_index, eq_values = [], []
    for ts in sorted(timeline):
        for sym, i in timeline[ts]:
            run.on_bar(sym, S[sym], i, ts, pending_entry, pending_exit)
        eq_index.append(ts)
        eq_values.append(run.equity())
    if eq_index:
        run.close_all(eq_index[-1])
        eq_values[-1] = run.cash

    tdf = pd.DataFrame([asdict(t) for t in run.trades])
    equity = pd.Series(eq_values, index=pd.DatetimeIndex(eq_index), name="equity", dtype=float)
    return OptionBacktestResult(spec.id, dict(params or {}), tdf, equity, list(dict.fromkeys(run.warnings)),
                                dict(run.skipped), cfg)
