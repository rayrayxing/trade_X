# trade_X: Level 4 Trading engine

Strategy registry, cost-aware backtester, walk-forward validation and the selection
engine for an autonomous US-stock and forex trading agent. Design background lives in
the thread-1 design doc ("Level 4 Trading: Strategy Library and Data Design").

Nothing in this repo places orders with a real broker. The trading core runs against a
simulated broker over recorded bars; data providers are read-only and take API keys from
environment variables on the machine that runs them.

## Layout

| Module | What it does |
|---|---|
| `tradex/data` | Bar conventions, CSV cache, read-only Alpaca / Massive / Oanda bar clients, synthetic bars for tests, and the IEX vs Massive feed comparison |
| `tradex/ta` | Feature registry: every TA-Lib function as `talib.NAME`, plus in-house stats, ATR zigzag pivots, zone touches, double tops/bottoms and a candlestick vote |
| `tradex/strategy` | Strategy YAML format, schema check (stage-2 gate), sandboxed rule expressions, look-ahead-safe multi-timeframe features, entry filters |
| `tradex/costs` | Moomoo SG stock costs (US$0.99 platform fee, settlement, SEC, FINRA TAF, spread, slippage, borrow, margin interest) and Oanda SG forex costs (spread, slippage, 17:00 NY financing with triple Friday) |
| `tradex/positions` | Position reviewer: every trade has a stop, a target and a time limit; closes stale, unreachable, rollover-negative or event-exposed positions; moves stops to breakeven or trails |
| `tradex/backtest` | Portfolio backtester, metrics (Sharpe, Sortino, drawdown, PSR, deflated Sharpe, Wilson lower bound), walk-forward validation and the stage-3 gate |
| `tradex/risk` | Fractional Kelly on the lower-bound win rate, block-bootstrap ruin simulation with stress cases, leverage gate, drawdown scaling |
| `tradex/selection` | Regime labels, strategy ranking, correlation clusters, risk-budget split and blending |
| `tradex/scout` | Daily market scout (10 to 20 stocks): working technical source, interfaces for news, chatter and a Claude headline reviewer, and replay of the scout in backtests |
| `tradex/pipeline.py` | Morning plan (scout, regime, allocation, universes) and the position-review pass, for thread 3 to schedule |
| `tradex/core` | The spine: record types, the hash-chained SQLite ledger, Clock / MarketData / Broker interfaces with replay versions, the trading core loop, the counterfactual ledger and the replay harness |
| `tradex/runtime` | Around the core: bar store (live appends, higher timeframes resampled only once closed), FX rate sources that refuse to guess in paper/live, incremental signals |
| `tradex/decision` | Ensemble: family votes become one finalised trade plan (entry, stop, targets, time stop) before any risk review |
| `tradex/events.py` | Event calendar (central banks, CPI, NFP, earnings, forex weekend) with blackout windows |
| `tradex/execution` | Simulated broker (next-open fills, brackets, idempotent order IDs, read-only external holdings), pre-trade short and sanity checks, and the order guard every venue adapter submits through (verdict ID, size within verdict, agent-owned account from `config/accounts.yaml`). Protected |
| `tradex/risk/exposure.py`, `gate.py` | Net open position per currency, expected shortfall with marginal charging, named stress replays, and the risk gate that sizes plans. Protected |
| `config/risk`, `config/gates` | Risk policy and promotion gate thresholds. Protected |
| `strategies/seeds` | 8 seed strategies (3 forex, 5 stocks) |

## Quick start

```bash
pip install -e '.[dev]'
pytest
python -m tradex check strategies

# on Ray's machine, with keys in the environment
export ALPACA_API_KEY_ID=... ALPACA_API_SECRET_KEY=... MASSIVE_API_KEY=... OANDA_API_TOKEN=...
python -m tradex fetch --provider alpaca --symbols SPY QQQ IWM --tf D1 --start 2016-01-01 --end 2026-10-01
python -m tradex validate strategies/seeds/stk-rsi2-meanrev.yaml --data data/cache/alpaca
python -m tradex select --reports reports --regime-bars data/cache/alpaca/SPY_D1.csv
python -m tradex compare-feeds --tf M15 --start 2025-01-01 --end 2026-09-30
```

## How a backtest stays honest

- Signals use closed bars; orders fill at the next bar's open. A test removes future bars
  and checks that no seed strategy's past signals change.
- Higher-timeframe features are only visible once their bar has closed.
- Inside a bar, if both stop and target are touched the stop wins; a gap through the stop
  fills at the open.
- Every fill pays half the spread plus slippage; every order pays fees; open positions pay
  financing, borrow and margin interest.
- Walk-forward: tune on the past, test once on the next unseen window, with a gap of one
  maximum holding period between them. Only test windows count.
- Deflated Sharpe discounts by the number of parameter sets tried.

Default gate (from the design doc, configurable in `Thresholds`): at least 100
out-of-sample trades, profit factor above 1.2 after costs, deflated Sharpe above 0.95,
drawdown no worse than 50%, at least half the test folds profitable.

## The trading spine

Every bar, in order:

1. **Votes.** Each active strategy reports long, short or flat with its own entry, stop and
   targets. Strategies not yet validated trade only in their own virtual book.
2. **Finalise.** Votes from at least 2 independent families (families whose signals
   correlate above 0.7 count as one) become one plan: farthest stop, nearest target as
   target 1, shortest time stop. The plan must clear 1.5:1 reward to risk after costs.
3. **Context vetoes.** Calendar blackouts, short-side checks (borrow, Rule 201, squeeze)
   and sanity checks can only block or shrink, never change the plan.
4. **Risk gate.** Sizes the plan from quarter Kelly down to whatever fits: book heat, per
   currency exposure, leverage tier, 97.5% expected shortfall (marginal), stress losses
   and position count. It never moves entry, stop or targets.

Every step writes to the ledger with a decision ID (`YYYY-MM-DD-NNNN`). Blocked plans are
followed to their would-be exit so each filter's cost is measurable.

The core runs on bar closes: `TradingCore.on_bar_close(tf, ts)` handles every bar of one
timeframe that closed at `ts`. Replay drives it from recorded bars; paper and live drive
it from the bar-close scheduler with the same code, so a replayed day must reproduce the
live ledger. Brokers are injected per book: the ensemble book gets the venue adapter
(behind the order guard), and virtual books stay simulated on the same live bars.
Signals are recomputed at each close on a rolling window of at least three lookbacks
(recursive indicators count 5 or 10 periods), which a test shows gives the same signals as
the full history on every seed. In paper and live a missing mark or FX rate blocks the
trade and writes a Health row; nothing is estimated.

In paper and live, `tradex/runtime/schedule.py` fires one close event per timeframe at
each bar boundary (finer timeframes first). Every close is claimed in the ledger's jobs
table, so it runs at most once, across restarts and crashes. After a sleep, the missed
closes of each timeframe collapse into one run of the latest close. The ensemble book can
span venues (`MultiVenueBook`, forex at Oanda and stocks at moomoo). Its equity is the sum
of the account equities converted to USD at live rates, and the risk gate also caps size
at the free margin of the venue the trade goes to.

```bash
python -m tradex replay --data data/cache/oanda --tf H4 --ledger runs/h4.sqlite --check-parity
python -m tradex why 2026-03-02-0007 --ledger runs/h4.sqlite
python -m tradex filters --ledger runs/h4.sqlite
python -m tradex verify-ledger --ledger runs/h4.sqlite
python -m tradex command pause --ledger runs/live.sqlite
```

### Protected paths

`config/protected_paths.txt` lists what agents may not change: risk, gate thresholds,
execution and the CI workflow itself. CI fails any `agent/` branch that touches them.
Human branches are not checked.

## Defaults chosen in this build

- **Day-trade cap off.** Ray reports moomoo SG does not apply the US 3-in-5 rule; it can be
  switched on per account with `EngineConfig(pdt=True)`.
- **Risk budget:** 12% of equity open risk across the book, 3% per strategy (5% once past
  paper), 6% per correlated cluster, capped by quarter-Kelly on the lower-bound win rate,
  halved at a 20% drawdown.
- **Correlation:** strategies whose daily returns correlate at 0.6 or more share one budget.
  With too little overlap to measure, strategies in the same asset class trading the same
  symbols are grouped (conservative).
- **Oanda admin fee:** the 2.5% is applied in full on both sides (conservative reading).
- **Historical forex rates:** `tradex/costs/policy_rates.csv`, written from memory and
  marked as an estimate; live trading reads Oanda's `financing` field.
- **Stock fees** are checked against moomoo SG's fee page and the SEC and FINRA notices
  (2026-10-04): US$0.99 platform fee plus 9% GST, settlement US$0.003/share capped at 1%,
  CAT, SEC US$20.60 per million on sales, FINRA TAF US$0.000195/share (max US$9.79). The
  TAF rises to US$0.000232 (max US$11.61) on 1 Jan 2027.
- **Oanda SG pricing:** modelled as spread-only. Core pricing (tighter spread plus
  commission) is offered, but Oanda's own pages disagree on the commission (US$30 vs
  US$40-50 per million), so it is not modelled until confirmed from the account. Retail
  leverage is capped at 20:1; the top risk tier uses 14:1. Oanda's order book and position
  book endpoints no longer work for v20 accounts, so nothing here relies on them.
- **Hard holding ceilings:** 30 days for stocks, 10 days for forex, on top of each
  strategy's own `max_bars`.

- **Spine:** start at leverage tier 2 (forex 5:1, stocks 1:1); currency cap 6% of equity;
  expected-shortfall budget 5%; worst named stress no worse than 35%. Hedging is off.
- **Win probability** in sizing is a base rate (0.40) until the probability model exists.

## Not done here

- One signal timeframe per replay run; mixed timeframes come with the runtime loop.
- Counterfactual entries use the plan's entry price, not a simulated fill.
- The drawdown governor holds a fixed tier; automatic tier changes are a follow-up.
- Stress scenario shocks are approximate and marked so in `config/risk/policy.yaml`.

- Running the IEX vs Massive comparison needs Ray's API keys, so it runs on his machine.
- Only 8 of the ~20 seed strategies; more chart patterns (head and shoulders, triangles,
  flags, wedges) are still to add to the registry.
- News, chatter and the Claude headline reviewer are interfaces; thread 3 connects them.
- `data/calendar` ships FOMC, BoJ and ECB decision days for 2026-2027 and the remaining
  2026 CPI dates, with sources. Jobs-report dates, 2027 CPI and earnings dates still need
  importing (BLS publishes an iCalendar feed; Alpha Vantage's free key has an earnings
  calendar). Without them those filters are inactive and the report says so.
