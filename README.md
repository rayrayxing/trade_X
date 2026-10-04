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
| `tradex/decision` | Ensemble: family votes become one finalised trade plan (entry, stop, targets, time stop) before any risk review |
| `tradex/events.py` | Event calendar (central banks, CPI, NFP, earnings, forex weekend) with blackout windows |
| `tradex/execution` | Simulated broker (next-open fills, brackets, idempotent order IDs, read-only external holdings) and pre-trade short and sanity checks. Protected |
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
- **Stock fees not in the design doc** (settlement US$0.003/share, SEC and FINRA rates)
  are from memory and need checking against Moomoo's fee page.
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
- Event calendars (earnings, NFP/CPI/FOMC) are inputs; without them those filters are
  inactive and the report says so.
