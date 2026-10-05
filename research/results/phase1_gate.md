# Phase-1 gate run

Run 2026-10-04 20:33 UTC on real bars only (moomoo OpenD, qfq-adjusted; Oanda practice bid/ask candles for FX). Stage-3 thresholds from `config/gates/thresholds.yaml`: {'min_trades': 100, 'min_profit_factor': 1.2, 'min_dsr': 0.95, 'max_drawdown': -0.5, 'min_positive_folds': 0.5}.
Walk-forward: {'n_folds': 5, 'initial_train_frac': 0.4, 'anchored': True, 'grid_points': 4, 'max_trials': 48, 'min_train_trades': 10, 'seed': 0}. Engine: {'initial_equity': 10000.0, 'risk_pct': 1.0, 'max_leverage': 1.0}. Costs: moomoo SG fees and default spread/slippage for US stocks; for FX the spread Oanda quoted at each bar's open plus 0.2 pip slippage, and financing from the bundled policy-rate estimates.
US universes: screened at every walk-forward test-fold start (and yearly before the first) from a pool of 102 stocks (the S&P 100 as of Oct 2026 plus the first run's names) and 22 ETFs; top 30 stocks, 10 ETFs, or 30 of both, ranked by trailing 60-day median dollar volume using bars before the screen date only.

**0 of 17 strategies run pass the gate.** 6 need data, 13 are not built or not standalone.


## Strategies run

| Strategy | Catalog entry | Market | Universe | Bars used | OOS trades | PF | Sharpe | DSR (N) | Max DD | Folds + | Result | Failing rung |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `fx-donchian-breakout-h4` | Donchian channel breakout | FX (Oanda practice bid/ask) H4, 2016-10-04..2026-10-02 | 6 pairs named by the spec; 6 symbols ever in it | 93,330 | 879 | 0.824 | -0.581 | 0.008 (48) | -0.238 | 1/5 | **fail** | profit_factor |
| `stk-ema-pullback-swing` | EMA pullback swing (seed) | US stocks/ETFs (moomoo) D1, 2006-09-20..2026-10-02 | top 30 of 124 all by trailing 60-day median dollar volume, re-screened at 13 dates; 67 symbols ever in it | 321,059 | 2132 | 0.993 | -0.007 | 0.0 (32) | -0.367 | 1/5 | **fail** | profit_factor |
| `stk-macd-trend-candle` | MACD trend with candle confirmation (seed) | US stocks/ETFs (moomoo) D1, 2006-09-20..2026-10-02 | top 30 of 124 all by trailing 60-day median dollar volume, re-screened at 13 dates; 67 symbols ever in it | 321,059 | 300 | 0.859 | -0.332 | 0.032 (32) | -0.245 | 2/5 | **fail** | profit_factor |
| `fx-trend-pullback-engulfing` | Trend pullback with engulfing candle (seed) | FX (Oanda practice bid/ask) H1, 2016-10-04..2026-10-02 | 3 pairs named by the spec; 3 symbols ever in it | 186,557 | 1786 | 0.855 | -0.89 | 0.0 (16) | -0.321 | 0/5 | **fail** | profit_factor |
| `stk-rsi2-meanrev` | RSI(2) pullback in uptrend (seed) | US stocks/ETFs (moomoo) D1, 2006-09-20..2026-10-02 | top 10 of 22 etfs by trailing 60-day median dollar volume, re-screened at 13 dates; 17 symbols ever in it | 78,943 | 715 | 1.039 | 0.075 | 0.288 (32) | -0.18 | 3/5 | **fail** | profit_factor |
| `fx-rsi-range-reversion` | RSI range reversion (seed) | FX (Oanda practice bid/ask) H1, 2016-10-04..2026-10-02 | 3 pairs named by the spec; 3 symbols ever in it | 186,523 | 50 | 0.826 | -0.197 | 0.0 (16) | -0.017 | 2/5 | **fail** | trades |
| `stk-breakout-volume` | Breakout with volume (seed) | US stocks/ETFs (moomoo) D1, 2006-09-20..2026-10-02 | top 30 of 102 stocks by trailing 60-day median dollar volume, re-screened at 13 dates; 60 symbols ever in it | 288,168 | 658 | 1.392 | 0.738 | 0.904 (32) | -0.138 | 5/5 | **fail** | deflated_sharpe |
| `stk-double-bottom-breakout` | Double bottom breakout (seed) | US stocks/ETFs (moomoo) D1, 2006-09-20..2026-10-02 | top 30 of 102 stocks by trailing 60-day median dollar volume, re-screened at 13 dates; 60 symbols ever in it | 288,168 | 331 | 1.121 | 0.208 | 0.115 (32) | -0.128 | 3/5 | **fail** | profit_factor |
| `stk-52w-high-momentum` | 52-week-high momentum | US stocks/ETFs (moomoo) D1, 2006-09-20..2026-10-02 | top 30 of 102 stocks by trailing 60-day median dollar volume, re-screened at 13 dates; 60 symbols ever in it | 288,168 | 2136 | 1.091 | 0.28 | 0.549 (32) | -0.2 | 2/5 | **fail** | profit_factor |
| `stk-sector-relative-strength` | Relative strength vs sector | US stocks/ETFs (moomoo) D1, 2006-09-20..2026-10-02 | top 30 of 102 stocks by trailing 60-day median dollar volume, re-screened at 13 dates; 60 symbols ever in it | 288,168 | 2682 | 1.175 | 0.51 | 0.65 (32) | -0.298 | 5/5 | **fail** | profit_factor |
| `stk-pairs-zscore` | Pairs trading (distance and cointegration) | US stocks/ETFs (moomoo) D1, 2006-09-20..2026-10-02 | top 30 of 102 stocks by trailing 60-day median dollar volume, re-screened at 13 dates; 18 symbols ever in it | 89,558 | 784 | 0.765 | -1.2 | 0.0 (32) | -0.514 | 0/5 | **fail** | profit_factor |
| `stk-gap-and-go-h1` | Gap-and-go after news gap | US stocks/ETFs (moomoo) H1, 2018-09-28..2026-10-02 | top 30 of 80 stocks by trailing 60-day median dollar volume, re-screened at 9 dates; 55 symbols ever in it | 764,887 | 1301 | 0.761 | -1.335 | 0.0 (32) | -0.531 | 1/5 | **fail** | profit_factor |
| `etf-overnight-hold` | Overnight vs intraday return split | US stocks/ETFs (moomoo) H1, 2018-09-28..2026-10-02 | top 10 of 22 etfs by trailing 60-day median dollar volume, re-screened at 9 dates; 17 symbols ever in it | 238,658 | 2422 | 0.626 | -2.492 | 0.0 (8) | -0.76 | 0/5 | **fail** | profit_factor |
| `etf-ts-xs-momentum` | Joint time-series and cross-sectional strategy | US stocks/ETFs (moomoo) D1, 2006-09-20..2026-10-02 | top 10 of 22 etfs by trailing 60-day median dollar volume, re-screened at 13 dates; 17 symbols ever in it | 78,943 | 1032 | 1.252 | 0.511 | 0.842 (32) | -0.198 | 4/5 | **fail** | deflated_sharpe |
| `fx-ts-xs-momentum` | Joint time-series and cross-sectional strategy | FX (Oanda practice bid/ask) D1, 2016-10-03..2026-10-01 | 6 pairs named by the spec; 6 symbols ever in it | 15,596 | 1492 | 0.699 | -1.51 | 0.0 (16) | -0.391 | 0/5 | **fail** | profit_factor |
| `etf-intraday-momentum` | Intraday momentum (first half-hour predicts last) | US stocks/ETFs (moomoo) H1, 2018-09-28..2026-10-02 | top 10 of 22 etfs by trailing 60-day median dollar volume, re-screened at 9 dates; 17 symbols ever in it | 238,658 | 10645 | 0.217 | -14.081 | 0.0 (4) | -1.0 | 0/5 | **fail** | profit_factor |
| `stk-earnings-jump` | Earnings-day jump continuation | US stocks/ETFs (moomoo) D1, 2006-09-20..2026-10-02 | top 30 of 102 stocks by trailing 60-day median dollar volume, re-screened at 13 dates; 60 symbols ever in it | 288,168 | 230 | 1.267 | 0.41 | 0.69 (16) | -0.066 | 4/5 | **fail** | deflated_sharpe |

Failing reasons:

- `fx-donchian-breakout-h4`: profit factor 0.82 after costs (need 1.2); deflated Sharpe 0.01 (need 0.95); only 20% of test folds profitable
- `stk-ema-pullback-swing`: profit factor 0.99 after costs (need 1.2); deflated Sharpe 0.00 (need 0.95); only 20% of test folds profitable
- `stk-macd-trend-candle`: profit factor 0.86 after costs (need 1.2); deflated Sharpe 0.03 (need 0.95); only 40% of test folds profitable
- `fx-trend-pullback-engulfing`: profit factor 0.85 after costs (need 1.2); deflated Sharpe 0.00 (need 0.95); only 0% of test folds profitable
- `stk-rsi2-meanrev`: profit factor 1.04 after costs (need 1.2); deflated Sharpe 0.29 (need 0.95)
- `fx-rsi-range-reversion`: only 50 out-of-sample trades (need 100); profit factor 0.83 after costs (need 1.2); deflated Sharpe 0.00 (need 0.95); only 40% of test folds profitable
- `stk-breakout-volume`: deflated Sharpe 0.90 (need 0.95)
- `stk-double-bottom-breakout`: profit factor 1.12 after costs (need 1.2); deflated Sharpe 0.12 (need 0.95)
- `stk-52w-high-momentum`: profit factor 1.09 after costs (need 1.2); deflated Sharpe 0.55 (need 0.95); only 40% of test folds profitable
- `stk-sector-relative-strength`: profit factor 1.17 after costs (need 1.2); deflated Sharpe 0.65 (need 0.95)
- `stk-pairs-zscore`: profit factor 0.76 after costs (need 1.2); deflated Sharpe 0.00 (need 0.95); max drawdown -51% (limit -50%); only 0% of test folds profitable
- `stk-gap-and-go-h1`: profit factor 0.76 after costs (need 1.2); deflated Sharpe 0.00 (need 0.95); max drawdown -53% (limit -50%); only 20% of test folds profitable
- `etf-overnight-hold`: profit factor 0.63 after costs (need 1.2); deflated Sharpe 0.00 (need 0.95); max drawdown -76% (limit -50%); only 0% of test folds profitable
- `etf-ts-xs-momentum`: deflated Sharpe 0.84 (need 0.95)
- `fx-ts-xs-momentum`: profit factor 0.70 after costs (need 1.2); deflated Sharpe 0.00 (need 0.95); only 0% of test folds profitable
- `etf-intraday-momentum`: profit factor 0.22 after costs (need 1.2); deflated Sharpe 0.00 (need 0.95); max drawdown -100% (limit -50%); only 0% of test folds profitable
- `stk-earnings-jump`: deflated Sharpe 0.69 (need 0.95)

## Not run

| Catalog entry | Status in catalog | Result | Why |
|---|---|---|---|
| Deep momentum network (LSTM trained on Sharpe) | Build | not built | needs a trained LSTM model; no model training in phase 1 |
| Momentum Transformer with changepoints | Build | not built | needs a trained transformer model |
| Slow momentum with fast reversion (changepoint detection) | Build | not built | needs the Gaussian-process changepoint model |
| Dynamic momentum learning (adaptive lookback) | Build | not built | needs the adaptive-lookback learner |
| Learning-to-rank cross-sectional momentum | Build | not built | needs a trained ranking model and a point-in-time universe |
| Deep learning statistical arbitrage (residual portfolios) | Build | not built | needs a trained model and factor returns |
| Deep chart-pattern recognition (head and shoulders, triangles) | Build: fills catalog gaps | not built | needs a trained pattern model |
| Large-tick trend filter (trade trend only where tick size is large) | Build: filter for all trend strategies | not a standalone signal | a filter for trend strategies |
| Uncertainty-gated stock ranker (skip when model unsure) | Build: gate for every ranker | not a standalone signal | a gate on a ranker that does not exist yet |
| Hidden Markov regime allocation | Build: regime input for ensemble | not a standalone signal | regime input for the ensemble |
| Realised-covariance regime detection | Build | not a standalone signal | regime detector, an input to allocation |
| Regime-switching volatility forecast for sizing | Build: sizing input | not a standalone signal | sizing input |
| Long calls or puts on high-conviction plans | Build: execution style, not a signal | not a standalone signal | execution style; needs live option chains |
| Pre-earnings run-up and IV crush (options) | Build | needs data | option implied-volatility history and option prices (earnings dates now exist; OpenD F10 has IV only around each report) |
| Pre-FOMC announcement drift | Build | needs data | historical FOMC dates (data/calendar covers 2026-2027 only) |
| LLM news sentiment long-short | Build | needs data | news headline history |
| ChatGPT headline scoring | Build | needs data | news headline history |
| Short high borrow-fee, high short-interest names | Build | needs data | borrow-fee and short-interest history (not downloaded in phase 1) |
| Short-seller flow signal | Build | needs data | short-volume history (not downloaded in phase 1) |

## What changed vs the first run

- Stock and ETF strategies run on a point-in-time liquidity screen over a 102-stock, 22-ETF pool instead of the symbols each spec names (80 more OpenD symbols fetched; history quota 124 of 300 used).
- Earnings calendar wired in: `no_earnings_3d` blocks entries 3 days before a report and the position reviewer exits ahead of one. 102 stocks covered; reports by source {'opend': 4834, 'sec_8k_2.02': 3996, 'sec_10q_10k': 20} (OpenD = real release date and before/after-market timing; sec_8k_2.02 = SEC 8-K item 2.02 filing-date proxy; sec_10q_10k = 10-Q/10-K filing-date proxy, weaker).
- Engine: `exit.session_close` flattens intraday strategies at the 16:00 New York close (DST-aware; early closes too), so intraday momentum is now tested.
- FX: ten years of Oanda practice bid/ask H1/H4/D candles for nine pairs; FX seeds and the FX leg of the joint time-series/cross-sectional entry run with the spread quoted at each bar.
- New strategies: `etf-intraday-momentum`, `stk-earnings-jump`, `fx-ts-xs-momentum`.
- Trial ledger: the screened universe and the earnings filter are recorded as a new variant of each strategy, so every parameter set re-run here adds to the DSR's N.

| Strategy | First run (hand-picked universe, no earnings filter) | This run | Result change |
|---|---|---|---|
| `fx-donchian-breakout-h4` | not run | 879 trades, PF 0.824, Sharpe -0.581, DSR 0.008 (N 48), 6 symbols | new: **fail** |
| `stk-ema-pullback-swing` | 1075 trades, PF 1.333, Sharpe 0.713, DSR 0.453 (N 16), 9 symbols | 2132 trades, PF 0.993, Sharpe -0.007, DSR 0.0 (N 32), 67 symbols | fail -> **fail** |
| `stk-macd-trend-candle` | 93 trades, PF 1.044, Sharpe 0.03, DSR 0.292 (N 16), 9 symbols | 300 trades, PF 0.859, Sharpe -0.332, DSR 0.032 (N 32), 67 symbols | fail -> **fail** |
| `fx-trend-pullback-engulfing` | not run | 1786 trades, PF 0.855, Sharpe -0.89, DSR 0.0 (N 16), 3 symbols | new: **fail** |
| `stk-rsi2-meanrev` | 286 trades, PF 1.1, Sharpe 0.142, DSR 0.494 (N 16), 3 symbols | 715 trades, PF 1.039, Sharpe 0.075, DSR 0.288 (N 32), 17 symbols | fail -> **fail** |
| `fx-rsi-range-reversion` | not run | 50 trades, PF 0.826, Sharpe -0.197, DSR 0.0 (N 16), 3 symbols | new: **fail** |
| `stk-breakout-volume` | 177 trades, PF 1.907, Sharpe 0.829, DSR 0.99 (N 16), 5 symbols | 658 trades, PF 1.392, Sharpe 0.738, DSR 0.904 (N 32), 60 symbols | pass -> **fail** |
| `stk-double-bottom-breakout` | 301 trades, PF 1.467, Sharpe 0.677, DSR 0.925 (N 16), 7 symbols | 331 trades, PF 1.121, Sharpe 0.208, DSR 0.115 (N 32), 60 symbols | fail -> **fail** |
| `stk-52w-high-momentum` | 2044 trades, PF 1.141, Sharpe 0.414, DSR 0.834 (N 16), 29 symbols | 2136 trades, PF 1.091, Sharpe 0.28, DSR 0.549 (N 32), 60 symbols | fail -> **fail** |
| `stk-sector-relative-strength` | 2635 trades, PF 1.241, Sharpe 0.671, DSR 0.93 (N 16), 29 symbols | 2682 trades, PF 1.175, Sharpe 0.51, DSR 0.65 (N 32), 60 symbols | fail -> **fail** |
| `stk-pairs-zscore` | 1772 trades, PF 0.861, Sharpe -0.823, DSR 0.0 (N 16), 22 symbols | 784 trades, PF 0.765, Sharpe -1.2, DSR 0.0 (N 32), 18 symbols | fail -> **fail** |
| `stk-gap-and-go-h1` | 531 trades, PF 0.98, Sharpe -0.063, DSR 0.177 (N 16), 7 symbols | 1301 trades, PF 0.761, Sharpe -1.335, DSR 0.0 (N 32), 55 symbols | fail -> **fail** |
| `etf-overnight-hold` | 3670 trades, PF 0.618, Sharpe -2.444, DSR 0.0 (N 4), 13 symbols | 2422 trades, PF 0.626, Sharpe -2.492, DSR 0.0 (N 8), 17 symbols | fail -> **fail** |
| `etf-ts-xs-momentum` | 1457 trades, PF 1.037, Sharpe 0.133, DSR 0.543 (N 16), 15 symbols | 1032 trades, PF 1.252, Sharpe 0.511, DSR 0.842 (N 32), 17 symbols | fail -> **fail** |
| `fx-ts-xs-momentum` | not run | 1492 trades, PF 0.699, Sharpe -1.51, DSR 0.0 (N 16), 6 symbols | new: **fail** |
| `etf-intraday-momentum` | not run | 10645 trades, PF 0.217, Sharpe -14.081, DSR 0.0 (N 4), 17 symbols | new: **fail** |
| `stk-earnings-jump` | not run | 230 trades, PF 1.267, Sharpe 0.41, DSR 0.69 (N 16), 60 symbols | new: **fail** |

## Caveats

- Survivorship bias remains: the candidate pool is today's S&P 100 and today's ETFs. OpenD has no delisted names, so stocks that left the index or failed since 2006 can never be picked by the screen. Results on screened universes are less hand-picked than the first run, not survivorship-free.
- The S&P 100 list is the October 2026 membership as written in tradex/research/universe.py (checked against OpenD's S&P 500 plate); the screen ranks within it, but membership itself is today's.
- Earnings dates before OpenD's coverage (about 2013 on) are SEC filing dates, a proxy: they can lag the release by a day and their before/after-market timing is unknown, so both filters treat them as before the open (conservative). Coverage starts late for: ACN from 2009-10-01, BLK from 2014-04-17, XOM from 2014-01-30.
- FX financing uses the bundled central-bank policy-rate estimates (tradex/costs/policy_rates.csv) minus Oanda's admin fee; Oanda does not publish historical financing rates.
- FX pairs with CAD, CHF or NZD (USD_CAD, USD_CHF, NZD_USD) are left out of FX runs: the bundled policy-rate table has no history for those currencies, and financing is not guessed.
- FX seeds' no_high_impact_news_30m filter is inactive: there is no historical macro calendar before 2026.
- Stock spreads are the cost model's defaults (1 bp half-spread + 2 bp slippage), not measured quotes.
- Fixed moomoo fees (US$0.99 + 9% GST per order) weigh heavily at the US$10,000 test equity.
- OpenD history starts 2006-09 for daily bars and 2018-09 for 60-minute bars; Oanda history here is 10 years.
- DSR's N is every parameter set ever recorded for the strategy in the trial ledger, across all variants.
