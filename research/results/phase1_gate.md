# Phase-1 gate run

Run 2026-10-05 07:53 UTC on real bars only (moomoo OpenD, qfq-adjusted; Oanda practice bid/ask candles for FX). Stage-3 thresholds from `config/gates/thresholds.yaml`: {'min_trades': 100, 'min_profit_factor': 1.2, 'min_dsr': 0.95, 'max_drawdown': -0.5, 'min_positive_folds': 0.5}.
Walk-forward: {'n_folds': 5, 'initial_train_frac': 0.4, 'anchored': True, 'grid_points': 4, 'max_trials': 48, 'min_train_trades': 10, 'seed': 0}. Engine: {'initial_equity': 10000.0, 'risk_pct': 1.0, 'max_leverage': 1.0}. Costs: moomoo SG fees and default spread/slippage for US stocks; for FX the spread Oanda quoted at each bar's open plus 0.2 pip slippage, and financing from official central-bank policy rates (data/cache/macro).
US universes: screened at every walk-forward test-fold start (and yearly before the first) from a pool of 102 stocks (the S&P 100 as of Oct 2026 plus the first run's names) and 22 ETFs; top 30 stocks, 10 ETFs, or 30 of both, ranked by trailing 60-day median dollar volume using bars before the screen date only.

**0 of 30 strategies run pass the gate.** 5 need data, 11 are not built or not standalone.


## Strategies run

| Strategy | Catalog entry | Market | Universe | Bars used | OOS trades | PF | Sharpe | DSR (N) | Max DD | Folds + | Result | Failing rung |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `fx-donchian-breakout-h4` | Donchian channel breakout | FX (Oanda practice bid/ask) H4, 2016-10-04..2026-10-02 | 6 pairs named by the spec; 6 symbols ever in it | 93,330 | 882 | 0.817 | -0.605 | 0.007 (48) | -0.245 | 1/5 | **fail** | profit_factor |
| `stk-ema-pullback-swing` | EMA pullback swing (seed) | US stocks/ETFs (moomoo) D1, 2006-09-20..2026-10-02 | top 30 of 124 all by trailing 60-day median dollar volume, re-screened at 13 dates; 67 symbols ever in it | 321,059 | 2132 | 0.993 | -0.007 | 0.0 (32) | -0.367 | 1/5 | **fail** | profit_factor |
| `stk-macd-trend-candle` | MACD trend with candle confirmation (seed) | US stocks/ETFs (moomoo) D1, 2006-09-20..2026-10-02 | top 30 of 124 all by trailing 60-day median dollar volume, re-screened at 13 dates; 67 symbols ever in it | 321,059 | 300 | 0.859 | -0.332 | 0.032 (32) | -0.245 | 2/5 | **fail** | profit_factor |
| `fx-trend-pullback-engulfing` | Trend pullback with engulfing candle (seed) | FX (Oanda practice bid/ask) H1, 2016-10-04..2026-10-02 | 3 pairs named by the spec; 3 symbols ever in it | 186,557 | 1782 | 0.854 | -0.892 | 0.0 (16) | -0.321 | 0/5 | **fail** | profit_factor |
| `stk-rsi2-meanrev` | RSI(2) pullback in uptrend (seed) | US stocks/ETFs (moomoo) D1, 2006-09-20..2026-10-02 | top 10 of 22 etfs by trailing 60-day median dollar volume, re-screened at 13 dates; 17 symbols ever in it | 78,943 | 715 | 1.039 | 0.075 | 0.288 (32) | -0.18 | 3/5 | **fail** | profit_factor |
| `fx-rsi-range-reversion` | RSI range reversion (seed) | FX (Oanda practice bid/ask) H1, 2016-10-04..2026-10-02 | 3 pairs named by the spec; 3 symbols ever in it | 186,523 | 50 | 0.827 | -0.196 | 0.0 (16) | -0.017 | 2/5 | **fail** | trades |
| `stk-breakout-volume` | Breakout with volume (seed) | US stocks/ETFs (moomoo) D1, 2006-09-20..2026-10-02 | top 30 of 102 stocks by trailing 60-day median dollar volume, re-screened at 13 dates; 60 symbols ever in it | 288,168 | 658 | 1.392 | 0.738 | 0.904 (32) | -0.138 | 5/5 | **fail** | deflated_sharpe |
| `stk-double-bottom-breakout` | Double bottom breakout (seed) | US stocks/ETFs (moomoo) D1, 2006-09-20..2026-10-02 | top 30 of 102 stocks by trailing 60-day median dollar volume, re-screened at 13 dates; 60 symbols ever in it | 288,168 | 331 | 1.121 | 0.208 | 0.115 (32) | -0.128 | 3/5 | **fail** | profit_factor |
| `stk-52w-high-momentum` | 52-week-high momentum | US stocks/ETFs (moomoo) D1, 2006-09-20..2026-10-02 | top 30 of 102 stocks by trailing 60-day median dollar volume, re-screened at 13 dates; 60 symbols ever in it | 288,168 | 2136 | 1.091 | 0.28 | 0.549 (32) | -0.2 | 2/5 | **fail** | profit_factor |
| `stk-sector-relative-strength` | Relative strength vs sector | US stocks/ETFs (moomoo) D1, 2006-09-20..2026-10-02 | top 30 of 102 stocks by trailing 60-day median dollar volume, re-screened at 13 dates; 60 symbols ever in it | 288,168 | 2682 | 1.175 | 0.51 | 0.65 (32) | -0.298 | 5/5 | **fail** | profit_factor |
| `stk-pairs-zscore` | Pairs trading (distance and cointegration) | US stocks/ETFs (moomoo) D1, 2006-09-20..2026-10-02 | top 30 of 102 stocks by trailing 60-day median dollar volume, re-screened at 13 dates; 18 symbols ever in it | 89,558 | 784 | 0.765 | -1.2 | 0.0 (32) | -0.514 | 0/5 | **fail** | profit_factor |
| `stk-gap-and-go-h1` | Gap-and-go after news gap | US stocks/ETFs (moomoo) H1, 2018-09-28..2026-10-02 | top 30 of 80 stocks by trailing 60-day median dollar volume, re-screened at 9 dates; 55 symbols ever in it | 764,887 | 1301 | 0.761 | -1.335 | 0.0 (32) | -0.531 | 1/5 | **fail** | profit_factor |
| `etf-overnight-hold` | Overnight vs intraday return split | US stocks/ETFs (moomoo) H1, 2018-09-28..2026-10-02 | top 10 of 22 etfs by trailing 60-day median dollar volume, re-screened at 9 dates; 17 symbols ever in it | 238,658 | 2422 | 0.626 | -2.492 | 0.0 (8) | -0.76 | 0/5 | **fail** | profit_factor |
| `etf-ts-xs-momentum` | Joint time-series and cross-sectional strategy | US stocks/ETFs (moomoo) D1, 2006-09-20..2026-10-02 | top 10 of 22 etfs by trailing 60-day median dollar volume, re-screened at 13 dates; 17 symbols ever in it | 78,943 | 1032 | 1.252 | 0.511 | 0.842 (32) | -0.198 | 4/5 | **fail** | deflated_sharpe |
| `fx-ts-xs-momentum` | Joint time-series and cross-sectional strategy | FX (Oanda practice bid/ask) D1, 2016-10-03..2026-10-01 | 9 pairs named by the spec; 9 symbols ever in it | 23,429 | 2544 | 0.609 | -2.374 | 0.0 (16) | -0.556 | 0/5 | **fail** | profit_factor |
| `etf-intraday-momentum` | Intraday momentum (first half-hour predicts last) | US stocks/ETFs (moomoo) H1, 2018-09-28..2026-10-02 | top 10 of 22 etfs by trailing 60-day median dollar volume, re-screened at 9 dates; 17 symbols ever in it | 238,658 | 10645 | 0.217 | -14.081 | 0.0 (4) | -1.0 | 0/5 | **fail** | profit_factor |
| `etf-intraday-momentum-m30` | Intraday momentum (first half-hour predicts last) | US stocks/ETFs (moomoo) M30, 2018-10-01..2026-10-02 | top 10 of 22 etfs by trailing 60-day median dollar volume, re-screened at 9 dates; 16 symbols ever in it | 416,864 | 10397 | 0.197 | -14.186 | 0.0 (4) | -1.0 | 0/5 | **fail** | profit_factor |
| `stk-earnings-jump` | Earnings-day jump continuation | US stocks/ETFs (moomoo) D1, 2006-09-20..2026-10-02 | top 30 of 102 stocks by trailing 60-day median dollar volume, re-screened at 13 dates; 60 symbols ever in it | 288,168 | 230 | 1.267 | 0.41 | 0.69 (16) | -0.066 | 4/5 | **fail** | deflated_sharpe |
| `stk-earnings-jump-continuation` | Earnings-day jump continuation | US stocks/ETFs (moomoo) D1, 2006-09-20..2026-10-02 | top 30 of 102 stocks by trailing 60-day median dollar volume, re-screened at 13 dates; 60 symbols ever in it | 288,168 | 313 | 1.242 | 0.453 | 0.792 (16) | -0.103 | 3/5 | **fail** | deflated_sharpe |
| `stk-pead-ear` | Post-earnings announcement drift | US stocks/ETFs (moomoo) D1, 2006-09-20..2026-10-02 | top 30 of 102 stocks by trailing 60-day median dollar volume, re-screened at 13 dates; 60 symbols ever in it | 288,168 | 463 | 1.053 | 0.127 | 0.407 (16) | -0.131 | 3/5 | **fail** | profit_factor |
| `etf-pre-fomc-drift-h1` | Pre-FOMC announcement drift | US stocks/ETFs (moomoo) H1, 2018-09-28..2026-10-02 | spec universe; 4 symbols ever in it | 56,156 | 96 | 1.81 | 0.65 | 0.902 (4) | -0.036 | 4/5 | **fail** | trades |
| `etf-risk-on-trend` | Hidden Markov regime allocation | US stocks/ETFs (moomoo) D1, 2006-09-20..2026-10-02 | spec universe; 13 symbols ever in it | 58,801 | 1224 | 1.205 | 0.319 | 0.521 (48) | -0.255 | 3/5 | **fail** | deflated_sharpe |
| `etf-corr-calm-trend` | Realised-covariance regime detection | US stocks/ETFs (moomoo) D1, 2006-09-20..2026-10-02 | spec universe; 13 symbols ever in it | 58,801 | 1238 | 0.976 | -0.003 | 0.166 (48) | -0.267 | 3/5 | **fail** | profit_factor |
| `etf-panic-rebound` | Buy equity after VIX spike above 30 | US stocks/ETFs (moomoo) D1, 2006-09-20..2026-10-02 | spec universe; 13 symbols ever in it | 58,801 | 82 | 2.451 | 0.348 | 0.853 (16) | -0.135 | 3/5 | **fail** | trades |
| `etf-vix-panic-rebound` | Buy equity after VIX spike above 30 | US stocks/ETFs (moomoo) D1, 2006-09-20..2026-10-02 | spec universe; 13 symbols ever in it | 58,801 | 97 | 1.77 | 0.273 | 0.79 (16) | -0.139 | 3/5 | **fail** | trades |
| `stk-rs-momentum-crash-protected` | Momentum with crash protection (vol-scaled) | US stocks/ETFs (moomoo) D1, 2006-09-20..2026-10-02 | top 30 of 102 stocks by trailing 60-day median dollar volume, re-screened at 13 dates; 60 symbols ever in it | 288,168 | 1480 | 1.318 | 0.658 | 0.949 (16) | -0.252 | 5/5 | **fail** | deflated_sharpe |
| `fx-currency-strength-momentum` | Joint time-series and cross-sectional strategy | FX (Oanda practice bid/ask) D1, 2016-10-03..2026-10-01 | 9 pairs named by the spec; 9 symbols ever in it | 23,429 | 1174 | 0.625 | -1.62 | 0.0 (16) | -0.499 | 0/5 | **fail** | profit_factor |
| `fx-carry-trend` | G10 carry (long high-rate, short low-rate) | FX (Oanda practice bid/ask) D1, 2016-10-03..2026-10-01 | 9 pairs named by the spec; 9 symbols ever in it | 23,429 | 1138 | 0.686 | -1.269 | 0.0 (16) | -0.379 | 1/5 | **fail** | profit_factor |
| `fx-carry-vol-filter` | Carry with volatility filter | FX (Oanda practice bid/ask) D1, 2016-10-03..2026-10-01 | 9 pairs named by the spec; 9 symbols ever in it | 23,429 | 1301 | 0.612 | -1.76 | 0.0 (16) | -0.468 | 0/5 | **fail** | profit_factor |
| `fx-rate-diff-trend` | Rate-differential trend (FX) | FX (Oanda practice bid/ask) D1, 2016-10-03..2026-10-01 | 9 pairs named by the spec; 9 symbols ever in it | 23,429 | 576 | 0.583 | -1.551 | 0.0 (16) | -0.431 | 0/5 | **fail** | profit_factor |

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
- `fx-ts-xs-momentum`: profit factor 0.61 after costs (need 1.2); deflated Sharpe 0.00 (need 0.95); max drawdown -56% (limit -50%); only 0% of test folds profitable
- `etf-intraday-momentum`: profit factor 0.22 after costs (need 1.2); deflated Sharpe 0.00 (need 0.95); max drawdown -100% (limit -50%); only 0% of test folds profitable
- `etf-intraday-momentum-m30`: profit factor 0.20 after costs (need 1.2); deflated Sharpe 0.00 (need 0.95); max drawdown -100% (limit -50%); only 0% of test folds profitable
- `stk-earnings-jump`: deflated Sharpe 0.69 (need 0.95)
- `stk-earnings-jump-continuation`: deflated Sharpe 0.79 (need 0.95)
- `stk-pead-ear`: profit factor 1.05 after costs (need 1.2); deflated Sharpe 0.41 (need 0.95)
- `etf-pre-fomc-drift-h1`: only 96 out-of-sample trades (need 100); deflated Sharpe 0.90 (need 0.95)
- `etf-risk-on-trend`: deflated Sharpe 0.52 (need 0.95)
- `etf-corr-calm-trend`: profit factor 0.98 after costs (need 1.2); deflated Sharpe 0.17 (need 0.95)
- `etf-panic-rebound`: only 82 out-of-sample trades (need 100); deflated Sharpe 0.85 (need 0.95)
- `etf-vix-panic-rebound`: only 97 out-of-sample trades (need 100); deflated Sharpe 0.79 (need 0.95)
- `stk-rs-momentum-crash-protected`: deflated Sharpe 0.95 (need 0.95)
- `fx-currency-strength-momentum`: profit factor 0.62 after costs (need 1.2); deflated Sharpe 0.00 (need 0.95); only 0% of test folds profitable
- `fx-carry-trend`: profit factor 0.69 after costs (need 1.2); deflated Sharpe 0.00 (need 0.95); only 20% of test folds profitable
- `fx-carry-vol-filter`: profit factor 0.61 after costs (need 1.2); deflated Sharpe 0.00 (need 0.95); only 0% of test folds profitable
- `fx-rate-diff-trend`: profit factor 0.58 after costs (need 1.2); deflated Sharpe 0.00 (need 0.95); only 0% of test folds profitable

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
| Regime-switching volatility forecast for sizing | Build: sizing input | not a standalone signal | sizing input |
| Long calls or puts on high-conviction plans | Build: execution style, not a signal | not a standalone signal | execution style; needs live option chains |
| Pre-earnings run-up and IV crush (options) | Build | needs data | option implied-volatility history and option prices (earnings dates now exist; OpenD F10 has IV only around each report) |
| LLM news sentiment long-short | Build | needs data | news headline history |
| ChatGPT headline scoring | Build | needs data | news headline history |
| Short high borrow-fee, high short-interest names | Build | needs data | borrow-fee and short-interest history (not downloaded in phase 1) |
| Short-seller flow signal | Build | needs data | short-volume history (not downloaded in phase 1) |

## What changed vs run 2

- Merged with the proposed-strategy work: the 11 strategies in strategies/proposed/ run here, built by tradex.research.builders from injected calendars and rate histories. The three stock ones run on the same liquidity screen as the seeds; the ETF ones keep their full index/sector-SPDR universe.
- Policy rates for all eight currencies from the central banks' own data (FRED for the Fed, ECB, BoE, RBA, BoC Valet, SNB data portal; BoJ and RBNZ via the BIS policy-rate dataset), cached in data/cache/macro. The carry features and FX financing read this one file. CAD, CHF and NZD pairs now run, financed.
- The bundled cost table tradex/costs/policy_rates.csv is rebuilt from the same official series (the old one was written from memory: it missed the 2026 ECB, RBA and Fed moves and dated RBA changes a day early).
- Scheduled FOMC meetings 2006-2027 from federalreserve.gov feed the pre-FOMC drift strategy; Cboe VIX daily history feeds `etf-vix-panic-rebound`, the literal VIX>30 form of the panic-rebound entry (new).
- Earnings strategies from the proposals read the run-2 earnings calendar (OpenD release dates and timing, SEC filing-date proxies before that).
- 30-minute OpenD bars for the 22 ETFs (no new history quota: 124 of 300 still used): `etf-intraday-momentum-m30` takes the first half hour as its signal and enters at 15:30 (new).
- Trial ledger: every parameter set of every variant run here is recorded, so N grows for re-run strategies.

| Strategy | Run 2 | This run | Result change |
|---|---|---|---|
| `fx-donchian-breakout-h4` | 879 trades, PF 0.824, Sharpe -0.581, DSR 0.008 (N 48), 6 symbols | 882 trades, PF 0.817, Sharpe -0.605, DSR 0.007 (N 48), 6 symbols | fail -> **fail** |
| `stk-ema-pullback-swing` | 2132 trades, PF 0.993, Sharpe -0.007, DSR 0.0 (N 32), 67 symbols | 2132 trades, PF 0.993, Sharpe -0.007, DSR 0.0 (N 32), 67 symbols | fail -> **fail** |
| `stk-macd-trend-candle` | 300 trades, PF 0.859, Sharpe -0.332, DSR 0.032 (N 32), 67 symbols | 300 trades, PF 0.859, Sharpe -0.332, DSR 0.032 (N 32), 67 symbols | fail -> **fail** |
| `fx-trend-pullback-engulfing` | 1786 trades, PF 0.855, Sharpe -0.89, DSR 0.0 (N 16), 3 symbols | 1782 trades, PF 0.854, Sharpe -0.892, DSR 0.0 (N 16), 3 symbols | fail -> **fail** |
| `stk-rsi2-meanrev` | 715 trades, PF 1.039, Sharpe 0.075, DSR 0.288 (N 32), 17 symbols | 715 trades, PF 1.039, Sharpe 0.075, DSR 0.288 (N 32), 17 symbols | fail -> **fail** |
| `fx-rsi-range-reversion` | 50 trades, PF 0.826, Sharpe -0.197, DSR 0.0 (N 16), 3 symbols | 50 trades, PF 0.827, Sharpe -0.196, DSR 0.0 (N 16), 3 symbols | fail -> **fail** |
| `stk-breakout-volume` | 658 trades, PF 1.392, Sharpe 0.738, DSR 0.904 (N 32), 60 symbols | 658 trades, PF 1.392, Sharpe 0.738, DSR 0.904 (N 32), 60 symbols | fail -> **fail** |
| `stk-double-bottom-breakout` | 331 trades, PF 1.121, Sharpe 0.208, DSR 0.115 (N 32), 60 symbols | 331 trades, PF 1.121, Sharpe 0.208, DSR 0.115 (N 32), 60 symbols | fail -> **fail** |
| `stk-52w-high-momentum` | 2136 trades, PF 1.091, Sharpe 0.28, DSR 0.549 (N 32), 60 symbols | 2136 trades, PF 1.091, Sharpe 0.28, DSR 0.549 (N 32), 60 symbols | fail -> **fail** |
| `stk-sector-relative-strength` | 2682 trades, PF 1.175, Sharpe 0.51, DSR 0.65 (N 32), 60 symbols | 2682 trades, PF 1.175, Sharpe 0.51, DSR 0.65 (N 32), 60 symbols | fail -> **fail** |
| `stk-pairs-zscore` | 784 trades, PF 0.765, Sharpe -1.2, DSR 0.0 (N 32), 18 symbols | 784 trades, PF 0.765, Sharpe -1.2, DSR 0.0 (N 32), 18 symbols | fail -> **fail** |
| `stk-gap-and-go-h1` | 1301 trades, PF 0.761, Sharpe -1.335, DSR 0.0 (N 32), 55 symbols | 1301 trades, PF 0.761, Sharpe -1.335, DSR 0.0 (N 32), 55 symbols | fail -> **fail** |
| `etf-overnight-hold` | 2422 trades, PF 0.626, Sharpe -2.492, DSR 0.0 (N 8), 17 symbols | 2422 trades, PF 0.626, Sharpe -2.492, DSR 0.0 (N 8), 17 symbols | fail -> **fail** |
| `etf-ts-xs-momentum` | 1032 trades, PF 1.252, Sharpe 0.511, DSR 0.842 (N 32), 17 symbols | 1032 trades, PF 1.252, Sharpe 0.511, DSR 0.842 (N 32), 17 symbols | fail -> **fail** |
| `fx-ts-xs-momentum` | 1492 trades, PF 0.699, Sharpe -1.51, DSR 0.0 (N 16), 6 symbols | 2544 trades, PF 0.609, Sharpe -2.374, DSR 0.0 (N 16), 9 symbols | fail -> **fail** |
| `etf-intraday-momentum` | 10645 trades, PF 0.217, Sharpe -14.081, DSR 0.0 (N 4), 17 symbols | 10645 trades, PF 0.217, Sharpe -14.081, DSR 0.0 (N 4), 17 symbols | fail -> **fail** |
| `etf-intraday-momentum-m30` | not run | 10397 trades, PF 0.197, Sharpe -14.186, DSR 0.0 (N 4), 16 symbols | new: **fail** |
| `stk-earnings-jump` | 230 trades, PF 1.267, Sharpe 0.41, DSR 0.69 (N 16), 60 symbols | 230 trades, PF 1.267, Sharpe 0.41, DSR 0.69 (N 16), 60 symbols | fail -> **fail** |
| `stk-earnings-jump-continuation` | not run | 313 trades, PF 1.242, Sharpe 0.453, DSR 0.792 (N 16), 60 symbols | new: **fail** |
| `stk-pead-ear` | not run | 463 trades, PF 1.053, Sharpe 0.127, DSR 0.407 (N 16), 60 symbols | new: **fail** |
| `etf-pre-fomc-drift-h1` | not run | 96 trades, PF 1.81, Sharpe 0.65, DSR 0.902 (N 4), 4 symbols | new: **fail** |
| `etf-risk-on-trend` | not run | 1224 trades, PF 1.205, Sharpe 0.319, DSR 0.521 (N 48), 13 symbols | new: **fail** |
| `etf-corr-calm-trend` | not run | 1238 trades, PF 0.976, Sharpe -0.003, DSR 0.166 (N 48), 13 symbols | new: **fail** |
| `etf-panic-rebound` | not run | 82 trades, PF 2.451, Sharpe 0.348, DSR 0.853 (N 16), 13 symbols | new: **fail** |
| `etf-vix-panic-rebound` | not run | 97 trades, PF 1.77, Sharpe 0.273, DSR 0.79 (N 16), 13 symbols | new: **fail** |
| `stk-rs-momentum-crash-protected` | not run | 1480 trades, PF 1.318, Sharpe 0.658, DSR 0.949 (N 16), 60 symbols | new: **fail** |
| `fx-currency-strength-momentum` | not run | 1174 trades, PF 0.625, Sharpe -1.62, DSR 0.0 (N 16), 9 symbols | new: **fail** |
| `fx-carry-trend` | not run | 1138 trades, PF 0.686, Sharpe -1.269, DSR 0.0 (N 16), 9 symbols | new: **fail** |
| `fx-carry-vol-filter` | not run | 1301 trades, PF 0.612, Sharpe -1.76, DSR 0.0 (N 16), 9 symbols | new: **fail** |
| `fx-rate-diff-trend` | not run | 576 trades, PF 0.583, Sharpe -1.551, DSR 0.0 (N 16), 9 symbols | new: **fail** |

## Caveats

- Survivorship bias remains: the candidate pool is today's S&P 100 and today's ETFs. OpenD has no delisted names, so stocks that left the index or failed since 2006 can never be picked by the screen. Results on screened universes are less hand-picked than the first run, not survivorship-free.
- The S&P 100 list is the October 2026 membership as written in tradex/research/universe.py (checked against OpenD's S&P 500 plate); the screen ranks within it, but membership itself is today's.
- Earnings dates before OpenD's coverage (about 2013 on) are SEC filing dates, a proxy: they can lag the release by a day and their before/after-market timing is unknown, so both filters treat them as before the open (conservative). Coverage starts late for: ACN from 2009-10-01, BLK from 2014-04-17, XOM from 2014-01-30.
- FX financing is the official policy-rate differential minus Oanda's admin fee; Oanda does not publish historical financing rates, so actual swap rates (which track interbank rates, not policy rates) differ.
- JPY and NZD rates are the BIS compilation of the BoJ and RBNZ series (rbnz.govt.nz refuses scripted downloads; BoJ has no policy-rate series). The BIS JPY series holds 0.05% (the 0-0.1% call-rate guideline) until 2016-09-21 and -0.1% from then. Swiss rates before 2019-06-13 are the SNB's 3-month Libor target-range midpoint, dated at month end (up to a month late, never early).
- FOMC statement times are not on the Fed's calendar pages: 14:00 New York from 2013 (12:30 on 2011-2012 press-conference days, 14:15 before). The 60-minute bars the pre-FOMC strategy uses start in 2018.
- FX seeds' no_high_impact_news_30m filter is inactive: there is no historical macro calendar before 2026.
- Stock spreads are the cost model's defaults (1 bp half-spread + 2 bp slippage), not measured quotes.
- Fixed moomoo fees (US$0.99 + 9% GST per order) weigh heavily at the US$10,000 test equity.
- OpenD history starts 2006-09 for daily bars and 2018-09 for 60-minute bars; Oanda history here is 10 years.
- DSR's N is every parameter set ever recorded for the strategy in the trial ledger, across all variants.
