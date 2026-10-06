# Phase-1 gate run

Run 2026-10-05 09:18 UTC on real bars only (moomoo OpenD, qfq-adjusted; Oanda practice bid/ask candles for FX). Stage-3 thresholds from `config/gates/thresholds.yaml`: {'min_trades': 100, 'min_profit_factor': 1.2, 'min_dsr': 0.95, 'max_drawdown': -0.5, 'min_positive_folds': 0.5}.
Walk-forward: {'n_folds': 5, 'initial_train_frac': 0.4, 'anchored': True, 'grid_points': 4, 'max_trials': 48, 'min_train_trades': 10, 'seed': 0}. Engine: {'initial_equity': 10000.0, 'risk_pct': 1.0, 'max_leverage': 1.0}. Costs: moomoo SG fees and default spread/slippage for US stocks; for FX the spread Oanda quoted at each bar's open plus 0.2 pip slippage, and financing from official central-bank policy rates (data/cache/macro).
US universes: screened at every walk-forward test-fold start (and yearly before the first) from a pool of 102 stocks (the S&P 100 as of Oct 2026 plus the first run's names) and 22 ETFs; top 30 stocks, 10 ETFs, or 30 of both, ranked by trailing 60-day median dollar volume using bars before the screen date only. The panic-rebound pool variants take every ETF listed at the screen date: all of the 22-ETF pool, or of a 34-ETF pool that adds 12 equity ETFs.

**0 of 35 strategies run pass the gate.** 5 need data, 11 are not built or not standalone.


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
| `etf-pre-fomc-drift-d1` | Pre-FOMC announcement drift | US stocks/ETFs (moomoo) D1, 2006-09-20..2026-10-02 | spec universe; 13 symbols ever in it | 58,801 | 551 | 0.816 | -0.232 | 0.065 (8) | -0.132 | 3/5 | **fail** | profit_factor |
| `etf-risk-on-trend` | Hidden Markov regime allocation | US stocks/ETFs (moomoo) D1, 2006-09-20..2026-10-02 | spec universe; 13 symbols ever in it | 58,801 | 1224 | 1.205 | 0.319 | 0.521 (48) | -0.255 | 3/5 | **fail** | deflated_sharpe |
| `etf-corr-calm-trend` | Realised-covariance regime detection | US stocks/ETFs (moomoo) D1, 2006-09-20..2026-10-02 | spec universe; 13 symbols ever in it | 58,801 | 1238 | 0.976 | -0.003 | 0.166 (48) | -0.267 | 3/5 | **fail** | profit_factor |
| `etf-panic-rebound` | Buy equity after VIX spike above 30 | US stocks/ETFs (moomoo) D1, 2006-09-20..2026-10-02 | spec universe; 13 symbols ever in it | 58,801 | 82 | 2.451 | 0.348 | 0.847 (32) | -0.135 | 3/5 | **fail** | trades |
| `etf-vix-panic-rebound` | Buy equity after VIX spike above 30 | US stocks/ETFs (moomoo) D1, 2006-09-20..2026-10-02 | spec universe; 13 symbols ever in it | 58,801 | 97 | 1.77 | 0.273 | 0.784 (32) | -0.139 | 3/5 | **fail** | trades |
| `etf-panic-rebound-pool22` | Buy equity after VIX spike above 30 | US stocks/ETFs (moomoo) D1, 2006-09-20..2026-10-02 | top 22 of 22 etfs by trailing 60-day median dollar volume, re-screened at 13 dates; 22 symbols ever in it | 97,475 | 83 | 2.128 | 0.42 | 0.857 (48) | -0.119 | 3/5 | **fail** | trades |
| `etf-vix-panic-rebound-pool22` | Buy equity after VIX spike above 30 | US stocks/ETFs (moomoo) D1, 2006-09-20..2026-10-02 | top 22 of 22 etfs by trailing 60-day median dollar volume, re-screened at 13 dates; 22 symbols ever in it | 97,475 | 109 | 1.326 | 0.206 | 0.603 (64) | -0.124 | 4/5 | **fail** | deflated_sharpe |
| `etf-panic-rebound-pool34` | Buy equity after VIX spike above 30 | US stocks/ETFs (moomoo) D1, 2006-09-20..2026-10-02 | top 34 of 34 etfs_wide by trailing 60-day median dollar volume, re-screened at 13 dates; 34 symbols ever in it | 147,000 | 90 | 1.914 | 0.37 | 0.84 (80) | -0.139 | 3/5 | **fail** | trades |
| `etf-vix-panic-rebound-pool34` | Buy equity after VIX spike above 30 | US stocks/ETFs (moomoo) D1, 2006-09-20..2026-10-02 | top 34 of 34 etfs_wide by trailing 60-day median dollar volume, re-screened at 13 dates; 34 symbols ever in it | 147,000 | 119 | 1.648 | 0.314 | 0.591 (96) | -0.082 | 4/5 | **fail** | deflated_sharpe |
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
- `etf-pre-fomc-drift-d1`: profit factor 0.82 after costs (need 1.2); deflated Sharpe 0.06 (need 0.95)
- `etf-risk-on-trend`: deflated Sharpe 0.52 (need 0.95)
- `etf-corr-calm-trend`: profit factor 0.98 after costs (need 1.2); deflated Sharpe 0.17 (need 0.95)
- `etf-panic-rebound`: only 82 out-of-sample trades (need 100); deflated Sharpe 0.85 (need 0.95)
- `etf-vix-panic-rebound`: only 97 out-of-sample trades (need 100); deflated Sharpe 0.78 (need 0.95)
- `etf-panic-rebound-pool22`: only 83 out-of-sample trades (need 100); deflated Sharpe 0.86 (need 0.95)
- `etf-vix-panic-rebound-pool22`: deflated Sharpe 0.60 (need 0.95)
- `etf-panic-rebound-pool34`: only 90 out-of-sample trades (need 100); deflated Sharpe 0.84 (need 0.95)
- `etf-vix-panic-rebound-pool34`: deflated Sharpe 0.59 (need 0.95)
- `stk-rs-momentum-crash-protected`: deflated Sharpe 0.95 (need 0.95)
- `fx-currency-strength-momentum`: profit factor 0.62 after costs (need 1.2); deflated Sharpe 0.00 (need 0.95); only 0% of test folds profitable
- `fx-carry-trend`: profit factor 0.69 after costs (need 1.2); deflated Sharpe 0.00 (need 0.95); only 20% of test folds profitable
- `fx-carry-vol-filter`: profit factor 0.61 after costs (need 1.2); deflated Sharpe 0.00 (need 0.95); only 0% of test folds profitable
- `fx-rate-diff-trend`: profit factor 0.58 after costs (need 1.2); deflated Sharpe 0.00 (need 0.95); only 0% of test folds profitable

## Near misses: how solid are they (report only, not part of the gate)

Strategies that clear every rung except the deflated Sharpe. Nothing was changed to make them pass. Since 2016: the same out-of-sample returns and trades from 2016-01-01 on. Costs x2: each fold's test window rerun with the parameters that fold chose, spread and slippage doubled (no new selection, so not a new trial). Bootstrap: stationary bootstrap of the daily out-of-sample returns, 5,000 resamples, mean block 10 days; the Sharpe interval is 5th to 95th percentile. Expected max Sharpe is what the best of N skill-less trials would show; the DSR asks how likely the observed Sharpe beats it.

| Strategy | OOS (all) | Since 2016 | Costs x2 | Bootstrap Sharpe 5-50-95% | P(Sharpe <= 0) | Expected max Sharpe (N) | Param stability |
|---|---|---|---|---|---|---|---|
| `stk-breakout-volume` | 658 trades, PF 1.392, Sharpe 0.738, DD -0.138 | 592 trades, PF 1.459, Sharpe 0.86, DD -0.138 | 655 trades, PF 1.353, Sharpe 0.674, folds 5/5 | 0.291 / 0.736 / 1.186 | 0.005 | 0.357 (32) | 0.6 |
| `etf-ts-xs-momentum` | 1032 trades, PF 1.252, Sharpe 0.511, DD -0.198 | 904 trades, PF 1.312, Sharpe 0.595, DD -0.186 | 1039 trades, PF 1.154, Sharpe 0.317, folds 3/5 | 0.045 / 0.521 / 0.987 | 0.035 | 0.218 (32) | 0.8 |
| `stk-earnings-jump` | 230 trades, PF 1.267, Sharpe 0.41, DD -0.066 | 206 trades, PF 1.327, Sharpe 0.498, DD -0.063 | 231 trades, PF 1.242, Sharpe 0.377, folds 4/5 | -0.028 / 0.41 / 0.847 | 0.065 | 0.268 (16) | 0.4 |
| `stk-earnings-jump-continuation` | 313 trades, PF 1.242, Sharpe 0.453, DD -0.103 | 281 trades, PF 1.277, Sharpe 0.504, DD -0.103 | 313 trades, PF 1.205, Sharpe 0.393, folds 3/5 | -0.001 / 0.451 / 0.898 | 0.05 | 0.22 (16) | 1.0 |
| `etf-risk-on-trend` | 1224 trades, PF 1.205, Sharpe 0.319, DD -0.255 | 1070 trades, PF 1.306, Sharpe 0.456, DD -0.255 | 1209 trades, PF 1.077, Sharpe 0.141, folds 2/5 | -0.138 / 0.324 / 0.785 | 0.127 | 0.304 (48) | 0.6 |
| `etf-vix-panic-rebound-pool22` | 109 trades, PF 1.326, Sharpe 0.206, DD -0.124 | 106 trades, PF 1.351, Sharpe 0.231, DD -0.124 | 107 trades, PF 1.412, Sharpe 0.254, folds 3/5 | -0.151 / 0.224 / 0.63 | 0.167 | 0.129 (64) | 0.6 |
| `etf-vix-panic-rebound-pool34` | 119 trades, PF 1.648, Sharpe 0.314, DD -0.082 | 111 trades, PF 1.697, Sharpe 0.359, DD -0.082 | 119 trades, PF 1.605, Sharpe 0.298, folds 4/5 | -0.044 / 0.324 / 0.696 | 0.075 | 0.247 (96) | 0.6 |
| `stk-rs-momentum-crash-protected` | 1480 trades, PF 1.318, Sharpe 0.658, DD -0.252 | 1329 trades, PF 1.345, Sharpe 0.671, DD -0.252 | 1483 trades, PF 1.301, Sharpe 0.628, folds 5/5 | 0.21 / 0.664 / 1.129 | 0.005 | 0.18 (16) | 0.4 |

Per fold (test window: return, annualised Sharpe, trades, chosen parameters):

- `stk-breakout-volume`: 2014-09-24..2017-02-16: +5.4%, Sharpe 0.376, 120 trades, {'stop_atr': 2.5, 'features.relvol.period': 23}; 2017-02-16..2019-07-16: +3.1%, Sharpe 0.218, 133 trades, {'stop_atr': 3.0, 'features.relvol.period': 30}; 2019-07-16..2021-12-07: +43.8%, Sharpe 1.397, 164 trades, {'stop_atr': 2.5, 'features.relvol.period': 30}; 2021-12-07..2024-05-06: +12.2%, Sharpe 0.711, 110 trades, {'stop_atr': 2.5, 'features.relvol.period': 30}; 2024-05-06..2026-10-02: +12.3%, Sharpe 0.655, 131 trades, {'stop_atr': 2.5, 'features.relvol.period': 30}
- `etf-ts-xs-momentum`: 2014-09-22..2017-02-15: -11.5%, Sharpe -0.592, 244 trades, {'stop_atr': 3.3333, 'target_r': 2.0}; 2017-02-15..2019-07-15: +11.1%, Sharpe 0.453, 216 trades, {'stop_atr': 2.6667, 'target_r': 2.0}; 2019-07-15..2021-12-06: +36.1%, Sharpe 1.206, 212 trades, {'stop_atr': 3.3333, 'target_r': 2.0}; 2021-12-06..2024-05-03: +1.8%, Sharpe 0.13, 142 trades, {'stop_atr': 3.3333, 'target_r': 2.0}; 2024-05-03..2026-10-02: +25.9%, Sharpe 1.075, 218 trades, {'stop_atr': 3.3333, 'target_r': 2.0}
- `stk-earnings-jump`: 2014-09-24..2017-02-16: -4.7%, Sharpe -0.984, 48 trades, {'stop_atr': 2.8333, 'max_bars': 10}; 2017-02-16..2019-07-16: +5.5%, Sharpe 0.518, 40 trades, {'stop_atr': 1.5, 'max_bars': 20}; 2019-07-16..2021-12-07: +4.9%, Sharpe 0.465, 20 trades, {'stop_atr': 1.5, 'max_bars': 40}; 2021-12-07..2024-05-06: +4.9%, Sharpe 0.477, 65 trades, {'stop_atr': 2.1667, 'max_bars': 20}; 2024-05-06..2026-10-02: +12.9%, Sharpe 0.74, 57 trades, {'stop_atr': 1.5, 'max_bars': 40}
- `stk-earnings-jump-continuation`: 2014-09-24..2017-02-16: +1.6%, Sharpe 0.189, 67 trades, {'stop_atr': 1.5, 'target_r': 1.5}; 2017-02-16..2019-07-16: -2.4%, Sharpe -0.202, 60 trades, {'stop_atr': 1.5, 'target_r': 1.5}; 2019-07-16..2021-12-07: +9.5%, Sharpe 0.887, 58 trades, {'stop_atr': 1.5, 'target_r': 1.5}; 2021-12-07..2024-05-06: -0.1%, Sharpe 0.007, 70 trades, {'stop_atr': 1.5, 'target_r': 1.5}; 2024-05-06..2026-10-02: +14.8%, Sharpe 1.36, 58 trades, {'stop_atr': 1.5, 'target_r': 1.5}
- `etf-risk-on-trend`: 2014-09-22..2017-02-15: -8.7%, Sharpe -0.336, 236 trades, {'stop_atr': 4.0, 'target_r': 2.6667, 'features.ema100.period': 117}; 2017-02-15..2019-07-15: +1.0%, Sharpe 0.092, 255 trades, {'stop_atr': 4.0, 'target_r': 2.6667, 'features.ema100.period': 150}; 2019-07-15..2021-12-06: +34.6%, Sharpe 1.08, 250 trades, {'stop_atr': 3.3333, 'target_r': 2.6667, 'features.ema100.period': 150}; 2021-12-06..2024-05-03: -8.0%, Sharpe -0.36, 229 trades, {'stop_atr': 4.0, 'target_r': 2.6667, 'features.ema100.period': 150}; 2024-05-03..2026-10-02: +21.3%, Sharpe 0.86, 254 trades, {'stop_atr': 4.0, 'target_r': 2.6667, 'features.ema100.period': 150}
- `etf-vix-panic-rebound-pool22`: 2014-09-22..2017-02-15: -0.5%, Sharpe -0.152, 3 trades, {'stop_atr': 4.3333, 'target_r': 2.6667}; 2017-02-15..2019-07-15: +1.7%, Sharpe 0.685, 6 trades, {'stop_atr': 4.3333, 'target_r': 2.6667}; 2019-07-15..2021-12-06: +4.8%, Sharpe 0.332, 48 trades, {'stop_atr': 4.3333, 'target_r': 2.6667}; 2021-12-06..2024-05-03: +0.0%, Sharpe 0.03, 39 trades, {'stop_atr': 4.3333, 'target_r': 4.0}; 2024-05-03..2026-10-02: +3.2%, Sharpe 0.558, 13 trades, {'stop_atr': 4.3333, 'target_r': 4.0}
- `etf-vix-panic-rebound-pool34`: 2014-09-22..2017-02-15: -0.6%, Sharpe -0.073, 8 trades, {'stop_atr': 5.0, 'target_r': 2.6667}; 2017-02-15..2019-07-15: +1.9%, Sharpe 0.742, 8 trades, {'stop_atr': 5.0, 'target_r': 2.6667}; 2019-07-15..2021-12-06: +10.6%, Sharpe 0.863, 47 trades, {'stop_atr': 5.0, 'target_r': 2.6667}; 2021-12-06..2024-05-03: +1.6%, Sharpe 0.135, 43 trades, {'stop_atr': 4.3333, 'target_r': 2.6667}; 2024-05-03..2026-10-02: +2.0%, Sharpe 0.239, 13 trades, {'stop_atr': 4.3333, 'target_r': 2.6667}
- `stk-rs-momentum-crash-protected`: 2014-09-24..2017-02-16: +6.8%, Sharpe 0.263, 285 trades, {'stop_atr': 4.0, 'target_r': 3.6667}; 2017-02-16..2019-07-16: +41.1%, Sharpe 0.984, 290 trades, {'stop_atr': 3.3333, 'target_r': 3.6667}; 2019-07-16..2021-12-07: +34.1%, Sharpe 0.76, 317 trades, {'stop_atr': 3.3333, 'target_r': 4.3333}; 2021-12-07..2024-05-06: +10.1%, Sharpe 0.355, 294 trades, {'stop_atr': 4.0, 'target_r': 3.0}; 2024-05-06..2026-10-02: +29.9%, Sharpe 0.915, 294 trades, {'stop_atr': 4.0, 'target_r': 3.0}

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

## What changed vs run 3

- Pre-FOMC drift on daily bars back to 2006 (`etf-pre-fomc-drift-d1`, new): the H1 hypothesis held close to close, from the close of the session before the statement day to the statement day's close, on the four index ETFs and the nine original sector SPDRs. Both orders are market-on-close orders placed a session ahead (new engine option `fill: next_close`), decided from the published FOMC schedule only. The H1 version runs as before; the two count their trials together.
- Panic rebound on more ETFs (four new pre-registered variants, rules and search space unchanged): `-pool22` runs on every ETF of the 22-ETF pool listed at each screen date, `-pool34` adds 12 liquid equity ETFs downloaded for this round (SMH, XBI, KRE, ITB, XHB, XOP, XRT, IBB, EWJ, EWZ, FXI, VWO; daily bars, OpenD history quota 124 -> 136 of 300). All six panic-rebound specs count their trials together.
- DSR's N can now span specs that test one hypothesis (`provenance.shares_trials_with`): the pre-FOMC pair and the six panic-rebound specs. This raises N for the run-3 versions of those strategies.
- Near misses (every rung passed except the deflated Sharpe) get a report-only robustness block: results since 2016, each fold's test window rerun with doubled spread and slippage at the parameters it chose, and a stationary-bootstrap interval of the out-of-sample Sharpe. Nothing about these strategies was changed.
- Gate thresholds are unchanged (config/gates/thresholds.yaml is not touched).
- Trial ledger kept: every parameter set run here is added, so N grows for re-run strategies.

| Strategy | Run 3 | This run | Result change |
|---|---|---|---|
| `fx-donchian-breakout-h4` | 882 trades, PF 0.817, Sharpe -0.605, DSR 0.007 (N 48), 6 symbols | 882 trades, PF 0.817, Sharpe -0.605, DSR 0.007 (N 48), 6 symbols | fail -> **fail** |
| `stk-ema-pullback-swing` | 2132 trades, PF 0.993, Sharpe -0.007, DSR 0.0 (N 32), 67 symbols | 2132 trades, PF 0.993, Sharpe -0.007, DSR 0.0 (N 32), 67 symbols | fail -> **fail** |
| `stk-macd-trend-candle` | 300 trades, PF 0.859, Sharpe -0.332, DSR 0.032 (N 32), 67 symbols | 300 trades, PF 0.859, Sharpe -0.332, DSR 0.032 (N 32), 67 symbols | fail -> **fail** |
| `fx-trend-pullback-engulfing` | 1782 trades, PF 0.854, Sharpe -0.892, DSR 0.0 (N 16), 3 symbols | 1782 trades, PF 0.854, Sharpe -0.892, DSR 0.0 (N 16), 3 symbols | fail -> **fail** |
| `stk-rsi2-meanrev` | 715 trades, PF 1.039, Sharpe 0.075, DSR 0.288 (N 32), 17 symbols | 715 trades, PF 1.039, Sharpe 0.075, DSR 0.288 (N 32), 17 symbols | fail -> **fail** |
| `fx-rsi-range-reversion` | 50 trades, PF 0.827, Sharpe -0.196, DSR 0.0 (N 16), 3 symbols | 50 trades, PF 0.827, Sharpe -0.196, DSR 0.0 (N 16), 3 symbols | fail -> **fail** |
| `stk-breakout-volume` | 658 trades, PF 1.392, Sharpe 0.738, DSR 0.904 (N 32), 60 symbols | 658 trades, PF 1.392, Sharpe 0.738, DSR 0.904 (N 32), 60 symbols | fail -> **fail** |
| `stk-double-bottom-breakout` | 331 trades, PF 1.121, Sharpe 0.208, DSR 0.115 (N 32), 60 symbols | 331 trades, PF 1.121, Sharpe 0.208, DSR 0.115 (N 32), 60 symbols | fail -> **fail** |
| `stk-52w-high-momentum` | 2136 trades, PF 1.091, Sharpe 0.28, DSR 0.549 (N 32), 60 symbols | 2136 trades, PF 1.091, Sharpe 0.28, DSR 0.549 (N 32), 60 symbols | fail -> **fail** |
| `stk-sector-relative-strength` | 2682 trades, PF 1.175, Sharpe 0.51, DSR 0.65 (N 32), 60 symbols | 2682 trades, PF 1.175, Sharpe 0.51, DSR 0.65 (N 32), 60 symbols | fail -> **fail** |
| `stk-pairs-zscore` | 784 trades, PF 0.765, Sharpe -1.2, DSR 0.0 (N 32), 18 symbols | 784 trades, PF 0.765, Sharpe -1.2, DSR 0.0 (N 32), 18 symbols | fail -> **fail** |
| `stk-gap-and-go-h1` | 1301 trades, PF 0.761, Sharpe -1.335, DSR 0.0 (N 32), 55 symbols | 1301 trades, PF 0.761, Sharpe -1.335, DSR 0.0 (N 32), 55 symbols | fail -> **fail** |
| `etf-overnight-hold` | 2422 trades, PF 0.626, Sharpe -2.492, DSR 0.0 (N 8), 17 symbols | 2422 trades, PF 0.626, Sharpe -2.492, DSR 0.0 (N 8), 17 symbols | fail -> **fail** |
| `etf-ts-xs-momentum` | 1032 trades, PF 1.252, Sharpe 0.511, DSR 0.842 (N 32), 17 symbols | 1032 trades, PF 1.252, Sharpe 0.511, DSR 0.842 (N 32), 17 symbols | fail -> **fail** |
| `fx-ts-xs-momentum` | 2544 trades, PF 0.609, Sharpe -2.374, DSR 0.0 (N 16), 9 symbols | 2544 trades, PF 0.609, Sharpe -2.374, DSR 0.0 (N 16), 9 symbols | fail -> **fail** |
| `etf-intraday-momentum` | 10645 trades, PF 0.217, Sharpe -14.081, DSR 0.0 (N 4), 17 symbols | 10645 trades, PF 0.217, Sharpe -14.081, DSR 0.0 (N 4), 17 symbols | fail -> **fail** |
| `etf-intraday-momentum-m30` | 10397 trades, PF 0.197, Sharpe -14.186, DSR 0.0 (N 4), 16 symbols | 10397 trades, PF 0.197, Sharpe -14.186, DSR 0.0 (N 4), 16 symbols | fail -> **fail** |
| `stk-earnings-jump` | 230 trades, PF 1.267, Sharpe 0.41, DSR 0.69 (N 16), 60 symbols | 230 trades, PF 1.267, Sharpe 0.41, DSR 0.69 (N 16), 60 symbols | fail -> **fail** |
| `stk-earnings-jump-continuation` | 313 trades, PF 1.242, Sharpe 0.453, DSR 0.792 (N 16), 60 symbols | 313 trades, PF 1.242, Sharpe 0.453, DSR 0.792 (N 16), 60 symbols | fail -> **fail** |
| `stk-pead-ear` | 463 trades, PF 1.053, Sharpe 0.127, DSR 0.407 (N 16), 60 symbols | 463 trades, PF 1.053, Sharpe 0.127, DSR 0.407 (N 16), 60 symbols | fail -> **fail** |
| `etf-pre-fomc-drift-h1` | 96 trades, PF 1.81, Sharpe 0.65, DSR 0.902 (N 4), 4 symbols | 96 trades, PF 1.81, Sharpe 0.65, DSR 0.902 (N 4), 4 symbols | fail -> **fail** |
| `etf-pre-fomc-drift-d1` | not run | 551 trades, PF 0.816, Sharpe -0.232, DSR 0.065 (N 8), 13 symbols | new: **fail** |
| `etf-risk-on-trend` | 1224 trades, PF 1.205, Sharpe 0.319, DSR 0.521 (N 48), 13 symbols | 1224 trades, PF 1.205, Sharpe 0.319, DSR 0.521 (N 48), 13 symbols | fail -> **fail** |
| `etf-corr-calm-trend` | 1238 trades, PF 0.976, Sharpe -0.003, DSR 0.166 (N 48), 13 symbols | 1238 trades, PF 0.976, Sharpe -0.003, DSR 0.166 (N 48), 13 symbols | fail -> **fail** |
| `etf-panic-rebound` | 82 trades, PF 2.451, Sharpe 0.348, DSR 0.853 (N 16), 13 symbols | 82 trades, PF 2.451, Sharpe 0.348, DSR 0.847 (N 32), 13 symbols | fail -> **fail** |
| `etf-vix-panic-rebound` | 97 trades, PF 1.77, Sharpe 0.273, DSR 0.79 (N 16), 13 symbols | 97 trades, PF 1.77, Sharpe 0.273, DSR 0.784 (N 32), 13 symbols | fail -> **fail** |
| `etf-panic-rebound-pool22` | not run | 83 trades, PF 2.128, Sharpe 0.42, DSR 0.857 (N 48), 22 symbols | new: **fail** |
| `etf-vix-panic-rebound-pool22` | not run | 109 trades, PF 1.326, Sharpe 0.206, DSR 0.603 (N 64), 22 symbols | new: **fail** |
| `etf-panic-rebound-pool34` | not run | 90 trades, PF 1.914, Sharpe 0.37, DSR 0.84 (N 80), 34 symbols | new: **fail** |
| `etf-vix-panic-rebound-pool34` | not run | 119 trades, PF 1.648, Sharpe 0.314, DSR 0.591 (N 96), 34 symbols | new: **fail** |
| `stk-rs-momentum-crash-protected` | 1480 trades, PF 1.318, Sharpe 0.658, DSR 0.949 (N 16), 60 symbols | 1480 trades, PF 1.318, Sharpe 0.658, DSR 0.949 (N 16), 60 symbols | fail -> **fail** |
| `fx-currency-strength-momentum` | 1174 trades, PF 0.625, Sharpe -1.62, DSR 0.0 (N 16), 9 symbols | 1174 trades, PF 0.625, Sharpe -1.62, DSR 0.0 (N 16), 9 symbols | fail -> **fail** |
| `fx-carry-trend` | 1138 trades, PF 0.686, Sharpe -1.269, DSR 0.0 (N 16), 9 symbols | 1138 trades, PF 0.686, Sharpe -1.269, DSR 0.0 (N 16), 9 symbols | fail -> **fail** |
| `fx-carry-vol-filter` | 1301 trades, PF 0.612, Sharpe -1.76, DSR 0.0 (N 16), 9 symbols | 1301 trades, PF 0.612, Sharpe -1.76, DSR 0.0 (N 16), 9 symbols | fail -> **fail** |
| `fx-rate-diff-trend` | 576 trades, PF 0.583, Sharpe -1.551, DSR 0.0 (N 16), 9 symbols | 576 trades, PF 0.583, Sharpe -1.551, DSR 0.0 (N 16), 9 symbols | fail -> **fail** |

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
- OpenD history starts 2006-09 for daily bars and 2018-09 for 60-minute bars; Oanda history here is 10 years. Of the 12 round-3 ETFs, OpenD serves daily bars from 2006 only for XRT, IBB, EWZ and FXI; the other eight start in 2012 (their own listings are older), so the point-in-time screen admits them from 2012.
- The daily pre-FOMC flags assume the exchange calendar is known in advance (it is published years ahead): the backtest reads it from the bars. Scheduled meetings only; the March 2020 meeting, replaced by the 15 March emergency cut, is not in the calendar, so no trade was planned for it. The statement-day close includes about two hours of reaction after the 14:00 (14:15 before 2013) statement.
- DSR's N is every parameter set ever recorded for the strategy in the trial ledger, across all variants.
