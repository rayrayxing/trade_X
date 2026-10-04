# Phase-1 gate run

Run 2026-10-04 15:18 UTC on real bars only (moomoo OpenD, qfq-adjusted). Stage-3 thresholds from `config/gates/thresholds.yaml`: {'min_trades': 100, 'min_profit_factor': 1.2, 'min_dsr': 0.95, 'max_drawdown': -0.5, 'min_positive_folds': 0.5}.
Walk-forward: {'n_folds': 5, 'initial_train_frac': 0.4, 'anchored': True, 'grid_points': 4, 'max_trials': 48, 'min_train_trades': 10, 'seed': 0}. Engine: {'initial_equity': 10000.0, 'risk_pct': 1.0, 'max_leverage': 1.0}. Costs: moomoo SG fees and default spread/slippage for US stocks; FX would use Oanda bid/ask.

**1 of 11 strategies run pass the gate.** 10 need data, 14 are not built or not standalone.

Note on `stk-breakout-volume`: it passes, but fails the robustness check(s) other_large_caps; treat it as unproven.

## Strategies run

| Strategy | Catalog entry | Market | Bars used | OOS trades | PF | Sharpe | DSR (N) | Max DD | Folds + | Result | Failing rung |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `stk-ema-pullback-swing` | EMA pullback swing (seed) | US stocks/ETFs (moomoo) D1, 9 symbols, 2006-09-20..2026-10-02 | 42,962 | 1075 | 1.333 | 0.713 | 0.453 (16) | -0.245 | 4/5 | **fail** | deflated_sharpe |
| `stk-macd-trend-candle` | MACD trend with candle confirmation (seed) | US stocks/ETFs (moomoo) D1, 9 symbols, 2006-09-20..2026-10-02 | 42,962 | 93 | 1.044 | 0.03 | 0.292 (16) | -0.119 | 3/5 | **fail** | trades |
| `stk-rsi2-meanrev` | RSI(2) pullback in uptrend (seed) | US stocks/ETFs (moomoo) D1, 3 symbols, 2006-09-20..2026-10-02 | 15,093 | 286 | 1.1 | 0.142 | 0.494 (16) | -0.169 | 3/5 | **fail** | profit_factor |
| `stk-breakout-volume` | Breakout with volume (seed) | US stocks/ETFs (moomoo) D1, 5 symbols, 2006-09-20..2026-10-02 | 22,825 | 177 | 1.907 | 0.829 | 0.99 (16) | -0.082 | 4/5 | **pass** | - |
| `stk-double-bottom-breakout` | Double bottom breakout (seed) | US stocks/ETFs (moomoo) D1, 7 symbols, 2006-09-20..2026-10-02 | 32,905 | 301 | 1.467 | 0.677 | 0.925 (16) | -0.163 | 3/5 | **fail** | deflated_sharpe |
| `stk-52w-high-momentum` | 52-week-high momentum | US stocks/ETFs (moomoo) D1, 29 symbols, 2006-09-20..2026-10-02 | 142,623 | 2044 | 1.141 | 0.414 | 0.834 (16) | -0.264 | 4/5 | **fail** | profit_factor |
| `stk-sector-relative-strength` | Relative strength vs sector | US stocks/ETFs (moomoo) D1, 29 symbols, 2006-09-20..2026-10-02 | 142,623 | 2635 | 1.241 | 0.671 | 0.93 (16) | -0.331 | 5/5 | **fail** | deflated_sharpe |
| `stk-pairs-zscore` | Pairs trading (distance and cointegration) | US stocks/ETFs (moomoo) D1, 22 symbols, 2006-09-20..2026-10-02 | 109,718 | 1772 | 0.861 | -0.823 | 0.0 (16) | -0.516 | 1/5 | **fail** | profit_factor |
| `stk-gap-and-go-h1` | Gap-and-go after news gap | US stocks/ETFs (moomoo) H1, 7 symbols, 2018-09-28..2026-10-02 | 98,273 | 531 | 0.98 | -0.063 | 0.177 (16) | -0.245 | 2/5 | **fail** | profit_factor |
| `etf-overnight-hold` | Overnight vs intraday return split | US stocks/ETFs (moomoo) H1, 13 symbols, 2018-09-28..2026-10-02 | 182,507 | 3670 | 0.618 | -2.444 | 0.0 (4) | -0.789 | 0/5 | **fail** | profit_factor |
| `etf-ts-xs-momentum` | Joint time-series and cross-sectional strategy | US stocks/ETFs (moomoo) D1, 15 symbols, 2006-09-20..2026-10-02 | 68,866 | 1457 | 1.037 | 0.133 | 0.543 (16) | -0.389 | 3/5 | **fail** | profit_factor |

Failing reasons:

- `stk-ema-pullback-swing`: deflated Sharpe 0.45 (need 0.95)
- `stk-macd-trend-candle`: only 93 out-of-sample trades (need 100); profit factor 1.04 after costs (need 1.2); deflated Sharpe 0.29 (need 0.95)
- `stk-rsi2-meanrev`: profit factor 1.10 after costs (need 1.2); deflated Sharpe 0.49 (need 0.95)
- `stk-breakout-volume`: none
- `stk-double-bottom-breakout`: deflated Sharpe 0.92 (need 0.95)
- `stk-52w-high-momentum`: profit factor 1.14 after costs (need 1.2); deflated Sharpe 0.83 (need 0.95)
- `stk-sector-relative-strength`: deflated Sharpe 0.93 (need 0.95)
- `stk-pairs-zscore`: profit factor 0.86 after costs (need 1.2); deflated Sharpe 0.00 (need 0.95); max drawdown -52% (limit -50%); only 20% of test folds profitable
- `stk-gap-and-go-h1`: profit factor 0.98 after costs (need 1.2); deflated Sharpe 0.18 (need 0.95); only 40% of test folds profitable
- `etf-overnight-hold`: profit factor 0.62 after costs (need 1.2); deflated Sharpe 0.00 (need 0.95); max drawdown -79% (limit -50%); only 0% of test folds profitable
- `etf-ts-xs-momentum`: profit factor 1.04 after costs (need 1.2); deflated Sharpe 0.54 (need 0.95)

## Robustness of the passes (not part of the gate)

- `stk-breakout-volume`, spread and slippage doubled: 177 trades, PF 1.858, Sharpe 0.782, DSR 0.985, max DD -0.085, folds 4/5: **pass**
- `stk-breakout-volume`, same rules on 24 other large caps: 584 trades, PF 1.0, Sharpe 0.019, DSR 0.211, max DD -0.223, folds 3/5: **fail** (profit factor 1.00 after costs (need 1.2); deflated Sharpe 0.21 (need 0.95))

## Not run

| Catalog entry | Status in catalog | Result | Why |
|---|---|---|---|
| Donchian channel breakout | Seed; screened: no FX edge since 2016 | needs data | Oanda practice bid/ask history not downloaded (no oanda_token in the Keychain yet) |
| Trend pullback with engulfing candle (seed) | Seed | needs data | Oanda practice bid/ask history not downloaded (no oanda_token in the Keychain yet) |
| RSI range reversion (seed) | Seed | needs data | Oanda practice bid/ask history not downloaded (no oanda_token in the Keychain yet) |
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
| Intraday momentum (first half-hour predicts last) | Build | not built | trades the last half hour; the engine fills exits at the next open, so it cannot exit at the 16:00 close yet |
| Earnings-day jump continuation | Build | needs data | historical earnings dates |
| Pre-earnings run-up and IV crush (options) | Build | needs data | historical earnings dates and option implied volatility |
| Pre-FOMC announcement drift | Build | needs data | historical FOMC dates (data/calendar covers 2026-2027 only) |
| LLM news sentiment long-short | Build | needs data | news headline history |
| ChatGPT headline scoring | Build | needs data | news headline history |
| Short high borrow-fee, high short-interest names | Build | needs data | borrow-fee and short-interest history (not downloaded in phase 1) |
| Short-seller flow signal | Build | needs data | short-volume history (not downloaded in phase 1) |

## Caveats

- US large caps are today's names, so cross-sectional and pairs results carry survivorship bias.
- The stock seeds' universes were picked in 2026 and lean to that period's biggest winners (NVDA, AMD, TSLA, META, AMZN). Their trial count does not include that choice, so a pass that fails on other large caps (see robustness) is most likely hindsight selection, not an edge.
- No historical earnings calendar: the seeds' no_earnings_3d filter was inactive.
- Stock spreads are the cost model's defaults (1 bp half-spread + 2 bp slippage), not measured quotes.
- Fixed moomoo fees (US$0.99 + 9% GST per order) weigh heavily at the US$10,000 test equity.
- OpenD history starts 2006-09 for daily bars and 2018-09 for 60-minute bars.
- DSR's N is every parameter set ever recorded for the strategy in the trial ledger.
