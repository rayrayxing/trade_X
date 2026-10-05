# Proposed strategies from the catalog

Eleven new strategy specs in `strategies/proposed/`, all `status: proposed`. None has been run on real data, so none is validated; the walk-forward gate on Ray's Mac is what decides. Each spec is built from the existing spec and expression language; the new inputs they need are research columns built by small new modules (see "Primitives"), read through `data.column` the same way the specs in `research/specs/` already do. No spec contains a date, rate or price: calendars and rates are injected from files on the Mac, and the gate reports "needs data" when a file is missing instead of running on anything else.

The 27 catalog entries marked Build, and where they stand:

| Catalog entry (Build) | Result |
|---|---|
| 52-week-high momentum; relative strength vs sector; pairs; gap-and-go; overnight vs intraday; joint time-series and cross-sectional (ETFs) | Already specced in `research/specs/` and gated (phase-1 run) |
| Earnings-day jump continuation; pre-FOMC drift; hidden-Markov regime allocation; realised-covariance regime detection | New specs below |
| Joint time-series and cross-sectional (FX form) | New spec below (second spec on that entry) |
| Deep momentum LSTM, Momentum Transformer, changepoint momentum, adaptive-lookback momentum, learning to rank, deep-learning stat-arb, deep chart-pattern recognition | Not built: each needs a trained model. Out of scope for the spec language |
| Large-tick trend filter | Not built: needs tick-size data per symbol and a filter hook on trend specs; the filter is not a signal |
| Uncertainty-gated ranker | Not built: gates a ranker that does not exist |
| Regime-switching volatility forecast for sizing | Not built as a spec: it is a sizing input. The volatility percentile it would use is `rg_vol_pct` in `tradex/research/regime.py` |
| Long calls or puts on high-conviction plans | Not a signal (execution style, needs live option chains) |
| Pre-earnings run-up and IV crush (options) | Not built: options are outside the spec language's asset classes, and need IV history |
| Intraday momentum, first half-hour predicts the last | Not built: the engine fills exits at the next open and cannot exit at the 16:00 close |
| LLM news sentiment; ChatGPT headline scoring | Not built: needs headline history |
| Short high borrow fee / high short interest; short-seller flow | Not built: needs borrow-fee and short-volume history from OpenD, not downloaded |

Six catalog entries that are not marked Build also got a spec, because the new primitives fit them: post-earnings drift, momentum with crash protection, G10 carry, carry with a volatility filter, rate-differential trend (all "Needs data") and buy equity after a volatility spike (Screened: positive).

## How to run one on the Mac

1. Check the specs and primitives still pass: `python -m tradex check strategies && pytest -q`
2. Make sure the bars exist (OpenD must be running; both commands are read-only):
   - US daily and hourly: `python -m tradex.research.universe opend`
   - FX daily/H4/H1 from Oanda practice: `python -m tradex.research.universe oanda`
3. Put the extra input file(s) for the strategy in place (the "Inputs" column below; formats in the next section).
4. Run the gate on that strategy only: `python -m tradex.research.gate <strategy-id>`. It prints one line (`pass`, `fail <rung>`, or `needs data <what>`) and writes `research/results/single/<strategy-id>.json` with the OOS statistics, fold table and gate ladder. It runs the same walk-forward, costs, global trial ledger (DSR's N) and protected thresholds as the phase-1 run. `python -m tradex.research.gate` with no argument still runs everything and writes `phase1_gate.{json,md}`.

Every gate run adds its parameter sets to the trial ledger, so run each strategy once with the grid as written; re-running to "see" does not reset N.

## Inputs the Mac has to supply

| File | Layout | Used by | Where to get it |
|---|---|---|---|
| `data/calendar/earnings/<SYMBOL>.csv` | `date,when`, one row per announcement; `when` is `bmo`, `amc` or blank. Session date of the announcement | the two earnings specs | Not available from anything in this repo. A historical earnings-date list per symbol from a vendor or the companies' IR pages; whether OpenD exposes one has not been checked. Supplying `when` makes the reaction session exact; blank is handled (see below) |
| `data/calendar/fomc_history.csv` | one column `time`, UTC ISO instants of the FOMC statement (14:00 New York) | pre-FOMC | Federal Reserve FOMC historical calendars (2018 onward is enough: hourly bars start 2018-09). `data/calendar/central_banks.yaml` only covers 2026-2027 and is also readable by `FileEventCalendar` |
| `data/rates/policy_rates.csv` | `date,currency,rate` with rate in percent, effective date, currencies USD EUR GBP JPY AUD CAD CHF NZD | the three carry specs (and FX financing cost in their gate runs, so signal and cost use the same history) | Central-bank decision histories, or Oanda financing history. `tradex/costs/policy_rates.csv` has the same layout but is an estimate written from memory; copying it gives a smoke run only and should not decide anything |

Missing files are not an error: the strategy shows as "needs data" with the path it looked for.

## Primitives (new files, all causal, all tested)

| Module | What it provides | Injected interface |
|---|---|---|
| `tradex/research/sources.py` | Protocols `EarningsCalendar`, `EventCalendar`, `RateSource`; file readers `CsvEarningsCalendar`, `FileEventCalendar`, `CsvRateSource`; `DataUnavailable` | the three protocols |
| `tradex/research/events.py` | `earnings_columns`: reaction return, gap, z-score against prior volatility, volume ratio, session age, return since reaction, sessions to next announcement; `event_window_columns`: entry and exit flags for the 24 hours before an announcement | an `EarningsCalendar` / `EventCalendar` (or the frames they return) |
| `tradex/research/rel_strength.py` | `rs_momentum` (ratio momentum with a skip), `rs_trend`, `currency_strength`, `pair_strength_diff` | none (takes closes) |
| `tradex/research/carry.py` | `carry_columns`: rate differential, carry per unit of volatility, rate change, trend, carry-trend agreement, volatility percentile | a `RateSource` |
| `tradex/research/regime.py` | realised-volatility percentile, drawdown from high, average pairwise correlation, absorption ratio and its shift, risk-on score across equities/bonds/gold/volatility, momentum-crash state, and `regime_columns` bundling them | none (takes closes) |
| `tradex/research/builders.py` | One builder per data shape; reads the OpenD/Oanda caches and the files above and attaches the columns | files above |

Earnings timing: when `when` is blank, the reaction is the larger-moving of the announcement session and the next one, and the columns start at the later session's close, so nothing is used before it is knowable. `ed_days_to` reads the schedule ahead of time, which is fine for scheduled dates and wrong for a date that moved; no spec uses it.

## The strategies

All stock specs use the same 29-name hand-picked universe as the existing stock specs (survivorship bias applies); FX specs use the nine pairs in `tradex/research/universe.py`.

### stk-earnings-jump-continuation
- Catalog: Earnings-day jump continuation (Build). Source: Christensen et al., arXiv 2601.08962. The paper is intraday; this is the daily-bar form.
- Idea: an earnings reaction session of more than 2 standard deviations on more than 1.5x normal volume, closing in the direction of the move, keeps drifting for a few sessions. Enter at the next open, hold up to 5 bars.
- Data: daily bars, earnings dates (`data/calendar/earnings/`).
- Run: `python -m tradex.research.gate stk-earnings-jump-continuation`

### stk-pead-ear
- Catalog: Post-earnings-announcement drift (Needs data). Sources: Bernard and Thomas 1989; arXiv 2009.03094. Price-reaction form (earnings announcement return, benchmark-adjusted against SPY), which does not need EPS estimates, the data the catalog said was missing.
- Idea: enter long or short the second session after a reaction of more than 1.5 standard deviations that has not reversed, hold up to 40 bars.
- Data: daily bars, SPY daily bars, earnings dates.
- Run: `python -m tradex.research.gate stk-pead-ear`

### etf-pre-fomc-drift-h1
- Catalog: Pre-FOMC announcement drift (Build). Source: Lucca and Moench 2015.
- Idea: long SPY/QQQ/IWM/DIA from about 24 hours before the statement until 30 minutes before it (fills at the 13:30 New York open on both days). The drift has weakened in later samples, so a fail is a plausible outcome.
- Data: 60-minute bars (from 2018-09 in OpenD), FOMC history.
- Note: `max_bars` is 16, not 7. The position reviewer closes any trade that is flat after half of `max_bars`, which would cut the 7-bar hold short; the real exit is the `evt_leave` signal.
- Run: `python -m tradex.research.gate etf-pre-fomc-drift-h1`

### etf-risk-on-trend
- Catalog: Hidden-Markov regime allocation (Build, regime input), arXiv 2605.27848. No HMM is fitted; the state is a transparent score (equity above 200-day mean, equity beating bonds, beating gold, calm volatility).
- Idea: trade ETF uptrends (close above the 100-day EMA) only when the score is 3 or more; leave at 1 or below.
- Data: daily bars for the 13 ETFs plus SPY, TLT, GLD and the nine sector ETFs (all in the phase-1 download).
- Run: `python -m tradex.research.gate etf-risk-on-trend`

### etf-corr-calm-trend
- Catalog: Realised-covariance regime detection (Build), arXiv 2104.03667; absorption ratio from Kritzman, Li, Page and Rigobon 2011.
- Idea: ETF uptrends only while the sector ETFs' absorption ratio has not risen against its one-year history (shift below 1 standard deviation); leave above 1.5.
- Data: as above.
- Run: `python -m tradex.research.gate etf-corr-calm-trend`

### etf-panic-rebound
- Catalog: Buy equity after a VIX spike above 30 (Screened: positive). VIX is not in the data set, so realised volatility (top decile of its history) stands in.
- Idea: first up day after volatility above its 90th percentile with the ETF more than 10% off its one-year high; exit when volatility normalises. Signals are rare; the 100-trade rung may be the one that fails.
- Data: as above.
- Run: `python -m tradex.research.gate etf-panic-rebound`

### stk-rs-momentum-crash-protected
- Catalog: Momentum with crash protection (Needs data). Sources: Barroso and Santa-Clara 2015; Daniel and Moskowitz 2016.
- Idea: top fifth of the universe by 12-month relative-strength momentum against SPY (skipping the latest month), long only, flat in the bear-market-plus-high-volatility state. The point-in-time universe problem the catalog flags still applies, so the result carries survivorship bias.
- Data: daily bars for the stocks, SPY, TLT, GLD, sector ETFs.
- Run: `python -m tradex.research.gate stk-rs-momentum-crash-protected`

### fx-currency-strength-momentum
- Catalog: Joint time-series and cross-sectional strategy (Build), arXiv 2302.10175; currency momentum, Menkhoff et al. 2012 (the catalog's FX momentum screen found little edge since 2000, so expect a hard gate).
- Idea: rank pairs by base-currency strength minus quote-currency strength over 63 days; long the top fifth and short the bottom fifth when the pair's own 63-day return agrees.
- Data: Oanda daily bars for the nine pairs.
- Run: `python -m tradex.research.gate fx-currency-strength-momentum`

### fx-carry-trend
- Catalog: G10 carry (Needs data). Sources: Lustig et al. 2011; Menkhoff et al. 2012.
- Idea: top 30% by rate differential with positive carry and a positive 100-day trend, long; mirrored short; exit when either turns.
- Data: Oanda daily bars, `data/rates/policy_rates.csv`.
- Run: `python -m tradex.research.gate fx-carry-trend`

### fx-carry-vol-filter
- Catalog: Carry with volatility filter (Needs data), arXiv 2101.09738. The pair's own 21-day realised volatility percentile replaces VIX.
- Idea: same carry ranking, entries only below the 70th volatility percentile, exit above the 90th.
- Data: as above.
- Run: `python -m tradex.research.gate fx-carry-vol-filter`

### fx-rate-diff-trend
- Catalog: Rate-differential trend FX (Needs data). Source: Lustig, Roussanov and Verdelhan 2011.
- Idea: the change in the rate differential over 63 bars (more than 25 bp) with an agreeing 100-day price trend, long or short.
- Data: as above.
- Run: `python -m tradex.research.gate fx-rate-diff-trend`

## Reading the results

- A carry strategy's cost model pays or earns financing from the same rate file, so the gate sees the carry it is trading. Without that file the strategy does not run.
- The three carry specs and the three regime-gated ETF specs overlap heavily within their group (same family, so the ensemble gives them one vote). Treat a pass by one as evidence about the idea, not three independent passes.
- A pass is only the start: check the robustness lines in the single-run JSON and run the paper loop before any promotion. Nothing here is marked validated.
