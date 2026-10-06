# Profit upgrades: wiring patches (for Ray)

The new modules in `tradex/profit/` are complete and tested on their own. They change nothing in the running system until the
core calls them. That wiring is here as patches so the files it touches are reviewed by their owners. Nothing here is applied. (`core-wiring.patch` was reviewed and applied as a commit on 5 Oct, see the end of this file.)
Each patch is checked by `tests/profit/test_profit_patches.py`: it applies the patch to a scratch copy and runs the patch's own
tests plus the existing suites around the files it changes. Check one yourself with `git apply --check patches/ov-exits/<file>`.

| Patch | Touches | Protected? | Needs |
|---|---|---|---|
| `dashboard-gate-verdicts.patch` | `tradex/dashboard/views.py`, `tradex/dashboard/static/app.js` | no | the dashboard's owner |
| `es-tail-count.patch` | `tradex/risk/exposure.py`, new `tests/test_es_tail_count.py` | **yes** | Ray |

## What the core wiring does (applied)

Adds one optional argument, `profit=None`, to `TradingCore`, `build_replay_core`/`run_replay` and `build_runtime`. With `profit=None`
(the default) nothing changes, and a `ProfitHooks()` with nothing switched on yields the identical decision digest (tested). The
call sites, all ensemble-book only:

1. `_decide`: the plan gets its calibrated probability and EV (`p_source="model"`) only when the model is calibrated and beats the raw
   base rate out of sample. Otherwise it is left exactly as the finaliser wrote it.
2. Before the order: vol targeting multiplies the quantity the gate already sized by a factor of at most 1.0. It can only shrink.
   The gate remains the ceiling; the order guard still sees a quantity no larger than the verdict.
3. The entry order's take-profit is the last target when partials are configured (otherwise a venue-side order would close everything
   at target 1 before the engine takes its partial).
4. `_review_positions`: the reviewer's actions and the exit engine's are merged into one close, the tightest stop, and a reduce
   expressed as a share of what is open now. The core still places every order itself.
5. Once per New York day (with the snapshot): refit the probability model and recalibrate the cost models from the ledger as of that
   day, swapping `CalibratedCosts` into the core's own cost dict. A failed refit is a `Health` row with `check="profit"`, not a crash.

To switch it on, build the hooks where the runtime is built and pass them in:

```python
from tradex.profit.exits import load_exit_policy
from tradex.profit.hooks import ProfitHooks
from tradex.profit.calibration import CalibrationConfig
from tradex.profit.costcal import CostCalConfig
from tradex.profit.suggest import VolTargetPolicy
hooks = ProfitHooks(exit_policy=load_exit_policy(), calibration=CalibrationConfig(), cost_cfg=CostCalConfig(),
                    vol_policy=VolTargetPolicy())
build_runtime(..., profit=hooks)        # or run_replay(..., profit=hooks)
```

Each piece is independent: leave an argument out and that piece stays off.

Known limits of this wiring:

- **Partial targets execute at the next open, not at the target price.** The venue carries one take-profit per entry. With partials the
  engine notices the bar that crossed target 1 and sends a market reduce; the fill is at the next open. A reduce-only resting limit at
  target 1 in the venue adapter (execution/, protected, the Mac session's) would fix that; until then partials give up some of the
  edge in the target.
- **Cost calibration is only as good as the adapters' `spread_slippage_usd`.** It compares the fill's booked spread-and-slippage dollars
  with the model's. If a live adapter books 0 there (rather than fill price against mid at order time), the ratio is under 1, the floor
  of 1.0 holds, and calibration silently does nothing. Check one paper fill from each adapter.
- **The exit-engine defaults are a starting point, not a result.** `config/exits.yaml` has not been shown to beat the plain exits.
  On the synthetic replay data in the tests it is slightly worse (paired difference about -0.1R, t about -1.4), which says nothing about
  real data. Run `python -m tradex.profit exits --ledger <paper ledger> --data <bars dir>` on forward paper plans first; it reports the
  paired R difference against the plain stop/target/time-stop and how many plans it could follow. Do not switch the engine on for the
  live book on backtest numbers.

## dashboard-gate-verdicts.patch

The dashboard's "What each filter saved or cost" table reads `filter_report`, which only says how many plans a gate blocked and their mean
R. This swaps in `tradex.profit.whatif.gate_rows`: same keys, plus the net effect of blocking in R, a 90% interval on the mean, and a verdict
(`saves R`, `costs R`, `inconclusive`, or `too few` below 20 followed plans). Two columns are added to the table.

## es-tail-count.patch (protected, yours)

`ESModel.es` takes the worst `ceil(n * (1 - alpha))` days. In floating point `600 * (1 - 0.975)` is `15.000000000000013`, so the tail is 16
days, not 15, and expected shortfall comes out slightly too small whenever the history length makes the tail a whole number. One line, a
test that fails before and passes after. It moves the gate's expected shortfall up (more conservative) by a fraction of a percent for those
lengths; nothing else. The ruin radar already uses the corrected count.

## Decisions for you

1. **Vol-target upsizing.** The gate's `size_factor` only shrinks today, so vol targeting is shrink-only. Letting calm periods size up (to
   1.5x, only with a calibrated edge, capped by quarter-Kelly on the lower bound, still under every gate limit) needs the gate's per-trade
   cap applied after the factor instead of before it. That is a protected change; I have not written it. Say if you want it.
2. **Cost-calibration floor.** The model may only get more pessimistic (`floor=1.0`). Lower it in `CostCalConfig` once a few hundred real
   fills agree.
3. **Shock windows.** `config/shock_windows.yaml` has twelve episodes with dates from memory and no magnitudes; sizes come from price history.
   Check the dates, and note the radar reports a window as `no data` when the bars directory does not reach back to it. FX history from 2015
   needs the Oanda history download (`tradex fetch`).

## Found while wiring (not fixed here)

`tradex/core/loop.py::_decide` passes the agent size factor to the gate (`gate.review(..., factor, ...)`, which multiplies risk by it) and then
shrinks the gate's quantity by the same factor again (`shrink_qty(verdict.qty, factor)`). A `shrink 0.5` from an agent therefore sizes at about
0.25. `tests/test_core_actions.py` does not cover the at-gate path. The vol-target factor in this patch is applied once, after the gate, so it
does not compound with this; the agent path is the core owner's to fix.

## Applied: core-wiring (reviewed 5 Oct)

Reviewed and applied as its own commit on top of the fix for the double agent shrink. With `profit=None` nothing changes, and a
`ProfitHooks()` with nothing on yields the identical digest (tested). Changes made while applying it:

- It sat on the old size code, where an agent's shrink factor also went into the gate and then shrank the quantity a second time. The
  fix landed first: the gate now gets only the calendar's halving, and an agent's shrink applies once, after the gate. Vol targeting
  multiplies the quantity after that, so each of the three factors acts once.
- The agent's "leaves no size" check now runs before vol targeting, so a quantity an agent shrank to zero is blocked as `agent`, not
  as `vol_target`.
- `ProfitHooks.size_factor` documented how it is applied: after the gate, to the approved size, not inside it.
- `CalibratedCosts` calls the base model's `fill`, so a strict paper/live cost model (`OandaFxCosts(mode=..., spread_source=...)`) still
  refuses to price a fill without a measured spread.
