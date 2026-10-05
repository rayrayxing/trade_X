# Options: the protected-file patch (for Ray)

`options-order-guard.patch` changes the files agents may not change. It is **not applied** by
the options pull request. Review it, then apply it yourself on a non-agent branch:

```
git checkout -b ray/options-risk origin/claude/p1-gap-wire     # or main, once the Phase 1 stack has landed
git apply --check patches/options-order-guard.patch            # dry run
git apply patches/options-order-guard.patch
pytest -q tests/test_options_risk.py tests/test_execution.py tests/test_spine.py
```

The options pull request runs this for you on a scratch copy (`tests/options/test_opt_patch.py`):
the patched tree passes the patch's own tests, the existing gate/guard/runtime suites, and a check
that the gate's gap formula matches the one the planner and backtester use. Once the patch is on
main, delete the patch, this file and that test.

## Why it is needed

Without it, an options order would go through a gate and a guard built for shares and currency:

1. **Quantity would be priced as shares.** A contract covers 100 shares, so a US$2.50 call costs
   US$250, not US$2.50. The gate would size 100 times too big.
2. **A naked short call has no ceiling,** and the gate would size it on the distance to its stop.
3. **Nothing would force a stop on it,** keep it off earnings, or keep it off heavily shorted names.

## What the patch changes

| File | Change |
| --- | --- |
| `tradex/risk/options.py` (new) | The option rules in one protected place: the multiplier, the gap-stress formula, the earnings and short-interest blockers, the option-code parser. Imports nothing from `tradex.options`, so the unprotected package cannot change a risk number. |
| `tradex/risk/gate.py` | An options plan needs `option=OptionInfo(...)` or is rejected. Every USD figure carries the x100 multiplier. A long is sized on its whole premium. A naked short call must have a buy-to-close stop above its credit, is sized on the gap-stressed loss, and is rejected near earnings, on heavily shorted names, or on unknown data. Open option legs count in heat with their own stress loss; a book holding option legs with no multiplier or underlying data blocks every new trade (it fails closed). Stock and forex arithmetic is unchanged. |
| `tradex/risk/exposure.py` | `Leg` gains `multiplier`, `underlying`, `underlying_price`, `delta`, `right`, `stress_loss_usd` (all defaulted). Option legs map to their stock's equity factor, delta-adjusted. A short call is stressed as short shares, so the single-stock 20% gap scenario sees it in full. |
| `tradex/execution/guard.py` | Options entries may only be a long call, a long put or a sell-to-open call. A short call needs a stop above its credit on the order, no looser than the stop the verdict was sized for. The verdict must carry the options sizing the gate writes (x100, 20% gap, no blockers). Short puts, fractional contracts and non-option symbols are refused. Exits are untouched: closing orders still need no verdict. |
| `tradex/execution/checks.py` | `sanity_check(..., multiplier=1.0)`: notional is contracts x premium x 100. |
| `config/risk/policy.yaml` | New `options:` section and an `options` leverage cap on every governor tier. |
| `config/accounts.yaml` | `options` added to the moomoo SIMULATE account's asset classes. |
| `tests/test_options_risk.py` (new) | 38 tests for all of the above. |

## The rules, in numbers

- Multiplier 100. Long call or put: risk is the whole premium (`premium x 100 x contracts`).
- Naked call: the stop triggers at underlying `S_stop` (premium `P_stop`), then the underlying
  gaps 20% further, to `S_gap = 1.2 x S_stop`. The call is repriced there (Black-Scholes at the
  contract's IV, floored at intrinsic plus the time value it had at the stop; without IV, the hard
  bound `P_stop + S_gap - S_stop`), plus a 3% execution cushion because the buy-back pays the
  ask. Risk per contract is `(that price - credit) x 100`. The 20% is also floored in code, so a
  config edit cannot lower it.
- Naked call blockers: an earnings date inside the holding window plus one day; short interest
  above 20% of float or more than 5 days to cover; any of that data unknown.

## Decisions for you (numbers I had to pick)

1. **Option leverage cap:** `options: 0.30` of equity as gross premium (0.15 at tier 1). It limits
   premium outlay; naked-call risk is limited by heat and margin instead.
2. **`exec_cushion_pct: 0.03`:** half spread 2% + slippage 1%, the options cost model's defaults.
3. **Naked calls enabled: true** in the policy. Set `enabled: false` to ship long options first.
4. **Small accounts:** at 1% risk per trade a US$10,000 account cannot buy one 40-delta call on a
   US$100 stock (about US$400) and cannot hold even one naked call (the gap-stressed loss is over
   US$1,000). Long options need `cap_risk_pct` around 5%; naked calls need a much larger account.

## What the patch does not do (follow-ups)

- **Core wiring.** `tradex/core/loop.py` is not protected but is not touched here. It must pass
  `option=OptionInfo(...)` to `gate.review` and build option `Leg`s with the multiplier and
  underlying (`_legs`, around line 530). Until then option plans are rejected and a book holding
  option legs blocks all trades. That is deliberate: it fails closed.
- **The stop has to be armed at the venue.** moomoo documents no stop orders for US options
  (not in SIMULATE either), so the buy-to-close stop cannot rest at the broker. The guard makes
  sure a stop is on the order and sized for; the moomoo adapter (a protected path, your adapter
  patches) must register it with a monitor that sends the buy-to-close. Place that monitor before
  any naked call goes to SIMULATE.
- **Adapters and the simulator.** `tradex/execution/sim.py` and the moomoo adapter compute P&L,
  margin and fills as price x quantity; for options they need x100 as well.
- **Margin.** The gate takes the venue's free margin from the caller (`BookState.margin_max_qty`).
  Use the broker's own figure for naked calls; the Reg-T estimate in `tradex.options.sizing` is for
  backtests.
- **Account approval.** moomoo SG must approve the account for uncovered (naked) options; check the
  SIMULATE account's options level before relying on it.
- **Earnings and short-interest data** are not provided by OpenD. The planner must fill
  `OptionInfo` from the research data feed; unknown data blocks a naked call.
