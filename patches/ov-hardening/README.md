# Patches from the hardening pass (claude/ov-hardening)

For Ray: two of these touch protected paths, so they are patch files, not commits. Nothing here is applied.
Each patch also deletes the strict-xfail markers of the gap tests it fixes (`tests/test_gaps_*.py`), so the
suite stays green after it is applied. Check with `git apply --check patches/ov-hardening/<file>` first.

| Patch | Touches | Fixes | Needs |
|---|---|---|---|
| `execution-guard-g2.patch` | `tradex/execution/guard.py` (protected) | G2, and the guard half of G1 | Ray |
| `protected-path-check-s7.patch` | `tools/check_protected_paths.py` (protected) | S7 | Ray's decision (see below) |
| `lane-fixes-g1-g10.patch` | `tradex/core/inbox.py`, `tradex/core/ledger.py` | G1 (trigger forging), G10 (lost inbox request) | the lane owners, or Ray; not protected |

## execution-guard-g2.patch

- A verdict is a budget: `OrderGuard.check` reserves the order's quantity against the verdict, so several orders
  citing one verdict cannot add up past its size. A resize (cancel, then re-place smaller) works because cancelling
  releases the reservation: `GuardedBroker.cancel` does it, and `VenueAdapter` subclasses get it automatically
  (`__init_subclass__` wraps their `cancel`). A venue error during `place` releases too.
- An entry without a stop loss is refused.
- `ledger_verdicts` also returns the plan; the order's symbol and side must match it and its stop may not be looser.
- `ledger_verdicts` only trusts a verdict row whose stored hash matches its own fields chained to the row before it,
  so a row inserted by a trigger or a raw connection is not a verdict. (A sha256 chain is not keyed: a writer who can
  compute hashes can still forge. A keyed HMAC written by the core is the next step, if you want it.)
- One existing test changes: `test_every_replay_entry_order_cites_a_verdict_the_guard_accepts` now passes the plan's
  stop, as the core does.
- Venue adapters (Mac session) need no change if they subclass `VenueAdapter` and implement `cancel`.

## protected-path-check-s7.patch

Makes `tools/check_protected_paths.py` treat `claude/` branches like `agent/` ones. Effect: every Claude lane PR that
changes `config/risk/`, `config/gates/`, `tradex/risk/`, `tradex/execution/`, `.github/workflows/` or
`config/accounts.yaml` goes red until you merge it yourself, which is what the gap review asked for. It will also turn
red the lane PRs already open that touch those paths. `test_protected_paths_check` is updated to match. Skip this patch
if you prefer to keep reviewing those PRs by eye.

## lane-fixes-g1-g10.patch

- `Mailbox` authorizer now also denies CREATE/DROP TRIGGER, CREATE VIEW, ATTACH/DETACH, writes to `sqlite_master`, and
  every PRAGMA except a read-only list (`writable_schema` was the second way to plant a trigger).
- `Ledger` (writer) refuses to open a file that has any trigger in it: the schema defines none.
- `ingest_inbox` writes the `agent_output` ledger row before it marks the inbox row applied, so a crash in between
  re-runs the (idempotent) handler instead of losing the request.
