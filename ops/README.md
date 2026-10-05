# Running trade_X on Ray's Mac

Everything here is for the one Mac that runs the paper-trading stack (Oanda practice and moomoo
SIMULATE). Nothing in `ops/` starts by itself: the installer copies files, and you load each service
with your own command when you are ready. Nothing here places an order or touches a live account.

| Service (launchd label) | What it runs | Needs |
|---|---|---|
| `com.tradex.core` | the paper trading loop (`tradex run --mode paper`), wrapped in `caffeinate` | venue adapters merged to main |
| `com.tradex.agent-worker` | queued agent jobs through the model gateway | a `tradex agents worker` command (does not exist yet) |
| `com.tradex.telegram` | alerts and `/pause` `/resume` `/flatten` `/status` | `telegram_bot_token`, `telegram_chat_id` |
| `com.tradex.gateway-proxy` | the model gateway proxy (EasyCLIProxyAPI) on localhost | the proxy binary, set in `ops.env` |
| `com.tradex.dashboard` | the read-only web dashboard, `http://127.0.0.1:8765` | `pip install -e '.[dashboard]'` |
| `com.tradex.backup` | nightly encrypted restic backup at 06:30 | external drive, `restic` |
| `com.tradex.heartbeat` | a timer, not a service: every 5 min, pings healthchecks.io if all is well | a healthchecks.io check |

They are **LaunchAgents** (they run as you, while you are logged in), not system daemons, because the
Keychain, OpenD and your files belong to your login session.

## 1. One-time prep

```bash
brew install restic python@3.11
cd ~/trade_X                                   # wherever the repo is checked out
python3.11 -m venv .venv
.venv/bin/pip install -e '.[dev]'              # includes the dashboard
.venv/bin/python -m tradex check strategies    # should print 8 "ok" lines
ops/bin/install.sh --dry-run                   # shows what would be copied
ops/bin/install.sh                             # copies the plists, makes folders and ~/.config/tradex/ops.env
```

`install.sh` fills your paths into the plists and writes them to `~/Library/LaunchAgents/`. It never runs
`launchctl`. Edit `~/.config/tradex/ops.env` for the backup drive name and the proxy path. That file holds
paths and switches only; keys and URLs with tokens live in the Keychain.

## 2. Accounts and credentials (`trade-x setup`)

Every key, token and account number goes into the macOS Keychain through hidden prompts. Nothing is echoed
or written to a file. Check what is missing at any time:

```bash
.venv/bin/python -m tradex setup --status          # prints "set" or "missing" per name, never a value
.venv/bin/python -m tradex setup --only telegram_bot_token telegram_chat_id
```

(`trade-x` is the same command as `python -m tradex` once the venv is active.)

| Name | Where it comes from |
|---|---|
| `oanda_token` | stored already. `oanda_account_id`: the practice account number in Oanda's web portal (Manage Funds, or the account selector), format `101-xxx-xxxxxxx-xxx` |
| `alpaca_key_id`, `alpaca_secret` | Alpaca dashboard, "API keys" (market data only) |
| `massive_key` | Massive (formerly Polygon) dashboard |
| `telegram_bot_token` | message `@BotFather`, `/newbot`, copy the token |
| `telegram_chat_id` | **do not look it up.** Run setup for `telegram_chat_id` *after* the token is set: it asks you to send your bot any message, then reads the chat ID from Telegram and asks you to confirm |
| `moomoo_sim_account_id` | start moomoo **OpenD**, log in, then run setup for this name: it lists only SIMULATE accounts (real accounts are filtered out before anything is shown) and you pick a number |
| `proxy_base_url`, `proxy_api_key` | `http://127.0.0.1:<port>/v1` of the gateway proxy, and the key you gave it |
| `anthropic_api_key`, `openai_api_key`, `google_api_key` | optional fallbacks, only if you want a category to bypass the proxy |

## 3. Approvals you can give now, before you sleep

The Mac needs your click only for the first use of each protected thing. Do these while you are at the
machine and awake; afterwards the unattended jobs run without a prompt. Expect these dialogs:

1. **Keychain access.** The first time a program reads a Keychain item, macOS asks
   "python wants to use the 'trade-x' keychain item". Click **Always Allow** (type your login password).
   Trigger every one now by running `tradex setup --status` once and `ops/bin/healthcheck.sh` once.
   The permission belongs to that exact Python binary: if you delete and recreate `.venv`, the prompts return.
2. **Restic and healthchecks secrets.** Add them once with `-T /usr/bin/security` so the scripts can read them:
   ```bash
   security add-generic-password -s trade-x-backup -a restic -T /usr/bin/security -w        # prompts for the password
   security add-generic-password -s trade-x-ops -a hc_ping_url -T /usr/bin/security -w      # paste the ping URL
   security add-generic-password -s trade-x-ops -a hc_backup_url -T /usr/bin/security -w
   ```
   then run `ops/bin/healthcheck.sh` and `ops/bin/backup.sh` by hand once and click Always Allow on any dialog.
3. **External drive.** First connection of the backup drive may ask for access to a "Removable Volume".
   Allow it, then run `ops/bin/backup.sh --init`, then a first `ops/bin/backup.sh`.
4. **OpenD (moomoo).** Start OpenD, log in with your moomoo credentials and any 2FA, and leave it running.
   Run `tradex opend-check` once; it is read-only. If OpenD asks to allow API access from this machine, allow it.
5. **Network prompts.** macOS may ask "Allow Python to accept incoming network connections" when the
   dashboard first starts. It listens on `127.0.0.1` only, so you can click **Deny** and it still works; Allow is also safe.
6. **Full Disk Access is not needed.** Everything lives in your home folder, the repo and the backup drive.

Things that still need you after a reboot or power cut: logging in. LaunchAgents start only after you log in, and
FileVault blocks automatic login. The dead-man ping (section 6) will tell you when this happened.

## 4. Starting things (in this order)

```bash
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.tradex.dashboard.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.tradex.telegram.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.tradex.backup.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.tradex.heartbeat.plist
```

Then, **only once they are ready**: `gateway-proxy` (after `TRADEX_PROXY_BIN` is set and the proxy listens on
127.0.0.1), `core` (after the venue adapter PRs are merged: until then `tradex run --mode paper` refuses to start
and launchd would just retry every 30 s), and `agent-worker` (after its command exists). Each wrapper exits with a
plain message (code 78) rather than guessing when something is missing. Add loaded services to
`TRADEX_REQUIRED_SERVICES` in `ops.env` so the heartbeat insists on them.

Day to day:

```bash
launchctl print gui/$(id -u)/com.tradex.core | head -20      # state and last exit code
launchctl bootout gui/$(id -u)/com.tradex.core                 # stop one
tail -f ~/Library/Logs/tradex/core.err.log                     # crashes land here; normal logs rotate in the app's own files
```

The dashboard is at <http://127.0.0.1:8765>. It is read-only, has no login, and binds to localhost on purpose. To
see it on your phone, use Tailscale (`--host` stays 127.0.0.1; use `tailscale serve`) rather than opening the port.

## 5. Keeping the Mac awake

- **Amphetamine** (already on): set it to "Indefinitely", start at login, and allow display sleep so the screen can
  turn off. Closing the lid on a MacBook sleeps it regardless of Amphetamine unless it is in clamshell mode (power + external display).
- **The core is also wrapped in `caffeinate -i -s`**, so while it runs the Mac will not idle-sleep on AC power.
- **pmset** is the system-level belt and braces. Check it (read-only) with `ops/bin/power-check.sh`. To apply what it asks for:
  ```bash
  sudo pmset -c sleep 0 disksleep 0 womp 1 autorestart 1 powernap 0
  ```
  `autorestart 1` powers the Mac back on after a power cut; it will then sit at the login window until you log in.
- Turn off automatic macOS updates that restart (System Settings, General, Software Update, "Install macOS updates" off), and update by hand on a weekend.

## 6. Dead-man switch (healthchecks.io)

1. Create a check "trade_X heartbeat": **period 5 minutes, grace 15 minutes**. Add yourself as the contact (email, plus Telegram or SMS if you want a loud one).
2. Create a second check "trade_X backup": **period 1 day, grace 6 hours**.
3. Store the two ping URLs in the Keychain (section 3, step 2).

`ops/bin/healthcheck.sh` runs every 5 minutes. It pings only if each required service has a running process, the ledger opens, and
the disk has room (default 5 GB). If a check fails it pings `/fail` with the reasons, so the alert says what is wrong. If the Mac sleeps,
dies, loses power or network, the pings simply stop and healthchecks.io alerts you. This is independent of Telegram on purpose.

## 7. Backups

**Restic, nightly, encrypted (the one that matters).** `ops/bin/backup.sh` runs at 06:30 (after the US close, before Asia opens):

- takes a consistent copy of the ledger and the research trials database with SQLite's backup API (never a raw file copy of a live WAL file) and re-verifies the ledger's hash chain on the copy;
- backs up `config/`, `strategies/`, `reports/`, `research/`, `data/calendar/`, the two database copies and `ops.env`. It skips the re-downloadable bar cache and the venv;
- keeps 14 daily, 8 weekly and 12 monthly snapshots, then spot-checks 2% of the stored data;
- refuses to run if the drive is not mounted (so it can never fill the Mac's own disk), and pings healthchecks.io `/fail` if anything goes wrong.

Set up once: format the drive APFS and name it `TradexBackup` (or change `TRADEX_BACKUP_VOLUME` in `ops.env`), add the restic password to the Keychain (section 3), then
`ops/bin/backup.sh --init`. **Write the restic password down somewhere that is not this Mac** (a password manager, plus a paper copy). Lose it and the backup cannot be opened.

Restore drill, once a month (restores into a scratch folder, changes nothing live):

```bash
export RESTIC_REPOSITORY=/Volumes/TradexBackup/tradex-restic
export RESTIC_PASSWORD_COMMAND="security find-generic-password -s trade-x-backup -a restic -w"
restic snapshots
restic restore latest --target ~/restore-test
.venv/bin/python -m tradex verify-ledger --ledger "$(find ~/restore-test -name ledger.sqlite | head -1)"
```

**Time Machine (the second net, for everything else).** Use a *different* drive from the restic one.

- Turn it on in System Settings, General, Time Machine, and encrypt the backup.
- Exclude things that are huge or not safely copyable while running:
  ```bash
  tmutil addexclusion -p ~/trade_X/.venv ~/trade_X/data/cache ~/trade_X/data/ledger
  ```
  The live ledger is excluded on purpose: Time Machine copies a file mid-write, which can tear a database. Time Machine still
  holds the consistent nightly copy in `data/state/backup-staging/`.
- Time Machine is not a substitute for the restic job: it is hourly, can be silently skipped when the drive is asleep, and cannot be spot-checked by the dashboard.

The dashboard's Accounts and system tab shows the age of the last good backup and the last good heartbeat, read from `data/state/backup_ok` and
`data/state/healthcheck_ok`, and turns red if they go stale.

## 8. What was deliberately left out

- No script loads a service, starts an order, or changes a setting on the Mac. `power-check.sh` only reads.
- No live mode: `run-service.sh` refuses any `TRADEX_MODE` other than `paper`.
- No secrets in plists, `ops.env` or the repo.
- The `agent-worker` and the gateway proxy are wired but cannot start until their commands exist.
