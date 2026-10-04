"""`trade-x setup`: Ray types each secret into a hidden prompt; it goes straight to the Keychain.

Nothing typed here is echoed, printed, logged or written to a file. The only output per
secret is its name and "set" or "missing". Prompt functions are injectable for tests.
"""
from __future__ import annotations

import getpass
import json
import sys
import urllib.request
from typing import Callable

from tradex import secrets

OPEND = ("127.0.0.1", 11111)


def print_status(names: list[str] | None = None, out=print) -> bool:
    st = secrets.status()
    names = names or list(secrets.NAMES)
    for n in names:
        out(f"{n:24s} {'set' if st[n] else 'missing'}")
    return all(st[n] for n in names)


def telegram_chat_id_from_updates(token: str, fetch: Callable[[str], dict] | None = None) -> str | None:
    """Chat ID of the newest message sent to the bot, or None. The token is only used in the request URL."""
    def _fetch(url: str) -> dict:
        with urllib.request.urlopen(url, timeout=15) as r:
            return json.loads(r.read().decode())
    data = (fetch or _fetch)(f"https://api.telegram.org/bot{token}/getUpdates")
    for upd in reversed(data.get("result", [])):
        msg = upd.get("message") or upd.get("channel_post") or upd.get("my_chat_member") or {}
        chat = msg.get("chat") or {}
        if chat.get("id") is not None:
            return str(chat["id"])
    return None


def simulate_accounts(list_accounts: Callable[[], list[dict]] | None = None) -> list[dict]:
    """SIMULATE accounts OpenD reports; REAL accounts are filtered out before anything is shown."""
    rows = (list_accounts or _opend_accounts)()
    return [r for r in rows if str(r.get("trd_env", "")).upper() == "SIMULATE"]


def _opend_accounts() -> list[dict]:
    from moomoo import OpenSecTradeContext, RET_OK, SecurityFirm, TrdMarket
    ctx = OpenSecTradeContext(filter_trdmarket=TrdMarket.NONE, host=OPEND[0], port=OPEND[1],
                              security_firm=SecurityFirm.FUTUSG)
    try:
        ret, df = ctx.get_acc_list()
        if ret != RET_OK:
            raise RuntimeError("OpenD refused get_acc_list")
        return df.to_dict("records")
    finally:
        ctx.close()


def run_setup(only: list[str] | None = None, status_only: bool = False, *, ask: Callable[[str], str] = input,
              secret: Callable[[str], str] = getpass.getpass, out=print,
              chat_id_lookup: Callable[[str], str | None] = telegram_chat_id_from_updates,
              accounts: Callable[[], list[dict]] | None = None) -> int:
    names = only or list(secrets.NAMES)
    bad = [n for n in names if n not in secrets.NAMES]
    if bad:
        out(f"unknown name(s): {', '.join(bad)}; known: {', '.join(secrets.NAMES)}")
        return 2
    if status_only:
        print_status(names, out)
        return 0
    out("Each secret is typed into a hidden prompt and stored in the macOS Keychain. Press Enter to skip one.")
    for n in names:
        have = secrets.has(n)
        out(f"\n{n} ({secrets.NAMES[n]}): {'set' if have else 'missing'}")
        if have and ask("  replace it? [y/N] ").strip().lower() != "y":
            continue
        if n == "telegram_chat_id" and secrets.has("telegram_bot_token"):
            if ask("  message your bot in Telegram now, then press Enter to read the chat ID from it (or type n): ").strip().lower() != "n":
                try:
                    cid = chat_id_lookup(secrets.get("telegram_bot_token"))
                except Exception as exc:  # never include the URL: it contains the bot token
                    cid = None
                    out(f"  lookup failed ({type(exc).__name__})")
                if cid and ask("  found a chat; use it? [Y/n] ").strip().lower() != "n":
                    secrets.set(n, cid)
                    out(f"  {n}: set")
                    continue
                if not cid:
                    out("  no messages found")
        if n == "moomoo_sim_account_id":
            try:
                accs = simulate_accounts(accounts)
            except Exception as exc:
                accs = []
                out(f"  could not list accounts from OpenD ({type(exc).__name__}); is it running and logged in?")
            for i, a in enumerate(accs, 1):
                out(f"  [{i}] SIMULATE account {a.get('acc_id')}  type={a.get('acc_type', '?')}  markets={a.get('trdmarket_auth', '?')}")
            if accs:
                pick = ask("  pick a number (Enter to skip): ").strip()
                if pick.isdigit() and 1 <= int(pick) <= len(accs):
                    secrets.set(n, str(accs[int(pick) - 1]["acc_id"]))
                    out(f"  {n}: set")
                    continue
        val = secret(f"  {n}: ")
        if val.strip():
            secrets.set(n, val.strip())
        out(f"  {n}: {'set' if secrets.has(n) else 'missing'}")
    out("")
    print_status(names, out)
    return 0
