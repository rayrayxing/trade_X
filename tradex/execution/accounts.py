"""Agent-owned venue accounts, from config/accounts.yaml (protected).

The file names each account's venue and mode but never its ID: the ID lives in the
Keychain (``trade-x setup`` writes it) and is resolved here through ``tradex.secrets``.
Phase 1 trades paper accounts only, so any other mode is refused at load time, and so is
a venue environment other than its paper one (Oanda ``practice``, moomoo ``SIMULATE``).

An account may instead say ``auto_select: true``: when its secret is not set, the venue
adapter picks the ID itself (moomoo: the single SIMULATE account with US trading rights)
through a resolver passed to ``load_accounts``. The ID still never touches the file.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import yaml

DEFAULT_ACCOUNTS = Path(__file__).resolve().parents[2] / "config" / "accounts.yaml"
ALLOWED_MODES = {"paper"}
PAPER_ENVIRONMENTS = {"oanda": "practice", "moomoo": "SIMULATE"}   # venue -> its only allowed environment


@dataclass
class AgentAccount:
    name: str
    venue: str
    mode: str
    asset_classes: list[str]
    account_id: str = ""               # empty until the Keychain has the ID: such an account is not usable
    secret: str = ""
    environment: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def resolved(self) -> bool:
        return bool(self.account_id)

    @property
    def auto_select(self) -> bool:
        return bool(self.extra.get("auto_select"))


def _secret_from_keychain(name: str) -> str | None:
    try:
        from tradex import secrets  # lane B; imported lazily so tests and CI never need a keyring
    except ImportError:
        return None
    try:
        return secrets.get(name)
    except Exception:  # noqa: BLE001 - MissingSecret or no keyring backend: the account stays unresolved
        return None


def load_accounts(path: str | Path | None = None,
                  resolve: Callable[[str], str | None] | None = None,
                  auto: dict[str, Callable[["AgentAccount"], str | None]] | None = None) -> list[AgentAccount]:
    """Accounts from ``path``; IDs left empty in the file come from ``resolve(secret)``, then,
    for ``auto_select`` accounts still unresolved, from ``auto[venue](account)``."""
    doc = yaml.safe_load(Path(path or DEFAULT_ACCOUNTS).read_text()) or {}
    resolve = resolve or _secret_from_keychain
    known = {"name", "venue", "mode", "asset_classes", "account_id", "secret", "environment"}
    out = []
    for d in doc.get("accounts", []):
        if d.get("mode") not in ALLOWED_MODES:
            raise ValueError(f"account {d.get('name')!r}: mode {d.get('mode')!r} is not allowed; "
                             f"phase 1 trades {sorted(ALLOWED_MODES)} accounts only")
        env = str(d.get("environment", ""))
        want = PAPER_ENVIRONMENTS.get(d.get("venue"))
        if want is not None and env != want:
            raise ValueError(f"account {d.get('name')!r}: environment {env!r} is not allowed; "
                             f"{d.get('venue')} trades {want!r} only")
        acc = AgentAccount(name=d["name"], venue=d["venue"], mode=d["mode"],
                           asset_classes=list(d.get("asset_classes", [])),
                           account_id=str(d.get("account_id") or ""), secret=d.get("secret", ""),
                           environment=str(d.get("environment", "")),
                           extra={k: v for k, v in d.items() if k not in known})
        if not acc.account_id and acc.secret:
            acc.account_id = str(resolve(acc.secret) or "")
        if not acc.account_id and acc.auto_select and auto and acc.venue in auto:
            acc.account_id = str(auto[acc.venue](acc) or "")
        out.append(acc)
    return out


def agent_account_ids(accounts: list[AgentAccount]) -> set[str]:
    return {a.account_id for a in accounts if a.resolved}
