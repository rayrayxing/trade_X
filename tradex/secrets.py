"""Secret store: macOS Keychain via keyring (service "trade-x").

Env vars ``TRADEX_<NAME>`` are honoured only when both ``CI`` and ``TRADEX_ALLOW_ENV_SECRETS`` are set, so a stray
variable on Ray's Mac can never silently override the Keychain. Values are never logged
or returned by anything except ``get``; ``status`` reports only set/missing.
"""
from __future__ import annotations

import os

SERVICE = "trade-x"

# name -> one-line description shown by `trade-x setup` (never contains a value)
NAMES: dict[str, str] = {
    "oanda_token": "Oanda practice API token",
    "oanda_account_id": "Oanda practice account ID",
    "alpaca_key_id": "Alpaca market-data key ID",
    "alpaca_secret": "Alpaca market-data secret",
    "massive_key": "Massive (Polygon) API key",
    "telegram_bot_token": "Telegram bot token",
    "telegram_chat_id": "Telegram chat ID for alerts",
    "proxy_base_url": "LLM proxy base URL",
    "proxy_api_key": "LLM proxy API key",
    "anthropic_api_key": "Anthropic API key",
    "openai_api_key": "OpenAI API key",
    "google_api_key": "Google API key",
    "deepseek_api_key": "DeepSeek API key",
    "moomoo_sim_account_id": "moomoo SIMULATE account ID",
}


class MissingSecret(KeyError):
    """A required secret is not set. The message names the secret, never a value."""


def _check(name: str) -> str:
    if name not in NAMES:
        raise ValueError(f"unknown secret name {name!r}; known: {sorted(NAMES)}")
    return name


def _ci() -> bool:
    return bool(os.environ.get("CI")) and bool(os.environ.get("TRADEX_ALLOW_ENV_SECRETS"))


def env_name(name: str) -> str:
    return f"TRADEX_{name.upper()}"


def _keyring():
    import keyring  # lazy: keeps import of tradex cheap and CI-safe
    return keyring


def get(name: str) -> str:
    _check(name)
    if _ci():
        val = os.environ.get(env_name(name))
        if not val:
            raise MissingSecret(f"{name} (env {env_name(name)} not set in CI)")
        return val
    try:
        val = _keyring().get_password(SERVICE, name)
    except Exception as exc:  # keyring backend errors must not leak into callers' logs
        raise MissingSecret(f"{name} (keychain unavailable: {type(exc).__name__})") from None
    if not val:
        raise MissingSecret(f"{name} (run `trade-x setup --only {name}`)")
    return val


def set(name: str, value: str) -> None:  # noqa: A001 - interface name fixed by the shared contract
    _check(name)
    if not value:
        raise ValueError("refusing to store an empty secret")
    if _ci():
        raise RuntimeError("secrets are read-only from the environment in CI")
    _keyring().set_password(SERVICE, name, value)


def delete(name: str) -> None:
    _check(name)
    try:
        _keyring().delete_password(SERVICE, name)
    except Exception:
        pass


def has(name: str) -> bool:
    try:
        get(name)
        return True
    except MissingSecret:
        return False


def status() -> dict[str, bool]:
    return {n: has(n) for n in NAMES}


def known_values() -> list[str]:
    """Every stored secret value, for the log scrubber only."""
    out = []
    for n in NAMES:
        try:
            out.append(get(n))
        except MissingSecret:
            pass
    return out
