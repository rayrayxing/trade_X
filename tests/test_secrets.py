import io
import json
import logging

import keyring
import pytest
from keyring.backend import KeyringBackend

from tradex import secrets
from tradex.ops.log import REDACTED, Scrubber, setup_logging
from tradex.setup_cmd import run_setup, simulate_accounts, telegram_chat_id_from_updates


class FakeKeyring(KeyringBackend):
    priority = 1

    def __init__(self):
        self.d = {}

    def get_password(self, service, username):
        return self.d.get((service, username))

    def set_password(self, service, username, password):
        self.d[(service, username)] = password

    def delete_password(self, service, username):
        self.d.pop((service, username), None)


@pytest.fixture(autouse=True)
def fake_keyring(monkeypatch):
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("TRADEX_ALLOW_ENV_SECRETS", raising=False)
    prev = keyring.get_keyring()
    fk = FakeKeyring()
    keyring.set_keyring(fk)
    yield fk
    keyring.set_keyring(prev)


def test_roundtrip_and_missing():
    assert not secrets.has("oanda_token")
    with pytest.raises(secrets.MissingSecret):
        secrets.get("oanda_token")
    secrets.set("oanda_token", "abc123456")
    assert secrets.get("oanda_token") == "abc123456" and secrets.has("oanda_token")
    with pytest.raises(ValueError):
        secrets.get("nope")


def test_env_only_in_ci(monkeypatch):
    monkeypatch.setenv("TRADEX_OANDA_TOKEN", "from-env-value")
    assert not secrets.has("oanda_token")          # ignored outside CI
    monkeypatch.setenv("CI", "true")
    assert not secrets.has("oanda_token")          # a bare CI variable is not enough
    monkeypatch.setenv("TRADEX_ALLOW_ENV_SECRETS", "1")
    assert secrets.get("oanda_token") == "from-env-value"
    secrets.status()  # does not raise


def test_setup_hides_and_stores(fake_keyring):
    out = []
    answers = iter([""] * 50)
    run_setup(["oanda_token", "proxy_base_url"], ask=lambda p: next(answers), secret=lambda p: "tok-SECRET-9999",
              out=out.append)
    assert secrets.get("oanda_token") == "tok-SECRET-9999"
    text = "\n".join(out)
    assert "SECRET" not in text and "oanda_token" in text and "set" in text


def test_setup_status_and_unknown():
    out = []
    assert run_setup(status_only=True, out=out.append) == 0
    assert all(line.endswith("missing") for line in out)
    assert run_setup(["bogus"], out=out.append) == 2


def test_setup_telegram_chat_id_and_sim_account():
    secrets.set("telegram_bot_token", "123456789:AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
    out, ans = [], iter(["", "y"])
    run_setup(["telegram_chat_id"], ask=lambda p: next(ans), secret=lambda p: "", out=out.append,
              chat_id_lookup=lambda tok: "555111222")
    assert secrets.get("telegram_chat_id") == "555111222"
    assert "555111222" not in "\n".join(out)

    rows = [{"acc_id": 1, "trd_env": "REAL"}, {"acc_id": 2, "trd_env": "SIMULATE", "acc_type": "CASH"}]
    assert [r["acc_id"] for r in simulate_accounts(lambda: rows)] == [2]
    out, ans = [], iter(["1"])
    run_setup(["moomoo_sim_account_id"], ask=lambda p: next(ans), secret=lambda p: "", out=out.append,
              accounts=lambda: rows)
    assert secrets.get("moomoo_sim_account_id") == "2"


def test_telegram_updates_parse():
    data = {"result": [{"message": {"chat": {"id": 42}}}, {"message": {"chat": {"id": 77}}}]}
    assert telegram_chat_id_from_updates("t", fetch=lambda url: data) == "77"
    assert telegram_chat_id_from_updates("t", fetch=lambda url: {"result": []}) is None


def test_logs_scrub_planted_secrets(tmp_path):
    planted = "zZ-planted-store-value-123"
    secrets.set("proxy_api_key", planted)
    oanda_like = "0123456789abcdef0123456789abcdef-fedcba9876543210fedcba9876543210"
    tg_like = "987654321:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi_-"
    path = tmp_path / "t.jsonl"
    setup_logging(path)
    log = logging.getLogger("tradex.test")
    log.info("key=%s and header Authorization: Bearer %s", planted, oanda_like)
    log.warning("telegram %s", tg_like, extra={"api_key": "plainvalue12345", "note": planted})
    try:
        raise RuntimeError(f"boom {planted}")
    except RuntimeError:
        log.exception("failed")
    for h in logging.getLogger("tradex").handlers:
        h.flush()
    raw = path.read_text()
    for s in (planted, oanda_like, tg_like, "plainvalue12345"):
        assert s not in raw
    assert REDACTED in raw
    assert all(json.loads(line)["level"] for line in raw.splitlines())
    logging.getLogger("tradex").handlers.clear()


def test_rotation_cap_enforced(tmp_path):
    with pytest.raises(ValueError):
        setup_logging(tmp_path / "x.jsonl", max_bytes=100 * 1024**2, backups=20)
    assert Scrubber(["short"]).scrub("short") == "short"   # below min length: not scrubbed


def test_providers_read_keychain_not_env(monkeypatch):
    from tradex.data import providers
    seen = {}

    def fake_get(url, headers=None, retries=4):
        seen["h"] = headers
        return {"candles": []}

    monkeypatch.setattr(providers, "_get_json", fake_get)
    monkeypatch.setenv("OANDA_API_TOKEN", "env-token-should-be-ignored")
    with pytest.raises(secrets.MissingSecret):
        providers.OandaProvider().get_bars("EUR_USD", "H1", "2024-01-01", "2024-01-02")
    secrets.set("oanda_token", "kc-token-value")
    providers.OandaProvider().get_bars("EUR_USD", "H1", "2024-01-01", "2024-01-02")
    assert seen["h"]["Authorization"] == "Bearer kc-token-value"
    assert providers.OandaProvider.base == "https://api-fxpractice.oanda.com"
