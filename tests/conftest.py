import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from tradex.data.synthetic import synthetic_bars
from tradex.strategy.spec import StrategySpec

os.environ.setdefault("TRADEX_LOG_PATH", os.path.join(tempfile.mkdtemp(prefix="tradex-test-logs-"), "tradex.jsonl"))


@pytest.fixture
def stock_data():
    return {s: synthetic_bars(1500, seed=i) for i, s in enumerate(["NVDA", "AMD", "AAPL"])}


@pytest.fixture
def fx_data():
    pairs = [("EUR_USD", 1.10), ("USD_JPY", 150.0), ("GBP_USD", 1.30)]
    return {s: synthetic_bars(4000, tf="H1", seed=20 + i, price=p, vol=0.002, trend_strength=0.0003,
                              regime_len=400, start="2024-01-01") for i, (s, p) in enumerate(pairs)}


def flat_bars(n=40, price=100.0, rng=1.0, start="2024-01-02", freq="B", tz="UTC"):
    idx = pd.date_range(start, periods=n, freq=freq, tz=tz).astype("datetime64[ns, UTC]")
    return pd.DataFrame({"open": price, "high": price + rng, "low": price - rng, "close": price,
                         "volume": 1e6}, index=idx).astype(float)


def simple_spec(asset_class="stocks", long="close > 0", short=None, tf="D1", **exit_kw):
    ex = {"stop_atr": 1.5, "target_r": 2.0, "max_bars": 10} | exit_kw
    d = {"id": "t", "version": 1, "asset_class": asset_class, "universe": ["X"],
         "timeframes": {"signal": tf}, "features": {}, "entry": {"long": long}, "exit": ex,
         "holding": {"expected_hours": 24, "crosses_rollover": True}}
    if short:
        d["entry"]["short"] = short
    return StrategySpec.from_dict(d)


@pytest.fixture(autouse=True)
def _isolated_trial_ledger(tmp_path, monkeypatch):
    """Walk-forward appends to the global trial ledger; keep tests out of data/research/."""
    monkeypatch.setenv("TRADEX_TRIALS_DB", str(tmp_path / "trials.sqlite"))
