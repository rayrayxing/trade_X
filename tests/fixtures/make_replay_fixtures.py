"""Regenerate tests/fixtures/replay/*.csv (run once; the CSVs are what the tests read).

The files are FROZEN deterministic H1 bars, not market recordings: they exist so the
full-stack replay smoke test keeps replaying the same bars even if ``synthetic_bars`` or its
seeds change later. Replace them with real Oanda practice candles recorded on Ray's Mac
(``tradex fetch``) when available; the test only needs the columns open/high/low/close/volume
on an hourly UTC index.
"""
from pathlib import Path

from tradex.data.synthetic import synthetic_bars

OUT = Path(__file__).parent / "replay"
KW = dict(tf="H1", vol=0.002, trend_strength=0.0004, regime_len=200, start="2025-01-06", business_days=False)

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for sym, seed, px in (("EUR_USD", 5, 1.10), ("USD_JPY", 6, 150.0)):
        df = synthetic_bars(1_100, seed=seed, price=px, **KW).round(6 if px < 10 else 4)
        df["volume"] = df["volume"].round(0)
        df.to_csv(OUT / f"{sym}_H1.csv")
