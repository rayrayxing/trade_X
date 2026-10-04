"""Phase-1 research universes and the history download entry point.

US list (44 symbols, inside the 60-symbol phase-1 budget of the 300/30-day OpenD quota):
index ETFs, the nine original sector SPDRs plus TLT and GLD, every seed's named stocks,
and same-industry large caps for pairs and cross-sectional tests. Today's large caps are
survivors, so cross-sectional results on this list carry survivorship bias.

    python -m tradex.research.universe opend      # US daily + 60-minute bars via OpenD
    python -m tradex.research.universe oanda      # FX bid/ask H1/H4/D via Oanda practice
"""
from __future__ import annotations

import sys

INDEX_ETFS = ["SPY", "QQQ", "IWM", "DIA"]
SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB"]
MACRO_ETFS = ["TLT", "GLD"]
SEED_STOCKS = ["NVDA", "AMD", "TSLA", "META", "AMZN", "AAPL", "MSFT"]
PAIRS = [("KO", "PEP"), ("XOM", "CVX"), ("V", "MA"), ("JPM", "BAC"), ("GS", "MS"), ("HD", "LOW"),
         ("MRK", "PFE"), ("WMT", "COST"), ("INTC", "CSCO"), ("GOOGL", "ORCL"), ("UNH", "JNJ")]
PAIR_STOCKS = [s for p in PAIRS for s in p]

US_DAILY = list(dict.fromkeys(INDEX_ETFS + SECTOR_ETFS + MACRO_ETFS + SEED_STOCKS + PAIR_STOCKS))
US_HOURLY = INDEX_ETFS + SECTOR_ETFS + MACRO_ETFS + SEED_STOCKS
LARGE_CAPS = SEED_STOCKS + PAIR_STOCKS

FX_PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD", "USD_CHF", "NZD_USD", "EUR_JPY", "GBP_JPY"]
FX_TFS = ["H1", "H4", "D1"]


def download_opend(max_new_symbols: int = 60) -> None:
    from tradex.data.opend import OpenDFetcher
    with OpenDFetcher(max_new_symbols=max_new_symbols) as f:
        used, remaining, _ = f.quota()
        print(f"OpenD history quota: {used} used, {remaining} remaining")
        for tf, syms in (("D1", US_DAILY), ("H1", US_HOURLY)):
            for s in syms:
                b = f.fetch(s, tf)
                print(f"{s} {tf}: {len(b)} bars {b.index[0].date() if len(b) else '-'}..{b.index[-1].date() if len(b) else '-'}",
                      flush=True)
        used, remaining, _ = f.quota()
        print(f"done; quota now {used} used, {remaining} remaining; new this run: {len(f.new_symbols)}")


def download_oanda(years: int = 10) -> None:
    from tradex.data.oanda_history import OandaHistory
    h = OandaHistory()
    for pair in FX_PAIRS:
        for tf in FX_TFS:
            b = h.fetch(pair, tf, years=years)
            print(f"{pair} {tf}: {len(b)} bars", flush=True)


if __name__ == "__main__":
    {"opend": download_opend, "oanda": download_oanda}[sys.argv[1]]()
