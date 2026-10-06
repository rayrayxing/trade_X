"""Phase-1 research universes and the history download entry point.

First list (44 symbols): index ETFs, the nine original sector SPDRs plus TLT and GLD,
every seed's named stocks, and same-industry large caps for pairs. Second list (the
candidate pool for the point-in-time liquidity screen, tradex.research.screen): the
S&P 100 as of October 2026 plus seven liquid ETFs, 80 more symbols of the 300/30-day
OpenD quota. Both are today's names: no delisted stock is available from OpenD, so
anything ranked or screened on them still carries survivorship bias.

    python -m tradex.research.universe opend      # first list, daily + 60-minute bars
    python -m tradex.research.universe pool       # candidate pool, daily + 60-minute bars
    python -m tradex.research.universe etfs3      # round-3 equity ETFs, daily bars
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

# S&P 100 members on 4 Oct 2026 (GOOG left out: same company as GOOGL). Checked against
# OpenD's S&P 500 plate (US..SPX) that day; the index's own history of members is not
# available, which is the survivorship bias the gate report states.
SP100 = ["AAPL", "ABBV", "ABT", "ACN", "ADBE", "AIG", "AMD", "AMGN", "AMT", "AMZN", "AVGO", "AXP", "BA", "BAC",
         "BKNG", "BLK", "BMY", "BRK.B", "C", "CAT", "CHTR", "CL", "CMCSA", "COF", "COP", "COST", "CRM",
         "CSCO", "CVS", "CVX", "DE", "DHR", "DIS", "DUK", "EMR", "F", "FDX", "GD", "GE", "GILD", "GM", "GOOGL",
         "GS", "HD", "HON", "IBM", "INTC", "INTU", "ISRG", "JNJ", "JPM", "KO", "LIN", "LLY", "LMT", "LOW", "MA",
         "BNY", "MCD", "MDLZ", "MDT", "MET", "META", "MMM", "MO", "MRK", "MS", "MSFT", "NEE", "NFLX", "NKE", "NOW",
         "NVDA", "ORCL", "PEP", "PFE", "PG", "PLTR", "PM", "PYPL", "QCOM", "RTX", "SBUX", "SCHW", "SO", "SPG",
         "T", "TGT", "TMO", "TMUS", "TSLA", "TXN", "UBER", "UNH", "UNP", "UPS", "USB", "V", "VZ", "WFC", "WMT",
         "XOM"]
EXTRA_ETFS = ["EFA", "EEM", "IEF", "HYG", "LQD", "SLV", "VNQ"]
ETFS = INDEX_ETFS + SECTOR_ETFS + MACRO_ETFS + EXTRA_ETFS
STOCKS = list(dict.fromkeys(SP100 + SEED_STOCKS + PAIR_STOCKS))
# Research round 3: liquid US-listed equity ETFs with daily history from about 2006, for the
# panic-rebound variants that need more symbols per event (daily bars only, 12 symbols of quota).
# A separate wider pool, so the 22-ETF pool every other strategy screens stays as it was.
ROUND3_ETFS = ["SMH", "XBI", "KRE", "ITB", "XHB", "XOP", "XRT", "IBB", "EWJ", "EWZ", "FXI", "VWO"]
ETFS_WIDE = ETFS + ROUND3_ETFS
POOL = [s for s in SP100 + EXTRA_ETFS if s not in US_DAILY]

FX_PAIRS = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD", "USD_CHF", "NZD_USD", "EUR_JPY", "GBP_JPY"]
FX_TFS = ["H1", "H4", "D1"]


def download_opend(max_new_symbols: int = 60, lists=None) -> None:
    from tradex.data.opend import OpenDError, OpenDFetcher, QuotaExceeded
    with OpenDFetcher(max_new_symbols=max_new_symbols) as f:
        used, remaining, _ = f.quota()
        print(f"OpenD history quota: {used} used, {remaining} remaining")
        for tf, syms in lists or (("D1", US_DAILY), ("H1", US_HOURLY)):
            for s in syms:
                try:
                    b = f.fetch(s, tf)
                except (OpenDError, QuotaExceeded) as exc:    # one bad code must not stop the batch
                    print(f"{s} {tf}: failed: {exc}", flush=True)
                    continue
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


def download_pool(max_new_symbols: int = 80) -> None:
    download_opend(max_new_symbols, (("D1", POOL), ("H1", POOL)))


def download_round3(max_new_symbols: int = 12) -> None:
    download_opend(max_new_symbols, (("D1", ROUND3_ETFS),))


if __name__ == "__main__":
    {"opend": download_opend, "pool": download_pool, "etfs3": download_round3, "oanda": download_oanda}[sys.argv[1]]()
