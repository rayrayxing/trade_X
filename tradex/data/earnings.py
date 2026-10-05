"""Historical US earnings dates for backtests: moomoo OpenD first, SEC EDGAR before that.

Two free sources, no paid key:

- OpenD ``get_financials_earnings_price_move`` (quote context only): the release date and
  whether it came before the open or after the close, for the last 50 reports (about 12
  years). This is the real release date.
- SEC EDGAR submissions JSON (data.sec.gov): filing dates of 8-K reports with item 2.02
  ("Results of Operations and Financial Condition"), the form companies file the day they
  release earnings. Used only where OpenD has no report within a month, and labelled
  ``sec_8k_2.02`` because it is a filing-date proxy: timing unknown, and the filing can
  lag the release by a day. In a year with no such 8-K (Berkshire reports in its 10-Q)
  the 10-Q/10-K filing dates stand in, labelled ``sec_10q_10k`` (a weaker proxy: those
  can come weeks after the release). Companies whose filer changed (Google -> Alphabet,
  Medtronic Inc -> plc, ...) are read under each CIK in ``FORMER_CIKS``.

Per symbol the merged table (``date``, ``timing``, ``source``) is cached to CSV
(data/cache/earnings/, gitignored). ETFs have no earnings and are not fetched.

``Earnings.event_time`` turns a row into the instant a position must be flat by, chosen
for an engine that fills exits at the next bar's open: a release before or during the
session, or with unknown timing, counts from 00:00 New York of its date; one after the
close counts from 16:00.

    python -m tradex.data.earnings AAPL MSFT ...     # fetch and cache
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable

import pandas as pd

NY = "America/New_York"
DEFAULT_CACHE = Path(__file__).resolve().parents[2] / "data" / "cache" / "earnings"
SEC_USER_AGENT = "trade-x research ruix.zheng@gmail.com"
SEC_HOSTS = ("data.sec.gov", "www.sec.gov")
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
COLS = ["date", "timing", "source"]
OPEND_TIMING = {"BEFORE_MARKET": "before", "BEFORE": "before", "AFTER_MARKET": "after", "AFTER": "after",
                "REGULAR": "during", "DURING_MARKET": "during", "IN_MARKET": "during"}
# Earlier SEC filers of today's companies, checked against each CIK's registered name on
# 4 Oct 2026 (GOOGLE INC., TWDC Enterprises 18 Corp., Avago Technologies LTD, Broadcom Pte.
# Ltd., LINDE INC, MEDTRONIC INC).
FORMER_CIKS = {"GOOGL": [1288776], "DIS": [1001039], "AVGO": [1441634, 1649338], "LIN": [884905], "MDT": [64670]}
SEC_GAP_DAYS = 30        # an SEC row this close to an OpenD report is the same report


def _sec_get(url: str) -> dict:
    """SEC fair-access rules: a User-Agent naming a contact, at most 10 requests a second."""
    if urllib.parse.urlparse(url).hostname not in SEC_HOSTS:
        raise ValueError(f"not an SEC host: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "identity"})
    delay = 1.0
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise
            if attempt == 3:
                raise
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def empty() -> pd.DataFrame:
    return pd.DataFrame(columns=COLS)


def parse_opend(df: pd.DataFrame | None) -> pd.DataFrame:
    """Rows of get_financials_earnings_price_move (one per day around each report) -> one row per report."""
    if df is None or not len(df):
        return empty()
    rep = df.drop_duplicates("pub_trading_day_str")
    out = pd.DataFrame({"date": pd.to_datetime(rep["pub_trading_day_str"]).dt.strftime("%Y-%m-%d"),
                        "timing": [OPEND_TIMING.get(str(t).upper(), "unknown") for t in rep["pub_type"]],
                        "source": "opend"})
    return out.sort_values("date").reset_index(drop=True)


def parse_sec(docs: list[dict]) -> pd.DataFrame:
    """SEC submissions documents (the main ``filings.recent`` block and any older ``files``
    pages) -> filing dates of 8-K reports with item 2.02, and 10-Q/10-K filing dates for
    the years that have no such 8-K."""
    k8, q = set(), set()
    for d in docs:
        block = d.get("filings", {}).get("recent", d)
        items = block.get("items") or [""] * len(block.get("form", []))
        for form, fd, it in zip(block.get("form", []), block.get("filingDate", []), items):
            if form == "8-K" and "2.02" in str(it).split(","):
                k8.add(fd)
            elif form in ("10-Q", "10-K"):
                q.add(fd)
    years = {d[:4] for d in k8}
    rows = [(d, "sec_8k_2.02") for d in k8] + [(d, "sec_10q_10k") for d in q if d[:4] not in years]
    rows.sort()
    return pd.DataFrame({"date": [r[0] for r in rows], "timing": "unknown", "source": [r[1] for r in rows]})


def merge(opend: pd.DataFrame, sec: pd.DataFrame) -> pd.DataFrame:
    """OpenD rows, plus SEC rows that have no OpenD report within ``SEC_GAP_DAYS`` (the
    years before OpenD's first report, and quarters OpenD skips)."""
    if len(opend) and len(sec):
        od = pd.to_datetime(opend["date"]).sort_values().to_numpy()
        sd = pd.to_datetime(sec["date"]).to_numpy()
        pos = od.searchsorted(sd)
        gap = pd.Timedelta(days=SEC_GAP_DAYS).to_timedelta64()
        near = [(i < len(od) and od[i] - d <= gap) or (i > 0 and d - od[i - 1] <= gap) for i, d in zip(pos, sd)]
        sec = sec[[not n for n in near]]
    out = pd.concat([x for x in (sec, opend) if len(x)]) if len(opend) or len(sec) else empty()
    return out.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)[COLS]


def event_time(date: str, timing: str) -> pd.Timestamp:
    """The instant (UTC) a position must be flat by for this report; see the module docstring."""
    t = pd.Timestamp(date).tz_localize(NY)
    return (t + pd.Timedelta(hours=16) if timing == "after" else t).tz_convert("UTC")


class Earnings:
    """Fetch, cache and read earnings tables. ``opend_ctx`` and ``sec_http`` are injectable for tests."""

    def __init__(self, cache_dir: str | Path = DEFAULT_CACHE, opend_ctx=None,
                 sec_http: Callable[[str], dict] = _sec_get, pause_s: float = 0.15):
        self.cache_dir = Path(cache_dir)
        self._ctx = opend_ctx
        self._own_ctx = opend_ctx is None
        self.sec_http, self.pause_s = sec_http, pause_s
        self._ciks: dict[str, int] | None = None

    @property
    def ctx(self):
        if self._ctx is None:
            from moomoo import OpenQuoteContext  # quote context only, never a trade context
            from tradex.data.opend import HOST, PORT
            self._ctx = OpenQuoteContext(host=HOST, port=PORT)
        return self._ctx

    def close(self) -> None:
        if self._own_ctx and self._ctx is not None:
            self._ctx.close()
            self._ctx = None

    def path(self, symbol: str) -> Path:
        return self.cache_dir / f"{symbol}.csv"

    def load(self, symbol: str) -> pd.DataFrame | None:
        p = self.path(symbol)
        return pd.read_csv(p, dtype=str).fillna("") if p.exists() else None

    def cik(self, symbol: str) -> int | None:
        if self._ciks is None:
            doc = self.sec_http(TICKERS_URL)
            self._ciks = {str(v["ticker"]).upper(): int(v["cik_str"]) for v in doc.values()}
        return self._ciks.get(symbol.replace(".", "-").upper())

    def from_opend(self, symbol: str) -> pd.DataFrame:
        from tradex.data.opend import us_code
        ret, df = self.ctx.get_financials_earnings_price_move(us_code(symbol), period_count=50)
        if ret != 0:
            raise RuntimeError(f"{symbol}: get_financials_earnings_price_move failed: {df}")
        return parse_opend(df)

    def from_sec(self, symbol: str, since: str = "2005-01-01") -> pd.DataFrame:
        cik = self.cik(symbol)
        if cik is None:
            return empty()
        docs = []
        for c in [cik] + FORMER_CIKS.get(symbol, []):
            main = self.sec_http(f"https://data.sec.gov/submissions/CIK{c:010d}.json")
            docs.append(main)
            for f in main.get("filings", {}).get("files", []):
                if f.get("filingTo", "9999") >= since:
                    time.sleep(self.pause_s)
                    docs.append(self.sec_http(f"https://data.sec.gov/submissions/{f['name']}"))
            time.sleep(self.pause_s)
        out = parse_sec(docs)
        return out[out["date"] >= since].reset_index(drop=True)

    def fetch(self, symbol: str, refresh: bool = False) -> pd.DataFrame:
        cached = self.load(symbol)
        if cached is not None and not refresh:
            return cached
        try:
            op = self.from_opend(symbol)
        except Exception as exc:          # OpenD down or no F10 data: fall back to SEC alone, and say so
            print(f"{symbol}: OpenD earnings unavailable ({exc}); SEC filing dates only", file=sys.stderr)
            op = empty()
        time.sleep(self.pause_s)
        out = merge(op, self.from_sec(symbol))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        out.to_csv(self.path(symbol), index=False)
        return out


def filter_dates(symbols, etfs=(), cache_dir: str | Path = DEFAULT_CACHE) -> dict[str, pd.DatetimeIndex]:
    """symbol -> report instants for the ``no_earnings_3d`` filter. ETFs map to no dates;
    stocks with no cached table are left out, so the filter reports itself inactive for them."""
    e = Earnings(cache_dir)
    out: dict[str, pd.DatetimeIndex] = {s: pd.DatetimeIndex([], tz="UTC") for s in symbols if s in set(etfs)}
    for s in symbols:
        if s in out:
            continue
        t = e.load(s)
        if t is not None:
            out[s] = pd.DatetimeIndex([event_time(d, tm) for d, tm in zip(t["date"], t["timing"])], tz="UTC")
    return out


def calendar_events(symbols, cache_dir: str | Path = DEFAULT_CACHE) -> list:
    """Cached earnings as tradex.events.Event rows (kind ``earnings``, scope the symbol)."""
    from tradex.events import Event
    e = Earnings(cache_dir)
    out = []
    for s in symbols:
        t = e.load(s)
        for d, tm, src in ([] if t is None else zip(t["date"], t["timing"], t["source"])):
            out.append(Event(event_time(d, tm), "earnings", s, f"{src}, {tm}"))
    return sorted(out, key=lambda x: x.time)


def main(symbols: list[str]) -> int:
    e = Earnings()
    try:
        for s in symbols:
            t = e.fetch(s)
            src = t["source"].value_counts().to_dict() if len(t) else {}
            print(f"{s}: {len(t)} reports {t['date'].min() if len(t) else '-'}..{t['date'].max() if len(t) else '-'} {src}",
                  flush=True)
    finally:
        e.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
