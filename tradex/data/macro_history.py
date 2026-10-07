"""Official macro history for research: central-bank policy rates, FOMC dates, Cboe VIX.

Every series comes from the publisher (or, where the publisher blocks scripted access, the
BIS policy-rate dataset that republishes it, labelled as such). Nothing is typed in by hand.
Raw responses are cached under data/cache/macro/ (gitignored) next to a ``.source.json``
holding the URL and fetch time; the derived tables are rebuilt from those files:

  policy_rates.csv   date, currency, rate (percent, from its effective date), source
                     (header comment: one URL per currency, fetch date)
  fomc_history.csv   time (UTC instant of the statement), date, release_rule, source:
                     scheduled FOMC meetings only (no conference calls, notation votes or
                     unscheduled meetings), from federalreserve.gov
  vix_daily.csv      date, open, high, low, close: Cboe's VIX_History.csv

Statement times are not printed on the Fed's calendar pages, so they follow the Fed's
release practice: 14:00 New York from 2013, 12:30 on 2011-2012 press-conference days,
14:15 otherwise (``release_rule`` says which applied).

    python -m tradex.data.macro_history rates|fomc|vix|all [--write-bundled] [--no-fetch]
"""
from __future__ import annotations

import csv
import io
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache" / "macro"
BUNDLED = ROOT / "tradex" / "costs" / "policy_rates.csv"
NY = "America/New_York"
USER_AGENT = "trade-x research ruix.zheng@gmail.com"
HOSTS = {"fred.stlouisfed.org", "data-api.ecb.europa.eu", "www.bankofengland.co.uk", "stats.bis.org",
         "www.rba.gov.au", "www.bankofcanada.ca", "data.snb.ch", "www.federalreserve.gov", "cdn.cboe.com",
         "cdn-api.cboe.com"}

BIS = "https://stats.bis.org/api/v2/data/dataflow/BIS/WS_CBPOL/1.0/D.{area}?format=csv&startPeriod=2000-01-01"
# raw file name -> URL. Policy rates: USD Fed funds target (single target to 2008-12-15, then the
# range midpoint), EUR ECB deposit facility, GBP Bank Rate, AUD cash rate target, CAD target for the
# overnight rate, CHF SNB policy rate (from 2019-06-13; before it the midpoint of the 3-month Libor
# target range), JPY and NZD from BIS (Bank of Japan has no policy-rate series; RBNZ blocks scripts).
RAW = {
    "fred_DFEDTAR.csv": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFEDTAR",
    "fred_DFEDTARU.csv": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFEDTARU",
    "fred_DFEDTARL.csv": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFEDTARL",
    "ecb_DFR.csv": "https://data-api.ecb.europa.eu/service/data/FM/D.U2.EUR.4F.KR.DFR.LEV?format=csvdata&startPeriod=2000-01-01",
    "boe_IUDBEDR.csv": ("https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp?csv.x=yes"
                        "&Datefrom=01/Jan/2000&Dateto=now&SeriesCodes=IUDBEDR&CSVF=TN&UsingCodes=Y"),
    "rba_f1.csv": "https://www.rba.gov.au/statistics/tables/csv/f1-data.csv",
    "boc_V39079.csv": "https://www.bankofcanada.ca/valet/observations/V39079/csv",
    "snb_snbgwdzid.csv": "https://data.snb.ch/api/cube/snbgwdzid/data/csv/en",
    "snb_snboffzisa.csv": "https://data.snb.ch/api/cube/snboffzisa/data/csv/en",
    "bis_JP.csv": BIS.format(area="JP"),
    "bis_NZ.csv": BIS.format(area="NZ"),
    "bis_US.csv": BIS.format(area="US"), "bis_XM.csv": BIS.format(area="XM"), "bis_GB.csv": BIS.format(area="GB"),
    "bis_AU.csv": BIS.format(area="AU"), "bis_CA.csv": BIS.format(area="CA"), "bis_CH.csv": BIS.format(area="CH"),
    "cboe_VIX_History.csv": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv",
    "fed_fomccalendars.htm": "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
}
FOMC_FIRST_YEAR = 2006
FOMC_HIST = "https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm"
RATE_FILES = {"USD": ["fred_DFEDTAR.csv", "fred_DFEDTARU.csv", "fred_DFEDTARL.csv"], "EUR": ["ecb_DFR.csv"],
              "GBP": ["boe_IUDBEDR.csv"], "JPY": ["bis_JP.csv"], "AUD": ["rba_f1.csv"], "CAD": ["boc_V39079.csv"],
              "CHF": ["snb_snbgwdzid.csv", "snb_snboffzisa.csv"], "NZD": ["bis_NZ.csv"]}
BIS_AREA = {"USD": "US", "EUR": "XM", "GBP": "GB", "AUD": "AU", "CAD": "CA", "CHF": "CH", "JPY": "JP", "NZD": "NZ"}


# --- fetch ---------------------------------------------------------------------------------

def http_get(url: str) -> bytes:
    if urllib.parse.urlparse(url).hostname not in HOSTS:
        raise ValueError(f"not an allowed official host: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    delay = 2.0
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def fetch_raw(name: str, url: str, cache: Path = CACHE, get=http_get) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    body = get(url)
    (cache / name).write_bytes(body)
    (cache / f"{name}.source.json").write_text(json.dumps(
        {"url": url, "fetched_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"), "bytes": len(body)}))
    return cache / name


def source_of(name: str, cache: Path = CACHE) -> dict:
    p = cache / f"{name}.source.json"
    return json.loads(p.read_text()) if p.exists() else {"url": RAW.get(name, ""), "fetched_at": "unknown"}


def _text(p: Path) -> str:
    return p.read_bytes().decode("utf-8-sig", errors="replace")


# --- policy-rate parsers: each returns a daily or change-date series in percent ---------------

def _series(dates, values) -> pd.Series:
    s = pd.Series(pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(),
                  index=pd.DatetimeIndex(pd.to_datetime(list(dates))))
    return s.dropna().sort_index()


def parse_fred(text: str) -> pd.Series:
    df = pd.read_csv(io.StringIO(text))
    return _series(df.iloc[:, 0], df.iloc[:, 1])


def parse_ecb(text: str) -> pd.Series:
    df = pd.read_csv(io.StringIO(text), usecols=["TIME_PERIOD", "OBS_VALUE"])
    return _series(df["TIME_PERIOD"], df["OBS_VALUE"])


def parse_boe(text: str) -> pd.Series:
    df = pd.read_csv(io.StringIO(text))
    return _series(pd.to_datetime(df["DATE"], format="%d %b %Y"), df.iloc[:, 1])


def parse_rba_f1(text: str, series_id: str = "FIRMMCRTD") -> pd.Series:
    rows = list(csv.reader(io.StringIO(text)))
    head = next(i for i, r in enumerate(rows) if r and r[0].strip() == "Series ID")
    col = rows[head].index(series_id)
    body = [r for r in rows[head + 1:] if r and re.match(r"\d{2}-[A-Za-z]{3}-\d{4}", r[0]) and len(r) > col]
    return _series(pd.to_datetime([r[0] for r in body], format="%d-%b-%Y"), [r[col] or None for r in body])


def parse_boc(text: str, series: str = "V39079") -> pd.Series:
    obs = text.split('"OBSERVATIONS"', 1)[1]
    df = pd.read_csv(io.StringIO(obs.strip()))
    return _series(df["date"], df[series])


def parse_snb(text: str, item: str) -> pd.Series:
    lines = text.splitlines()
    head = next(i for i, l in enumerate(lines) if l.startswith('"Date"'))
    df = pd.read_csv(io.StringIO("\n".join(lines[head:])), sep=";", dtype=str)
    df = df[df["D0"] == item]
    return _series(df["Date"], df["Value"])


def parse_bis(text: str) -> pd.Series:
    df = pd.read_csv(io.StringIO(text), usecols=["TIME_PERIOD", "OBS_VALUE"])
    return _series(df["TIME_PERIOD"], df["OBS_VALUE"])


def changes(s: pd.Series) -> pd.Series:
    """Keep the first observation and every date where the value changes (rounded to 1/1000 pp)."""
    s = s.round(3)
    return s[s.ne(s.shift())]


def currency_series(ccy: str, cache: Path = CACHE) -> pd.Series:
    """The policy-rate change points of one currency, from its cached raw file(s)."""
    f = {n: _text(cache / n) for n in RATE_FILES[ccy]}
    if ccy == "USD":
        single = parse_fred(f["fred_DFEDTAR.csv"])
        mid = (parse_fred(f["fred_DFEDTARU.csv"]) + parse_fred(f["fred_DFEDTARL.csv"])) / 2
        s = pd.concat([single[single.index < mid.index[0]], mid.dropna()])
    elif ccy == "EUR":
        s = parse_ecb(f["ecb_DFR.csv"])
    elif ccy == "GBP":
        s = parse_boe(f["boe_IUDBEDR.csv"])
    elif ccy == "AUD":
        s = parse_rba_f1(f["rba_f1.csv"])
    elif ccy == "CAD":
        s = parse_boc(f["boc_V39079.csv"])
    elif ccy == "CHF":
        daily = parse_snb(f["snb_snbgwdzid.csv"], "LZ")
        lo, hi = parse_snb(f["snb_snboffzisa.csv"], "UG0"), parse_snb(f["snb_snboffzisa.csv"], "OG0")
        band = ((lo + hi) / 2).dropna()
        band.index = band.index + pd.offsets.MonthEnd(0)   # a monthly value is only known by the month's end
        s = pd.concat([band[band.index < daily.index[0]], daily])
    else:
        s = parse_bis(f[RATE_FILES[ccy][0]])
    return changes(s)


def rates_table(cache: Path = CACHE, since: str = "2000-01-01") -> pd.DataFrame:
    rows = []
    for ccy, files in RATE_FILES.items():
        s = currency_series(ccy, cache)
        s = s[s.index >= since]
        src = "bis" if files[0].startswith("bis_") else files[0].split("_")[0]
        rows.append(pd.DataFrame({"date": s.index.strftime("%Y-%m-%d"), "currency": ccy, "rate": s.values,
                                  "source": src}))
    return pd.concat(rows, ignore_index=True)


def rates_header(cache: Path = CACHE, bundled: bool = False) -> str:
    lines = ["# Central-bank policy rates (percent) from the publishers' own data, rebuilt by",
             "# `python -m tradex.data.macro_history rates`. One row per change, from its effective date.",
             "# USD = Fed funds target (midpoint of the range from 2008-12-16). EUR = ECB deposit facility.",
             "# GBP = Bank Rate. JPY = BoJ policy rate as compiled by the BIS. AUD = cash rate target.",
             "# CAD = target for the overnight rate. CHF = SNB policy rate (3M Libor target-range midpoint before",
             "# 2019-06-13, dated at month end). NZD = official cash rate as compiled by the BIS (rbnz.govt.nz",
             "# refuses scripted downloads)."]
    if bundled:
        lines.append("# Backtests only: live and paper read financing from Oanda's instrument `financing` field.")
    for ccy, files in RATE_FILES.items():
        for n in files:
            m = source_of(n, cache)
            lines.append(f"# {ccy}: {m['url']} fetched {m['fetched_at'][:10]}")
    return "\n".join(lines) + "\n"


def write_rates(cache: Path = CACHE, out: Path | None = None) -> Path:
    out = out or cache / "policy_rates.csv"
    out.write_text(rates_header(cache) + rates_table(cache).to_csv(index=False))
    return out


def write_bundled(cache: Path = CACHE, path: Path = BUNDLED, since: str = "2015-01-01") -> Path:
    """The cost model's bundled table (date,currency,rate), from the same official series."""
    t = rates_table(cache)
    first = t[t["date"] < since].groupby("currency").tail(1).assign(date=since)   # the rate in force at ``since``
    t = pd.concat([first, t[t["date"] >= since]]).sort_values(["currency", "date"])
    path.write_text(rates_header(cache, bundled=True) + t[["date", "currency", "rate"]].to_csv(index=False))
    return path


def bis_check(cache: Path = CACHE, since: str = "2015-01-01") -> dict[str, list[str]]:
    """Days since ``since`` where a primary series and the BIS copy disagree by more than 1 bp,
    compared on BIS's daily dates (a lag of a day or two at a change shows up here too)."""
    out = {}
    for ccy, area in BIS_AREA.items():
        if RATE_FILES[ccy][0].startswith("bis_") or not (cache / f"bis_{area}.csv").exists():
            continue
        bis = parse_bis(_text(cache / f"bis_{area}.csv"))
        bis = bis[bis.index >= since]
        prim = currency_series(ccy, cache)
        mine = prim.reindex(prim.index.union(bis.index)).ffill().reindex(bis.index)
        bad = bis[(mine - bis).abs() > 0.01]
        out[ccy] = [f"{d.date()} primary {mine[d]} bis {v}" for d, v in bad.items()]
    return out


# --- FOMC ------------------------------------------------------------------------------------

MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov",
                                      "dec"], 1)}     # by the first three letters: pages use both "Jan" and "January"
HIST_HEAD = re.compile(r"<h5[^>]*>\s*([^<]+?)\s*</h5>")
HEAD_RE = re.compile(r"^(?P<m1>[A-Za-z]+)(?:/(?P<m2>[A-Za-z]+))?\s+(?P<d1>\d+)"
                     r"(?:\s*-\s*(?:(?P<m3>[A-Za-z]+)\s+)?(?P<d2>\d+))?\s*(?P<rest>.*?)\s+-\s+(?P<y>\d{4})$")


def release_time(day: pd.Timestamp, press_conference: bool) -> tuple[pd.Timestamp, str]:
    if day.year >= 2013:
        hm, rule = (14, 0), "14:00 ET (2013 on)"
    elif press_conference and day.year >= 2011:
        hm, rule = (12, 30), "12:30 ET (2011-2012 press-conference day)"
    else:
        hm, rule = (14, 15), "14:15 ET (before 2013)"
    t = pd.Timestamp(day.year, day.month, day.day, *hm).tz_localize(NY)
    return t.tz_convert("UTC"), rule


def _last_day(m1, m2, m3, d1, d2, year) -> pd.Timestamp:
    month = (m3 or m2 or m1) if d2 else m1
    return pd.Timestamp(int(year), MONTHS[month.lower()[:3]], int(d2 or d1))


def parse_fomc_historical(html: str) -> list[dict]:
    """Scheduled meetings on a fomchistoricalYYYY.htm page: headings ``<dates> Meeting - YYYY``."""
    out = []
    parts = HIST_HEAD.split(html)
    for i in range(1, len(parts), 2):
        head, body = " ".join(parts[i].split()), parts[i + 1]
        m = HEAD_RE.match(head)
        if not m or m["rest"].strip() != "Meeting":
            continue          # conference calls, notation votes, (unscheduled), (cancelled)
        day = _last_day(m["m1"], m["m2"], m["m3"], m["d1"], m["d2"], m["y"])
        out.append({"date": day, "press_conference": "Press Conference" in body})
    return out


def parse_fomc_calendar(html: str) -> list[dict]:
    """Scheduled meetings on fomccalendars.htm (the last five years and the next one)."""
    out = []
    for block in re.split(r'<h4><a id="\d+">', html)[1:]:
        y = re.match(r"(\d{4}) FOMC Meetings", block)
        if not y:
            continue
        rows = re.findall(r'fomc-meeting__month[^>]*><strong>([^<]+)</strong></div>\s*'
                          r'<div class="fomc-meeting__date[^>]*>([^<]+)</div>', block)
        for month, dates in rows:
            dates = dates.strip()
            m = re.fullmatch(r"(\d+)(?:-(\d+))?\*?", dates)
            if not m:
                continue      # notation votes, unscheduled meetings
            ms = month.strip().split("/")
            day = pd.Timestamp(int(y[1]), MONTHS[(ms[-1] if m[2] else ms[0]).strip().lower()[:3]], int(m[2] or m[1]))
            out.append({"date": day, "press_conference": True})
    return out


def fomc_table(cache: Path = CACHE) -> pd.DataFrame:
    cal_name = "fed_fomccalendars.htm"
    cal = parse_fomc_calendar(_text(cache / cal_name))
    first_cal = min(r["date"].year for r in cal)
    rows = [r | {"source": RAW[cal_name]} for r in cal]
    for y in range(FOMC_FIRST_YEAR, first_cal):
        n = f"fed_fomchistorical{y}.htm"
        rows += [r | {"source": FOMC_HIST.format(year=y)} for r in parse_fomc_historical(_text(cache / n))]
    out = []
    for r in sorted(rows, key=lambda r: r["date"]):
        t, rule = release_time(r["date"], r["press_conference"])
        out.append({"time": t.isoformat(), "date": r["date"].strftime("%Y-%m-%d"), "release_rule": rule,
                    "source": r["source"]})
    return pd.DataFrame(out).drop_duplicates("date")


def write_fomc(cache: Path = CACHE) -> Path:
    t = fomc_table(cache)
    fetched = source_of("fed_fomccalendars.htm", cache)["fetched_at"][:10]
    p = cache / "fomc_history.csv"
    p.write_text(f"# Scheduled FOMC meetings from federalreserve.gov (fomccalendars.htm and fomchistoricalYYYY.htm),"
                 f" fetched {fetched}. time = statement release (UTC), see release_rule.\n" + t.to_csv(index=False))
    return p


# --- VIX ---------------------------------------------------------------------------------------

def parse_vix(text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text))
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y").dt.strftime("%Y-%m-%d")
    return df[["date", "open", "high", "low", "close"]]


def write_vix(cache: Path = CACHE) -> Path:
    m = source_of("cboe_VIX_History.csv", cache)
    p = cache / "vix_daily.csv"
    p.write_text(f"# Cboe VIX index daily OHLC from {m['url']}, fetched {m['fetched_at'][:10]}.\n"
                 + parse_vix(_text(cache / "cboe_VIX_History.csv")).to_csv(index=False))
    return p


def load_vix(path: Path) -> pd.Series:
    """VIX close by session date (tz-naive)."""
    df = pd.read_csv(path, comment="#")
    return pd.Series(df["close"].to_numpy(float), index=pd.DatetimeIndex(pd.to_datetime(df["date"])))


# --- CLI ---------------------------------------------------------------------------------------

def fetch(what: str, cache: Path = CACHE) -> list[str]:
    done = []
    names = {"rates": [n for f in RATE_FILES.values() for n in f] + [f"bis_{a}.csv" for a in BIS_AREA.values()],
             "vix": ["cboe_VIX_History.csv"], "fomc": ["fed_fomccalendars.htm"]}
    for kind in (["rates", "fomc", "vix"] if what == "all" else [what]):
        for n in dict.fromkeys(names[kind]):
            fetch_raw(n, RAW[n], cache)
            done.append(n)
        if kind == "fomc":
            first_cal = min(r["date"].year for r in parse_fomc_calendar(_text(cache / "fed_fomccalendars.htm")))
            for y in range(FOMC_FIRST_YEAR, first_cal):
                fetch_raw(f"fed_fomchistorical{y}.htm", FOMC_HIST.format(year=y), cache)
                done.append(f"fed_fomchistorical{y}.htm")
                time.sleep(0.5)
    return done


def main(argv: list[str]) -> int:
    what = argv[0] if argv else "all"
    if "--no-fetch" not in argv:      # rebuild the derived tables from the raw files already cached
        fetch(what)
    if what in ("rates", "all"):
        print(write_rates())
        for ccy, bad in bis_check().items():
            print(f"{ccy}: {'matches BIS' if not bad else f'{len(bad)} days differ from BIS, e.g. {bad[:3]}'}")
        if "--write-bundled" in argv:
            print(write_bundled())
    if what in ("fomc", "all"):
        print(write_fomc())
    if what in ("vix", "all"):
        print(write_vix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
