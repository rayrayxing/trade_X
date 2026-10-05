"""Refresh the calendar files under data/calendar/ from the real sources, all or nothing.

    python -m tradex.events.refresh --contact you@example.com --symbols NVDA AMD AAPL --days 120

Run it on Ray's Mac with OpenD up (a quote context is opened read-only on 127.0.0.1:11111; no
trade context is ever opened). It writes ``us_jobs.yaml`` (BLS schedule, cross-checked against the
OpenD economic calendar) and ``earnings.yaml`` (OpenD earnings calendar) in the format
``EventCalendar.add_file`` loads. Both sources are read and validated first; if either raises
``CalendarDataMissing`` nothing is written and the previous files stay as they were.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

from tradex.events.earnings import import_earnings
from tradex.events.jobs_report import HttpBlsClient, import_jobs_reports
from tradex.events.opend_calendar import OpenDCalendar
from tradex.events.sources import CalendarDataMissing, write_events_yaml

DEFAULT_DIR = Path(__file__).resolve().parents[2] / "data" / "calendar"


def refresh(*, start, end, symbols: Iterable[str], bls, opend, out_dir: str | Path = DEFAULT_DIR,
            assume_tz: str | None = None) -> dict[str, Path]:
    jobs = import_jobs_reports(bls, opend, start=start, end=end, assume_tz=assume_tz)
    earnings = import_earnings(opend, list(symbols), start, end) if list(symbols) else []
    out = Path(out_dir)
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    written = {"jobs": write_events_yaml(jobs, out / "us_jobs.yaml",
                                         f"US jobs reports (nfp), generated {stamp} by tradex.events.refresh.\n"
                                         "Source: BLS release schedule, cross-checked against the OpenD economic calendar.")}
    if earnings:
        written["earnings"] = write_events_yaml(earnings, out / "earnings.yaml",
                                                f"Earnings dates, generated {stamp} by tradex.events.refresh.\n"
                                                "Source: OpenD earnings calendar.")
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--contact", required=True, help="email for the BLS User-Agent")
    ap.add_argument("--symbols", nargs="*", default=[])
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--out", default=str(DEFAULT_DIR))
    a = ap.parse_args(argv)
    from moomoo import OpenQuoteContext                          # only on the machine that runs OpenD
    quote = OpenQuoteContext("127.0.0.1", 11111)
    try:
        now = pd.Timestamp.now(tz="UTC")
        files = refresh(start=now, end=now + pd.Timedelta(days=a.days), symbols=a.symbols,
                        bls=HttpBlsClient(a.contact), opend=OpenDCalendar(quote), out_dir=a.out)
    except CalendarDataMissing as exc:
        print(f"calendar NOT refreshed (nothing written): {exc}", file=sys.stderr)
        return 1
    finally:
        quote.close()
    for name, path in files.items():
        print(f"wrote {name}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
