import ast
from pathlib import Path

import pandas as pd
import pytest

from tradex.data import opend


class FakeCtx:
    """Replays OpenD responses recorded on 4 Oct 2026 (US.SPY 60-minute bars, qfq)."""

    def __init__(self, seen=(), remaining=300):
        self.seen, self.remaining, self.calls = set(seen), remaining, []

    def get_history_kl_quota(self, get_detail=False):
        return 0, (300 - self.remaining, self.remaining, [{"code": c} for c in self.seen])

    def request_history_kline(self, code, start, end, ktype, autype, max_count, page_req_key):
        self.calls.append((code, ktype, autype, page_req_key))
        pages = {None: (["2026-09-29 10:30:00", "2026-09-29 11:30:00"], b"p2"),
                 b"p2": (["2026-09-29 15:30:00", "2026-09-29 16:00:00"], None)}
        times, nxt = pages[page_req_key]
        rows = {"2026-09-29 10:30:00": (766.83, 766.98, 764.60, 765.31, 5883595.0),
                "2026-09-29 11:30:00": (765.30, 765.75, 762.57, 762.9398, 4067188.0),
                "2026-09-29 15:30:00": (763.965, 765.30, 763.57, 764.57, 3901633.0),
                "2026-09-29 16:00:00": (764.59, 764.83, 763.76, 764.20, 8264700.0)}
        df = pd.DataFrame([dict(code=code, time_key=t, **dict(zip(["open", "high", "low", "close", "volume"], rows[t])))
                           for t in times])
        return 0, df, nxt


def fetcher(tmp_path, **kw):
    sleeps = []
    lim = opend.RateLimiter(n=60, window_s=30, sleep=sleeps.append)
    return opend.OpenDFetcher(cache_dir=tmp_path, limiter=lim, **kw)


def test_hourly_bars_are_restamped_to_open_time_utc(tmp_path):
    ctx = FakeCtx()
    f = fetcher(tmp_path, ctx=ctx)
    bars = f.fetch("SPY", "H1", "2026-09-29", "2026-09-30")
    assert [c[3] for c in ctx.calls] == [None, b"p2"]                       # paged
    assert all(c[1] == "K_60M" and c[2] == "qfq" for c in ctx.calls)
    # 10:30 NY end label -> 09:30 NY open = 13:30 UTC (EDT); the 16:00 half bar -> 15:00 NY.
    assert list(bars.index) == [pd.Timestamp(x, tz="UTC") for x in
                                ("2026-09-29 13:30", "2026-09-29 14:30", "2026-09-29 18:30", "2026-09-29 19:00")]
    assert bars["close"].iloc[-1] == 764.20
    # cached: a second call does not touch OpenD
    f.fetch("SPY", "H1", "2026-09-29", "2026-09-30")
    assert len(ctx.calls) == 2


def test_daily_bars_stamped_midnight_new_york():
    df = pd.DataFrame({"time_key": ["2026-01-05 00:00:00"], "open": [1.0], "high": [2.0], "low": [0.5],
                       "close": [1.5], "volume": [10.0]})
    assert opend.to_bars(df, "D1").index[0] == pd.Timestamp("2026-01-05 05:00", tz="UTC")


def test_quota_budget_and_exhaustion(tmp_path):
    f = fetcher(tmp_path, ctx=FakeCtx(), max_new_symbols=1)
    f.fetch("SPY", "H1", "2026-09-29", "2026-09-30")
    with pytest.raises(opend.QuotaExceeded, match="budget"):
        f.fetch("QQQ", "H1", "2026-09-29", "2026-09-30")
    f2 = fetcher(tmp_path / "b", ctx=FakeCtx(remaining=0))
    with pytest.raises(opend.QuotaExceeded, match="exhausted"):
        f2.fetch("IWM", "H1", "2026-09-29", "2026-09-30")
    # a symbol already downloaded this month costs no quota
    f3 = fetcher(tmp_path / "c", ctx=FakeCtx(seen={"US.IWM"}, remaining=0), max_new_symbols=0)
    assert len(f3.fetch("IWM", "H1", "2026-09-29", "2026-09-30")) == 4


def test_rate_limiter_waits_when_window_full():
    t = [0.0]
    slept = []
    lim = opend.RateLimiter(n=2, window_s=30, clock=lambda: t[0], sleep=lambda s: (slept.append(s), t.__setitem__(0, t[0] + s)))
    lim.wait(); lim.wait(); lim.wait()
    assert slept and slept[0] == pytest.approx(30.05)


def test_module_never_opens_a_trade_context():
    src = Path(opend.__file__).read_text()
    names = {n.id if isinstance(n, ast.Name) else n.attr for n in ast.walk(ast.parse(src))
             if isinstance(n, (ast.Name, ast.Attribute))}
    imported = {a.name for n in ast.walk(ast.parse(src)) if isinstance(n, ast.ImportFrom) for a in n.names}
    assert not any("Trade" in x for x in names | imported)
    assert imported & {"OpenQuoteContext"}
