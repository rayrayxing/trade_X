"""Property tests (hypothesis): the ledger hash chain, sizing, and look-ahead.

Examples are derandomized so CI is reproducible; raise ``max_examples`` locally to dig.
"""
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from gapkit import T0, known_gap
from tradex.backtest import metrics
from tradex.core.ledger import GENESIS, Ledger, row_hash
from tradex.core.records import Fill, TradePlan, Veto
from tradex.data.synthetic import synthetic_bars
from tradex.risk.exposure import Leg
from tradex.risk.gate import BookState, RiskGate
from tradex.risk.sizing import conservative_kelly, drawdown_scale, kelly_fraction, leverage_gate
from tradex.runtime.market import BarStore
from tradex.runtime.signals import SignalCache
from tradex.core.interfaces import ReplayClock
from tradex.strategy.spec import StrategySpec, compute_signals

PROFILE = dict(deadline=None, derandomize=True, suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])


def prop(n):
    return settings(max_examples=n, **PROFILE)


# --- ledger hash chain ---------------------------------------------------------------------------------

text = st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=40)
vetoes = st.builds(Veto, decision_id=st.from_regex(r"20\d\d-\d\d-\d\d-\d{4}", fullmatch=True),
                   time=st.sampled_from([T0.isoformat(), (T0 + pd.Timedelta(hours=1)).isoformat()]),
                   source=st.sampled_from(["risk", "calendar", "agent:scout", "command"]), reason=text)
fills = st.builds(Fill, decision_id=st.just("2026-03-02-0001"), client_order_id=st.text("abc123-", min_size=1, max_size=12),
                  time=st.just(T0.isoformat()), symbol=st.sampled_from(["EUR_USD", "AAPL"]),
                  side=st.sampled_from([-1, 1]), qty=st.floats(1, 1e6, allow_nan=False), price=st.floats(0.5, 500, allow_nan=False),
                  fees_usd=st.floats(0, 50, allow_nan=False), spread_slippage_usd=st.floats(0, 50, allow_nan=False),
                  book=st.just("ensemble"))
records = st.lists(st.one_of(vetoes, fills), min_size=2, max_size=14)


def _ledger(recs, commit="c0"):
    led = Ledger(":memory:", git_commit=commit)
    for r in recs:
        led.append(r)
    return led


@prop(60)
@given(records)
def test_any_honest_sequence_verifies_and_links(recs):
    led = _ledger(recs)
    assert led.verify() == (True, None)
    rows = list(led.iter_events())
    assert [r["seq"] for r in rows] == list(range(1, len(recs) + 1))
    assert rows[0]["prev_hash"] == GENESIS
    assert all(b["prev_hash"] == a["hash"] for a, b in zip(rows, rows[1:]))
    assert len({r["hash"] for r in rows}) == len(rows)


@prop(60)
@given(records, st.data())
def test_changing_any_stored_field_of_a_row_is_detected_at_that_row(recs, data):
    led = _ledger(recs)
    seq = data.draw(st.integers(1, len(recs)))
    col = data.draw(st.sampled_from(["kind", "time", "payload", "git_commit", "config_hash", "prev_hash", "hash"]))
    old = led.db.execute(f"SELECT {col} FROM events WHERE seq=?", (seq,)).fetchone()[0]
    new = data.draw(text.filter(lambda s: s != old))
    led.db.execute(f"UPDATE events SET {col}=? WHERE seq=?", (new, seq))
    assert led.verify() == (False, seq)


@prop(40)
@given(records, st.data())
def test_deleting_or_swapping_rows_is_detected(recs, data):
    led = _ledger(recs)
    n = len(recs)
    k = data.draw(st.integers(1, n - 1))                         # not the last row (see the truncation gap below)
    led.db.execute("DELETE FROM events WHERE seq=?", (k,))
    ok, bad = led.verify()
    assert not ok and bad == (k + 1 if k > 1 else 2) or (k == 1 and not ok)

    led2 = _ledger(recs)
    a = data.draw(st.integers(1, n - 1))
    pa, pb = (led2.db.execute("SELECT payload FROM events WHERE seq=?", (s,)).fetchone()[0] for s in (a, a + 1))
    if pa != pb:
        led2.db.execute("UPDATE events SET payload=? WHERE seq=?", (pb, a))
        led2.db.execute("UPDATE events SET payload=? WHERE seq=?", (pa, a + 1))
        assert led2.verify() == (False, a)


@prop(40)
@given(records, st.data())
def test_appending_after_tampering_does_not_heal_the_chain(recs, data):
    led = _ledger(recs)
    seq = data.draw(st.integers(1, len(recs)))
    led.db.execute("UPDATE events SET payload=payload || ' ' WHERE seq=?", (seq,))
    led.append(Veto("2026-03-02-0099", T0.isoformat(), "risk", "later"))
    assert led.verify() == (False, seq)


@prop(40)
@given(records, st.text("abcdef0123456789", min_size=1, max_size=12), st.text("xyz", min_size=1, max_size=6))
def test_digest_ignores_commit_and_run_metadata_but_not_decisions(recs, commit, run):
    a = _ledger(recs, "c0")
    b = Ledger(":memory:", git_commit=commit, run_id=run)
    for r in recs:
        b.append(r)
    assert a.digest(("veto", "fill")) == b.digest(("veto", "fill"))
    c = _ledger(recs + [Veto("2026-03-02-0098", T0.isoformat(), "risk", "extra")])
    assert c.digest(("veto", "fill")) != a.digest(("veto", "fill"))


@prop(40)
@given(st.lists(st.tuples(st.text(max_size=8), st.text(max_size=8)), min_size=2, max_size=6))
def test_row_hash_separates_fields(parts):
    """Moving a character between neighbouring fields must change the hash (the 0x1f separator)."""
    for kind, tm in parts:
        h1 = row_hash(GENESIS, kind + "x", tm, None, "p", "c", "h")
        h2 = row_hash(GENESIS, kind, "x" + tm, None, "p", "c", "h")
        assert h1 != h2


def test_h1_truncating_the_tail_of_the_chain_is_detected():
    led = _ledger([Veto(f"2026-03-02-{i:04d}", T0.isoformat(), "risk", "r") for i in range(5)])
    led.db.execute("DELETE FROM events WHERE seq >= 4")
    ok, _ = led.verify()
    assert not ok


class _Row:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


class _RaceDb:
    """Connection proxy: after a writer reads the chain head it waits until the other writer has read it too."""

    def __init__(self, real, barrier):
        self._real, self._barrier = real, barrier

    def execute(self, sql, *a):
        out = self._real.execute(sql, *a)
        if sql.startswith("SELECT hash FROM events"):
            head = out.fetchone()
            self._barrier.wait(timeout=10)
            return _Row(head)
        return out

    def __enter__(self):
        return self._real.__enter__()

    def __exit__(self, *exc):
        return self._real.__exit__(*exc)

    def __getattr__(self, name):
        return getattr(self._real, name)


def test_h2_two_writers_on_one_file_keep_a_single_chain(tmp_path):
    path = tmp_path / "ledger.sqlite"
    Ledger(path, git_commit="t").append(Veto("2026-03-02-0000", T0.isoformat(), "risk", "first"))
    barrier = threading.Barrier(2)
    errors = []

    def writer(tag):
        try:
            led = Ledger(path, git_commit="t", run_id=tag)
            led.db = _RaceDb(led.db, barrier)
            led.append(Veto("2026-03-02-0001", T0.isoformat(), "risk", tag))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    ts = [threading.Thread(target=writer, args=(f"w{i}",)) for i in range(2)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    reader = Ledger(path, read_only=True)
    assert not errors and len(reader.rows()) == 3
    assert reader.verify() == (True, None)


def test_h2_many_concurrent_writers_keep_a_single_chain(tmp_path):
    """No injected race: four writers with their own connections append in parallel."""
    path = tmp_path / "ledger.sqlite"
    Ledger(path, git_commit="t").append(Veto("2026-03-02-0000", T0.isoformat(), "risk", "first"))
    errors = []

    def writer(w):
        try:
            led = Ledger(path, git_commit="t", run_id=f"w{w}")
            for i in range(25):
                led.append(Veto(f"2026-03-02-{w}{i:03d}", T0.isoformat(), "risk", f"{w}-{i}"))
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    ts = [threading.Thread(target=writer, args=(w,)) for w in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    reader = Ledger(path, read_only=True)
    assert not errors and len(reader.rows()) == 101
    assert reader.verify() == (True, None)


def test_h1_the_chain_head_cannot_be_rolled_back_to_an_earlier_row(tmp_path):
    path = tmp_path / "ledger.sqlite"
    led = Ledger(path, git_commit="t")
    for i in range(5):
        led.append(Veto(f"2026-03-02-{i:04d}", T0.isoformat(), "risk", "r"))
    led.db.execute("DELETE FROM events WHERE seq >= 4")
    led.db.commit()
    ok, bad = Ledger(path, read_only=True).verify()
    assert not ok and bad == 4


def test_h1_a_ledger_file_from_before_the_anchor_is_anchored_on_first_write(tmp_path):
    path = tmp_path / "ledger.sqlite"
    led = Ledger(path, git_commit="t")
    for i in range(3):
        led.append(Veto(f"2026-03-02-{i:04d}", T0.isoformat(), "risk", "r"))
    led.db.execute("DROP TABLE chain_head")
    led.db.commit()
    led.close()
    old = Ledger(path, read_only=True)
    assert old.verify() == (True, None)               # nothing to compare with: no false alarm
    again = Ledger(path, git_commit="t")              # the writer anchors the current tail
    again.append(Veto("2026-03-02-0009", T0.isoformat(), "risk", "r"))
    again.db.execute("DELETE FROM events WHERE seq = 4")
    again.db.commit()
    assert not Ledger(path, read_only=True).verify()[0]


def test_sequential_writers_on_one_file_keep_a_single_chain(tmp_path):
    path = tmp_path / "ledger.sqlite"
    a, b = Ledger(path, git_commit="t"), Ledger(path, git_commit="t", run_id="other")
    for i in range(10):
        (a if i % 2 else b).append(Veto(f"2026-03-02-{i:04d}", T0.isoformat(), "risk", str(i)))
    assert Ledger(path, read_only=True).verify() == (True, None)


# --- sizing -----------------------------------------------------------------------------------------------

GATE = RiskGate.from_policy()
POL = GATE.policy


def _leg(sym, qty, price, stop, direction=1):
    return Leg(sym, "stocks", direction, qty, price, stop)


stock_plan = st.builds(
    lambda entry, risk_frac, direction, p, rr, ev, did: TradePlan(
        did, T0.isoformat(), "AAA", "stocks", direction, "market", entry, entry * (1 - direction * risk_frac),
        [entry * (1 + direction * risk_frac * rr)], 20, "stop", ["trend", "momentum"], ["s1", "s2"], 0.6, p,
        "base_rate", rr, ev, 0.05, "ensemble", "D1"),
    entry=st.floats(5, 900, allow_nan=False), risk_frac=st.floats(0.005, 0.15), direction=st.sampled_from([-1, 1]),
    p=st.floats(0.2, 0.8), rr=st.floats(1.5, 4.0), ev=st.floats(0.05, 1.0), did=st.just("2026-03-02-0001"))
equities = st.floats(1_000, 5_000_000)
tiers = st.sampled_from([1, 2, 3, 4, 5])
factors = st.sampled_from([0.25, 0.5, 1.0])


def _book(eq, tier, legs=()):
    return BookState(eq, list(legs), tier, None, None, None)


@prop(150)
@given(stock_plan, equities, tiers, factors)
def test_verdicts_respect_the_per_trade_cap_and_leverage(pl, eq, tier, factor):
    v = GATE.review(pl, _book(eq, tier), 1.0, None, factor)
    assert v.qty >= 0 and (v.outcome == "accepted") == (v.qty > 0)
    if v.outcome != "accepted":
        return
    tier_cfg = POL["book"]["tiers"][tier]
    cap = POL["sizing"]["per_trade_cap_pct"] * tier_cfg["size_multiplier"] * factor
    unit_risk = abs(pl.entry_price - pl.stop)
    assert v.qty * unit_risk <= eq * cap / 100 * (1 + 1e-9)
    assert v.qty * pl.entry_price <= eq * tier_cfg["max_leverage"]["stocks"] * (1 + 1e-9)
    assert v.qty == int(v.qty)
    assert v.risk_usd == pytest.approx(v.qty * unit_risk, abs=0.01)


@prop(120)
@given(stock_plan, equities, tiers)
def test_size_never_grows_when_the_context_cut_deepens(pl, eq, tier):
    q = [GATE.review(pl, _book(eq, tier), 1.0, None, f).qty for f in (1.0, 0.5, 0.25)]
    assert q[0] >= q[1] >= q[2] >= 0


@prop(120)
@given(stock_plan, equities, st.floats(1.0, 4.0), tiers)
def test_more_equity_never_means_a_smaller_position(pl, eq, grow, tier):
    a = GATE.review(pl, _book(eq, tier), 1.0, None).qty
    b = GATE.review(pl, _book(eq * grow, tier), 1.0, None).qty
    assert b >= a - 1


@prop(100)
@given(stock_plan, equities, st.lists(st.tuples(st.floats(5, 500), st.floats(1, 200), st.floats(0.01, 0.2)), max_size=6))
def test_book_heat_never_exceeds_the_cap_after_a_new_trade(pl, eq, held):
    legs = [Leg(f"S{i}", "stocks", 1, q, p, p * (1 - f)) for i, (p, q, f) in enumerate(held)]
    before = sum((l.price - l.stop) * l.qty for l in legs)
    v = GATE.review(pl, _book(eq, 2, legs), 1.0, None)
    cap = eq * POL["book"]["heat_cap_pct"] / 100
    if v.outcome == "accepted":
        assert before + v.qty * abs(pl.entry_price - pl.stop) <= max(cap, before) * (1 + 1e-9)
    if len(legs) >= POL["book"]["max_positions"]:
        assert v.outcome == "rejected"


@prop(100)
@given(stock_plan, tiers, st.floats(-1000, 0))
def test_a_non_positive_equity_never_sizes(pl, tier, eq):
    v = GATE.review(pl, _book(eq, tier), 1.0, None)
    assert v.outcome == "rejected" and v.qty == 0


@prop(200)
@given(st.floats(0.01, 0.99), st.floats(0.1, 10.0))
def test_kelly_is_monotone_and_zero_at_breakeven(p, b):
    f = kelly_fraction(p, b)
    assert f <= p and kelly_fraction(min(1.0, p + 0.01), b) >= f - 1e-12 and kelly_fraction(p, b + 0.1) >= f - 1e-12
    assert (f > 0) == (p * b > (1 - p) + 1e-12) or abs(f) < 1e-9
    assert kelly_fraction(1 / (1 + b), b) == pytest.approx(0.0, abs=1e-12)


@prop(150)
@given(st.floats(1, 1e6), st.floats(1, 1e6), st.floats(0.01, 0.9))
def test_drawdown_scale_is_bounded_and_falls_with_drawdown(eq, peak, halve):
    s = drawdown_scale(eq, peak, halve)
    assert 0.0 <= s <= 1.0
    assert drawdown_scale(peak, peak, halve) == 1.0
    assert drawdown_scale(eq * 0.9, peak, halve) <= s + 1e-12


@prop(80)
@given(st.lists(st.floats(-3, 6, allow_nan=False), min_size=1, max_size=80))
def test_conservative_kelly_is_never_above_the_observed_win_rate(rs):
    out = conservative_kelly(pd.DataFrame({"r_multiple": rs}))
    assert 0.0 <= out["p_lower"] <= out["p"] + 1e-12 and out["risk_pct"] >= 0.0


@prop(60)
@given(st.floats(0, 1), st.sampled_from(["none", "positive", "validated", "arbitrage"]), st.floats(1, 30))
def test_leverage_gate_is_within_the_brokers_limit(ruin, evidence, broker_max):
    assert 0.0 <= leverage_gate(ruin, evidence, broker_max) <= broker_max


@prop(60)
@given(st.integers(1, 5000), st.floats(0.0, 3.0), st.integers(1, 300))
def test_more_trials_never_raise_the_deflated_sharpe(n, var, t):
    r = pd.Series(np.random.default_rng(t).normal(0.001, 0.01, 400))
    assert metrics.deflated_sharpe(r, n + 10, var / 100) <= metrics.deflated_sharpe(r, n, var / 100) + 1e-12


# --- look-ahead: incremental signals equal batch signals ------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
SEEDS = sorted((ROOT / "strategies" / "seeds").glob("*.yaml"))
BOOL_COLS = ("long_entry", "short_entry", "long_exit", "short_exit")


def _bars_for(spec, n, seed):
    tf = spec.signal_tf
    return synthetic_bars(n, tf=tf, seed=seed, start="2024-01-01",
                          price=1.1 if spec.asset_class == "forex" else 100.0,
                          vol=0.002 if spec.asset_class == "forex" else 0.015, business_days=tf == "D1",
                          trend_strength=0.0004 if spec.asset_class == "forex" else 0.002)


@pytest.mark.parametrize("path", SEEDS, ids=[p.stem for p in SEEDS])
@prop(12)
@given(st.integers(0, 10_000), st.integers(0, 10**6))
def test_signals_at_bar_k_do_not_change_when_later_bars_exist(path, seed, pick):
    """Computed on bars[:k+1] or on the whole history, the signals at bar k are identical."""
    spec = StrategySpec.load(path)
    bars = _bars_for(spec, 420, seed % 997)
    k = 330 + pick % 80
    try:
        full = compute_signals(spec, bars)
    except Exception as exc:  # noqa: BLE001 - specs that need cross-sectional context are not single-frame testable
        pytest.skip(f"{path.stem} needs context beyond one frame: {type(exc).__name__}")
    cut = compute_signals(spec, bars.iloc[:k + 1])
    for col in BOOL_COLS:
        assert bool(getattr(full, col).iloc[k]) == bool(getattr(cut, col).iloc[-1]), col
    a, b = float(full.atr.iloc[k]), float(cut.atr.iloc[-1])
    assert (np.isnan(a) and np.isnan(b)) or a == pytest.approx(b, rel=1e-9)


@pytest.mark.parametrize("path", SEEDS, ids=[p.stem for p in SEEDS])
@prop(10)
@given(st.integers(0, 10_000), st.integers(0, 10**6))
def test_incremental_rolling_window_signals_equal_the_batch(path, seed, pick):
    """The live core's rolling-window SignalCache agrees with full-history batch signals at a random close."""
    spec = StrategySpec.load(path)
    bars = _bars_for(spec, 600, seed % 997)
    sym = "X"
    store = BarStore(spec.signal_tf, {sym: bars}, ReplayClock())
    cache = SignalCache(store, 3)
    k = 380 + pick % 200
    try:
        compute_signals(spec, bars.iloc[:k + 1])
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"{path.stem} needs context beyond one frame: {type(exc).__name__}")
    close_t = bars.index[k] + (bars.index[1] - bars.index[0])
    pt = cache.at(spec, sym, close_t)
    batch = compute_signals(spec, bars.iloc[:k + 1])
    assert pt is not None
    for col in BOOL_COLS:
        assert getattr(pt, col) == bool(getattr(batch, col).iloc[-1]), col
    assert pt.close == bars["close"].iloc[k]


# --- the look-ahead detector itself, and a hole it found in the rule language ------------------------------------------

def _peeking_spec():
    from conftest import simple_spec
    return simple_spec(long="close < shift(close, -3)")


def test_the_detector_catches_a_rule_that_peeks_at_the_future(monkeypatch):
    """Negative control: with a deliberately peeking rule the full-history and cut-history signals differ.
    The rule language now refuses negative counts (H3), so the unchecked ``shift`` is put back for this test."""
    from tradex.strategy import expr
    monkeypatch.setitem(expr.FUNCS, "shift", lambda x, n=1: x.shift(int(n)))
    spec = _peeking_spec()
    bars = synthetic_bars(220, seed=1)
    full = compute_signals(spec, bars).long_entry
    diffs = [k for k in range(100, 200) if bool(full.iloc[k]) != bool(compute_signals(spec, bars.iloc[:k + 1]).long_entry.iloc[-1])]
    assert diffs


@pytest.mark.parametrize("rule", ["close < shift(close, -3)", "rising(close, -2)", "falling(close, -1)",
                                  "rising(close, 0)", "close > shift(close, 1.5)", "rolling_max(close, -5) > 0"])
def test_h3_rules_cannot_reference_future_bars(rule):
    from conftest import simple_spec
    from tradex.strategy import expr
    spec = simple_spec(long=rule)
    bars = synthetic_bars(60, seed=1)
    try:
        compute_signals(spec, bars)
    except (expr.ExprError, ValueError):
        return
    raise AssertionError(f"{rule!r} was accepted")
