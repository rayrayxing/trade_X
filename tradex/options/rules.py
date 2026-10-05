"""Entry and holding rules for naked short calls.

Ray's brief (4 Oct 2026): naked calls only with an always-set buy-to-close stop, sized as if
a 20% gap past the stop, never held through earnings, never on heavily shorted names. This
module is the unprotected, testable form of those rules for the planner and the backtester.
The enforcement copy lives in the protected risk gate and order guard (see
``patches/options-order-guard.patch``); both read the same thresholds from
``config/risk/policy.yaml`` once the patch adds its ``options`` section.

Unknown data blocks: if the earnings date or short-interest figures are not available the
trade is refused, because a naked call cannot be allowed to pass on missing information.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from tradex.options.contract import NY
from tradex.options.providers import UnderlyingRiskProvider
from tradex.options.sizing import GAP_PCT_FLOOR

DEFAULT_POLICY = Path(__file__).resolve().parents[2] / "config" / "risk" / "policy.yaml"


@dataclass
class NakedCallPolicy:
    gap_pct: float = GAP_PCT_FLOOR
    stop_required: bool = True
    block_through_earnings: bool = True
    earnings_buffer_days: int = 1              # close at least this many days before the report
    max_short_interest_pct_float: float = 20.0
    max_days_to_cover: float = 5.0
    require_known_data: bool = True

    @classmethod
    def from_policy(cls, path: str | Path | None = None) -> "NakedCallPolicy":
        """Thresholds from the risk policy: ``shorts`` for squeeze limits, ``options.naked_call``
        (present once the patch is applied) for the rest. Tighter values win over the defaults."""
        doc: dict[str, Any] = yaml.safe_load(Path(path or DEFAULT_POLICY).read_text()) or {}
        pol = cls()
        shorts = doc.get("shorts", {})
        pol.max_short_interest_pct_float = float(shorts.get("max_short_interest_pct_float", pol.max_short_interest_pct_float))
        pol.max_days_to_cover = float(shorts.get("max_days_to_cover", pol.max_days_to_cover))
        naked = (doc.get("options") or {}).get("naked_call") or {}
        for k in ("gap_pct", "earnings_buffer_days"):
            if k in naked:
                setattr(pol, k, type(getattr(pol, k))(naked[k]))
        for k in ("stop_required", "block_through_earnings", "require_known_data"):
            if k in naked:
                setattr(pol, k, bool(naked[k]))
        pol.max_short_interest_pct_float = float(naked.get("max_short_interest_pct_float", pol.max_short_interest_pct_float))
        pol.max_days_to_cover = float(naked.get("max_days_to_cover", pol.max_days_to_cover))
        pol.gap_pct = max(pol.gap_pct, GAP_PCT_FLOOR)
        return pol


def stop_problem(stop_premium: float | None, credit: float, policy: NakedCallPolicy) -> str | None:
    """Reason a naked call's buy-to-close stop is not acceptable, else None."""
    if not policy.stop_required:
        return None
    if stop_premium is None or stop_premium <= 0:
        return "naked call has no buy-to-close stop"
    if stop_premium <= credit:
        return f"buy-to-close stop {stop_premium:.2f} is not above the credit {credit:.2f}: no protection"
    return None


def blocker_kind(reasons: list[str]) -> str:
    """Category of the most important blocker: earnings, squeeze or unknown_data ('' when none)."""
    kinds = [r.split(":", 1)[0] for r in reasons]
    for k in ("earnings", "squeeze", "unknown_data"):
        if k in kinds:
            return k
    return ""


def naked_call_blockers(policy: NakedCallPolicy, underlying: str, asof: pd.Timestamp, horizon_end: dt.date,
                        risk: UnderlyingRiskProvider | None) -> list[str]:
    """Reasons a naked call on ``underlying`` may not be opened or held to ``horizon_end``.

    ``horizon_end`` is the last date the position could still be open (entry: the planned exit
    date, bounded by expiry; holding: today plus the buffer). An earnings date inside
    ``[asof date, horizon_end + buffer]`` blocks it. Each reason starts with its category
    (``earnings:``, ``squeeze:`` or ``unknown_data:``); see ``blocker_kind``.
    """
    reasons: list[str] = []
    if risk is None:
        return ["unknown_data: no earnings / short-interest data source injected; unknown blocks a naked call"]
    today = asof.tz_convert(NY).date() if asof.tzinfo else asof.date()
    if policy.block_through_earnings:
        info = risk.next_earnings(underlying, asof)
        if info is None:
            if policy.require_known_data:
                reasons.append(f"unknown_data: {underlying}: earnings date unknown")
        elif info.next_date is not None:
            limit = horizon_end + dt.timedelta(days=policy.earnings_buffer_days)
            if today <= info.next_date <= limit:
                reasons.append(f"earnings: {underlying}: earnings on {info.next_date} fall inside the holding window")
    si = risk.short_info(underlying, asof)
    if si is None or (si.short_interest_pct_float is None and si.days_to_cover is None):
        if policy.require_known_data:
            reasons.append(f"unknown_data: {underlying}: short-interest data unknown")
    else:
        if si.short_interest_pct_float is not None and si.short_interest_pct_float > policy.max_short_interest_pct_float:
            reasons.append(f"squeeze: {underlying}: short interest {si.short_interest_pct_float:.0f}% of float "
                           f"(limit {policy.max_short_interest_pct_float:.0f}%): squeeze risk")
        if si.days_to_cover is not None and si.days_to_cover > policy.max_days_to_cover:
            reasons.append(f"squeeze: {underlying}: {si.days_to_cover:.1f} days to cover (limit "
                           f"{policy.max_days_to_cover:.0f}): squeeze risk")
    return reasons
