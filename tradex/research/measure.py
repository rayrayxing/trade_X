"""Measured hit rates: from the research loop's real-data walk-forward evidence into each spec's ``stats:``.

Paper and live vote with a strategy's measured win rate and with nothing else (no default). The number is the
Wilson lower bound of the out-of-sample win rate, so a thin sample reads low. It is written for every strategy
that has walk-forward evidence, passed or rejected: it is a measurement, not an approval, and ``stats`` is not
part of the spec's content hash, so nothing re-validates.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

MIN_TRADES = 30                      # out-of-sample trades below this are too few to size on
_STATS_BLOCK = re.compile(r"^stats:[^\n]*\n(?:[ \t]+[^\n]*\n|\n)*", re.MULTILINE)


def hit_rate_from_evidence(data: dict[str, Any]) -> dict[str, Any] | None:
    """Stats for a walk-forward evidence row, or None when it carries no usable win rate."""
    lower, n = data.get("win_rate_lower"), int(data.get("oos_trades") or 0)
    if lower is None or n < MIN_TRADES or not 0.0 < float(lower) < 1.0:
        return None
    return {"hit_rate": round(float(lower), 4), "hit_rate_trades": n,
            "hit_rate_win_rate": round(float(data.get("win_rate") or 0.0), 4),
            "hit_rate_source": f"walk_forward {data.get('data_key', '')}".strip()}


def write_stats(path: str | Path, stats: dict[str, Any]) -> bool:
    """Replace the top-level ``stats:`` block of a spec file (adding it when absent) and leave every other
    line alone. Refuses an edit that would change anything else. Returns whether the file changed."""
    p = Path(path)
    text = p.read_text()
    before = yaml.safe_load(text)
    block = "stats:\n" + "".join(f"  {ln}\n" for ln in yaml.safe_dump(stats, sort_keys=False).splitlines())
    new = _STATS_BLOCK.sub(lambda _m: block, text, count=1) if _STATS_BLOCK.search(text) \
        else text.rstrip("\n") + "\n" + block
    after = yaml.safe_load(new)
    if {k: v for k, v in after.items() if k != "stats"} != {k: v for k, v in before.items() if k != "stats"} \
            or after.get("stats") != stats:
        raise ValueError(f"{p}: writing stats would alter something else; edit refused")
    if new == text:
        return False
    tmp = p.with_suffix(".yaml.tmp")
    tmp.write_text(new)
    tmp.replace(p)
    return True


def measure(state, specs) -> dict[str, str]:
    """Write the measured hit rate into every spec file that has walk-forward evidence. ``state`` is a LoopState,
    ``specs`` a spec store. Returns {strategy id: what happened}."""
    out: dict[str, str] = {}
    for rec in specs.list_specs():
        ev = state.latest_evidence(rec.spec_id, rec.version, "walk_forward")
        if ev is None:
            out[rec.spec_id] = "no walk-forward evidence"
            continue
        stats = hit_rate_from_evidence(ev["data"])
        if stats is None:
            out[rec.spec_id] = (f"no usable win rate (needs {MIN_TRADES}+ out-of-sample trades and the win_rate_lower "
                                "field: re-run the walk_forward stage)")
            continue
        changed = write_stats(rec.path, stats)
        out[rec.spec_id] = f"hit_rate {stats['hit_rate']:.1%} from {stats['hit_rate_trades']} trades" + \
                           ("" if changed else " (unchanged)")
    return out
