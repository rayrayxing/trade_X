import ast
from pathlib import Path

import pandas as pd
import pytest

from tradex.data.guard import RealDataMissing, SyntheticDataRefused, require_present, require_real_data
from tradex.data.providers import CachedProvider, CsvProvider, OandaProvider


def test_paper_live_refuse_stand_ins(tmp_path):
    for mode in ("paper", "live"):
        with pytest.raises(SyntheticDataRefused):
            require_real_data(mode, CsvProvider(tmp_path))
        with pytest.raises(SyntheticDataRefused):
            require_real_data(mode, "synthetic")
        with pytest.raises(SyntheticDataRefused):
            require_real_data(mode, CachedProvider(OandaProvider(), tmp_path, "x"))
        with pytest.raises(RealDataMissing):
            require_real_data(mode, None)
        require_real_data(mode, OandaProvider())
    for mode in ("backtest", "replay"):
        require_real_data(mode, CsvProvider(tmp_path))
    with pytest.raises(ValueError):
        require_real_data("demo", OandaProvider())


def test_require_present():
    assert require_present("paper", 1.2, "EUR_USD") == 1.2
    for bad in (None, float("nan"), pd.DataFrame()):
        with pytest.raises(RealDataMissing):
            require_present("live", bad, "EUR_USD")
        require_present("backtest", bad, "EUR_USD")


def test_no_module_outside_tests_imports_synthetic():
    root = Path(__file__).resolve().parents[1]
    offenders = []
    for py in [*(root / "tradex").rglob("*.py"), *(root / "tools").rglob("*.py")]:
        if py.name == "synthetic.py" and py.parent.name == "data":
            continue
        for node in ast.walk(ast.parse(py.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = "." * node.level + (node.module or "")
                names = [base] + [f"{base}.{a.name}" for a in node.names]
            if any(n.endswith("data.synthetic") or n.endswith(".synthetic") or n == "synthetic" for n in names):
                offenders.append(f"{py.relative_to(root)}:{node.lineno}")
    assert not offenders, f"non-test modules import tradex.data.synthetic: {offenders}"
