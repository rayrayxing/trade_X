import json
import sys
import types

import pandas as pd
import pytest
from tradex.data.opend_check import collect


@pytest.fixture(autouse=True)
def _moomoo_sdk(monkeypatch):
    """CI installs only `.[dev]`, so the moomoo SDK is absent there; `collect` only needs its RET_OK constant."""
    try:
        import moomoo  # noqa: F401
    except ImportError:
        fake = types.ModuleType("moomoo")
        fake.RET_OK = 0
        monkeypatch.setitem(sys.modules, "moomoo", fake)


class Q:
    def get_global_state(self): return 0, {"program_status_type": "READY", "trd_logined": True, "qot_logined": True, "server_ver": "1"}
    def get_user_info(self): return 0, {"api_level": "N/A", "sub_quota": 300, "history_kl_quota": 300, "user_id": 99, "nick_name": "x"}
    def query_subscription(self): return 0, {"total_used": 1, "own_used": 0, "remain": 299, "option_used_quota": 0, "option_remain_quota": 60}
    def get_history_kl_quota(self, get_detail=False): return 0, (0, 300, [])


class T:
    def get_acc_list(self): return 0, pd.DataFrame({"acc_id": [111, 222], "trd_env": ["REAL", "SIMULATE"]})


def test_collect_has_no_identifiers():
    rep = collect(Q(), T())
    s = json.dumps(rep)
    assert rep["accounts"] == {"simulate": 1, "real_ignored": 1} and rep["history_kline_quota"]["remain"] == 300
    assert "111" not in s and "222" not in s and "user_id" not in s and "nick_name" not in s
