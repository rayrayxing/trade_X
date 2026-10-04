"""`tradex opend-check`: read-only report on the local OpenD gateway.

Reads quota and entitlement info only. Never places, modifies or cancels an order.
Account IDs, user IDs and nicknames are never printed. What SIMULATE accepts is taken
from moomoo's documentation (not probed), because the only way to probe is to place orders.
"""
from __future__ import annotations

HOST, PORT = "127.0.0.1", 11111

# Documentation: "Paper trade only supports limit orders (NORMAL) and market orders (MARKET)."
# https://openapi.moomoo.com/moomoo-api-doc/en/trade/trade.html (checked 2026-10-04).
# US stock paper trading also excludes pre/post-market and overnight sessions.
SIMULATE_DOCUMENTED = {
    "US stock": {"market": True, "limit": True, "stop": False, "stop-limit": False, "trailing stop": False},
    "US option": {"market": True, "limit": True, "stop": False, "stop-limit": False, "trailing stop": False,
                  "combo": "not documented for SIMULATE; place_combo_order exists but support is unconfirmed"},
}


def collect(quote, trade=None) -> dict:
    """Gather the report from open contexts (injectable for tests). Returns a dict free of identifiers."""
    try:
        from moomoo import RET_OK
    except ImportError:          # moomoo-api is an optional extra; its success code is 0
        RET_OK = 0
    rep: dict = {}
    ret, st = quote.get_global_state()
    rep["opend"] = {"ready": ret == RET_OK and st.get("program_status_type") == "READY",
                    "trade_logged_in": bool(st.get("trd_logined")) if ret == RET_OK else None,
                    "quote_logged_in": bool(st.get("qot_logined")) if ret == RET_OK else None,
                    "server_ver": st.get("server_ver") if ret == RET_OK else None}
    ret, ui = quote.get_user_info()
    if ret == RET_OK:
        rep["quota_tier"] = {k: ui.get(k) for k in ("api_level", "us_qot_right", "us_option_qot_right", "sub_quota",
                                                    "history_kl_quota")}
    ret, sub = quote.query_subscription()
    if ret == RET_OK:
        rep["subscription"] = {k: sub.get(k) for k in ("total_used", "own_used", "remain", "option_used_quota",
                                                       "option_remain_quota")}
    ret, kl = quote.get_history_kl_quota(get_detail=False)
    if ret == RET_OK:
        rep["history_kline_quota"] = {"used": kl[0], "remain": kl[1]}
    if trade is not None:
        ret, df = trade.get_acc_list()
        if ret == RET_OK:
            envs = df["trd_env"].astype(str).str.upper()
            rep["accounts"] = {"simulate": int((envs == "SIMULATE").sum()), "real_ignored": int((envs == "REAL").sum())}
    rep["simulate_order_types"] = SIMULATE_DOCUMENTED
    return rep


def run(out=print) -> int:
    from moomoo import OpenQuoteContext, OpenSecTradeContext, TrdMarket
    q = OpenQuoteContext(HOST, PORT)
    t = None
    try:
        try:
            t = OpenSecTradeContext(filter_trdmarket=TrdMarket.NONE, host=HOST, port=PORT)
        except Exception:
            t = None
        rep = collect(q, t)
    finally:
        q.close()
        if t is not None:
            t.close()
    import json
    out(json.dumps(rep, indent=2))
    return 0 if rep["opend"]["ready"] else 2
