import urllib.request

from tradex.data import oanda


class _Sock:
    timeout = None

    def settimeout(self, s):
        self.timeout = s


class _Resp:
    def __init__(self):
        self.sock = _Sock()
        self.fp = type("FP", (), {"raw": type("Raw", (), {"_sock": self.sock})()})()

    def __iter__(self):
        return iter([b'{"type":"HEARTBEAT","time":"2026-10-06T16:00:00Z"}'])


def test_open_waits_longer_than_the_silence_window(monkeypatch):
    seen = {}
    resp = _Resp()

    def urlopen(req, timeout):
        seen["timeout"] = timeout
        return resp

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    ps = oanda.PriceStream(["EUR_USD"], account_id="101-000-TEST", token="t")
    list(ps._open())
    assert seen["timeout"] == 60.0                  # slow practice headers (~24 s) must not time out the open
    assert resp.sock.timeout == 7.0                 # after the headers, 7 s of silence means a dead stream
