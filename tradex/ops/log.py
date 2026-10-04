"""Structured JSON logs with size-capped rotation and a secret scrubber.

One JSON object per line. The scrubber runs on the final serialized line, so message
text, ``extra`` fields and exception tracebacks are all covered. It removes (a) every
value currently in the secrets store and (b) token-like patterns, so a secret that was
never registered (a rotated key, a pasted header) is still caught.
"""
from __future__ import annotations

import json
import logging
import re
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path

from tradex import secrets

REDACTED = "[REDACTED]"
MAX_TOTAL_BYTES = 800 * 1024 * 1024         # hard design cap, below the 1 GB budget
_FILE_BYTES = 50 * 1024 * 1024
_BACKUPS = 15                                # 16 files x 50 MB = 800 MB

_PATTERNS = [
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"\b[0-9a-f]{32}-[0-9a-f]{32}\b"),                     # Oanda token
    re.compile(r"\b\d{6,12}:[A-Za-z0-9_-]{30,}\b"),                   # Telegram bot token
    re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_-]{16,}\b"),               # OpenAI / Anthropic style
    re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b"),                        # Google API key
    re.compile(r"(?i)((?:api[_-]?key|apikey|secret|token|password|passwd|authorization)[\"']?\s*[:=]\s*[\"']?)[^\s\"',&}]{6,}"),
]
_MIN_SECRET_LEN = 6
_TTL_S = 60.0


class Scrubber:
    def __init__(self, extra: list[str] | None = None, use_store: bool = True):
        self._extra = [v for v in (extra or []) if v]
        self._use_store = use_store
        self._cache: list[str] = []
        self._at = -1e18

    def register(self, value: str) -> None:
        if value:
            self._extra.append(value)

    def refresh(self) -> None:
        vals = list(self._extra)
        if self._use_store:
            try:
                vals += secrets.known_values()
            except Exception:
                pass
        self._cache = sorted({v for v in vals if len(v) >= _MIN_SECRET_LEN}, key=len, reverse=True)
        self._at = time.monotonic()

    def scrub(self, text: str) -> str:
        if time.monotonic() - self._at > _TTL_S:
            self.refresh()
        for v in self._cache:
            text = text.replace(v, REDACTED)
        for pat in _PATTERNS:
            if pat.groups:
                text = pat.sub(lambda m: m.group(1) + REDACTED, text)
            else:
                text = pat.sub(REDACTED, text)
        return text


class JsonFormatter(logging.Formatter):
    _SKIP = set(vars(logging.LogRecord("", 0, "", 0, "", (), None))) | {"message", "asctime", "taskName"}

    def __init__(self, scrubber: Scrubber):
        super().__init__()
        self.scrubber = scrubber

    def format(self, record: logging.LogRecord) -> str:
        d = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
             "level": record.levelname, "logger": record.name, "msg": record.getMessage()}
        for k, v in record.__dict__.items():
            if k not in self._SKIP and not k.startswith("_"):
                d[k] = v
        if record.exc_info:
            d["exc"] = self.formatException(record.exc_info)
        return self.scrubber.scrub(json.dumps(d, default=str, ensure_ascii=False))


def setup_logging(path: str | Path = "data/logs/tradex.jsonl", level: int = logging.INFO,
                  scrubber: Scrubber | None = None, max_bytes: int = _FILE_BYTES,
                  backups: int = _BACKUPS, console: bool = False) -> Scrubber:
    """Attach a rotating JSON file handler to the ``tradex`` logger. Total size <= max_bytes*(backups+1)."""
    if max_bytes * (backups + 1) >= 1024**3:
        raise ValueError("log rotation settings would allow 1 GB or more")
    scrubber = scrubber or Scrubber()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger("tradex")
    root.setLevel(level)
    for h in list(root.handlers):
        if getattr(h, "_tradex", False):
            root.removeHandler(h)
            h.close()
    handlers: list[logging.Handler] = [RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backups, encoding="utf-8")]
    if console:
        handlers.append(logging.StreamHandler())
    for h in handlers:
        h.setFormatter(JsonFormatter(scrubber))
        h._tradex = True  # type: ignore[attr-defined]
        root.addHandler(h)
    return scrubber
