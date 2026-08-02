"""Isolated structured logging without global logger configuration."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from io import TextIOBase
from typing import Any


class JsonFormatter(logging.Formatter):
    """Format a stable JSON object from a standard-library log record."""

    def format(self, record: logging.LogRecord) -> str:
        value: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        event = getattr(record, "event", None)
        fields = getattr(record, "fields", None)
        if event is not None:
            value["event"] = str(event)
        if fields is not None:
            if not isinstance(fields, dict) or not all(isinstance(key, str) for key in fields):
                raise TypeError("structured log fields must be a string-keyed mapping")
            value["fields"] = fields
        if record.exc_info:
            value["exception"] = self.formatException(record.exc_info)
        return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def make_structured_logger(
    name: str,
    *,
    stream: TextIOBase,
    level: int = logging.INFO,
) -> logging.Logger:
    """Create an isolated logger without mutating the process-wide registry or root logger."""

    logger = logging.Logger(name=name, level=level)
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.propagate = False
    return logger
