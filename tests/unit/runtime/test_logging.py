from __future__ import annotations

import io
import json
import logging

import pytest

from nanopt.runtime.logging import JsonFormatter, make_structured_logger


def test_structured_logger_emits_stable_fields_without_touching_root() -> None:
    stream = io.StringIO()
    root_handlers = tuple(logging.root.handlers)
    logger = make_structured_logger("nanopt.test", stream=stream)
    logger.info("created", extra={"event": "run_created", "fields": {"run_id": "fixture"}})
    value = json.loads(stream.getvalue())
    assert value["level"] == "info"
    assert value["event"] == "run_created"
    assert value["fields"] == {"run_id": "fixture"}
    assert tuple(logging.root.handlers) == root_handlers


def test_structured_logger_rejects_non_mapping_fields() -> None:
    record = logging.makeLogRecord(
        {"name": "nanopt.test", "levelno": logging.INFO, "levelname": "INFO", "msg": "bad"}
    )
    record.fields = ["not", "a", "mapping"]
    with pytest.raises(TypeError, match="string-keyed mapping"):
        JsonFormatter().format(record)
