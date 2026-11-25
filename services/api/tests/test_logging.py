import json
import logging

from qa_api.logging_config import JsonFormatter


def test_json_formatter_includes_extra_fields() -> None:
    record = logging.LogRecord("t", logging.INFO, __file__, 1, "hello %s", ("world",), None)
    record.doc_id = "abc"
    out = json.loads(JsonFormatter().format(record))
    assert out["message"] == "hello world"
    assert out["level"] == "INFO"
    assert out["doc_id"] == "abc"
