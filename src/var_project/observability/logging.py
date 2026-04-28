from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from typing import Any

from var_project.observability.context import get_account_id, get_action, get_request_id, get_run_id


def _string_or_dash(value: Any) -> str:
    if value in {None, "", "null"}:
        return "-"
    return str(value)


class CorrelationContextFilter(logging.Filter):
    """Inject correlation identifiers into log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        request_id = getattr(record, "request_id", None) or get_request_id()
        run_id = getattr(record, "run_id", None) or get_run_id()
        account_id = getattr(record, "account_id", None) or get_account_id()
        action = getattr(record, "action", None) or get_action()

        record.request_id = _string_or_dash(request_id)
        record.run_id = _string_or_dash(run_id)
        record.account_id = _string_or_dash(account_id)
        record.action = _string_or_dash(action)
        return True


class JsonLogFormatter(logging.Formatter):
    """Optional JSON log formatter for machine parsing."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "level": str(record.levelname),
            "logger": str(record.name),
            "message": str(record.getMessage()),
            "request_id": _string_or_dash(getattr(record, "request_id", None)),
            "run_id": _string_or_dash(getattr(record, "run_id", None)),
            "account_id": _string_or_dash(getattr(record, "account_id", None)),
            "action": _string_or_dash(getattr(record, "action", None)),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=True)

