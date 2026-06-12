"""Structured logging configuration and JSONL trajectory writer."""

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any
from uuid import UUID

from config import settings

_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[\s-]?)?0\d{8,9}\b")


def _mask_pii(value: Any) -> Any:
    """Recursively mask emails and phone numbers in strings, dicts, and lists.

    Args:
        value: The value to mask. Strings have emails/phone numbers replaced
            with `***`; dicts and lists are processed recursively; all other
            types are returned unchanged.

    Returns:
        Any: The value with any PII masked.
    """
    if isinstance(value, str):
        masked = _EMAIL_RE.sub("***@***", value)
        masked = _PHONE_RE.sub("***", masked)
        return masked
    if isinstance(value, dict):
        return {key: _mask_pii(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_mask_pii(item) for item in value]
    return value


class JsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        """Render a log record as a JSON string.

        Args:
            record: The log record to format.

        Returns:
            str: A JSON-encoded string representing the log record.
        """
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": _mask_pii(record.getMessage()),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging() -> None:
    """Configure the root logger with a JSON formatter and stdout handler.

    Idempotent: calling this multiple times does not duplicate handlers.
    """
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root_logger.addHandler(handler)
    root_logger.setLevel(settings.log_level.upper())


def write_trajectory(run_id: UUID, trajectory: dict[str, Any]) -> None:
    """Write a triage run's trajectory to a JSONL file.

    Args:
        run_id: Unique identifier of the triage run.
        trajectory: Dictionary describing the run, including the
            `reasoning_trace` and final outputs.

    Raises:
        OSError: If the trajectory file cannot be written. The error is
            logged but not re-raised.
    """
    logger = logging.getLogger(__name__)
    trajectories_dir = Path(settings.trajectories_dir)

    try:
        trajectories_dir.mkdir(parents=True, exist_ok=True)
        file_path = trajectories_dir / f"{run_id}.jsonl"
        masked_trajectory = _mask_pii(trajectory)
        with open(file_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(masked_trajectory, ensure_ascii=False, default=str) + "\n")
    except OSError as exc:
        logger.error("Failed to write trajectory for run %s: %s", run_id, exc)
