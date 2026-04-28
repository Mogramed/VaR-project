from __future__ import annotations

from datetime import timezone

import pandas as pd

from var_project.storage.serialization import coerce_datetime


def test_coerce_datetime_returns_none_for_nat() -> None:
    assert coerce_datetime(pd.NaT) is None


def test_coerce_datetime_keeps_utc_for_valid_timestamp() -> None:
    result = coerce_datetime(pd.Timestamp("2026-04-23T10:15:00+00:00"))
    assert result is not None
    assert result.tzinfo is not None
    assert result.tzinfo == timezone.utc
