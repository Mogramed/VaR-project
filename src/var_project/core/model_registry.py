from __future__ import annotations

import re
from typing import Iterable


CANONICAL_MODEL_ORDER: tuple[str, ...] = ("hist", "param", "mc", "ewma", "garch", "fhs")
_MODEL_COLUMN_RE = re.compile(r"^(?:var|es|exc)_(?P<model>[a-z0-9]+?)(?:_a\d+_h\d+)?$", re.IGNORECASE)


def ordered_model_names(names: Iterable[str]) -> list[str]:
    order_index = {name: idx for idx, name in enumerate(CANONICAL_MODEL_ORDER)}
    unique = {str(name).strip().lower() for name in names if str(name).strip()}
    return sorted(unique, key=lambda name: (order_index.get(name, len(CANONICAL_MODEL_ORDER)), name))


def infer_model_names_from_columns(columns: Iterable[str]) -> list[str]:
    models: set[str] = set()
    for column in columns:
        match = _MODEL_COLUMN_RE.match(str(column).strip().lower())
        if match is not None:
            models.add(match.group("model"))
    return ordered_model_names(models)
