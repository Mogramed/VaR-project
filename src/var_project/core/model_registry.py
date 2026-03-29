from __future__ import annotations

from typing import Iterable


CANONICAL_MODEL_ORDER: tuple[str, ...] = ("hist", "param", "mc", "ewma", "garch", "fhs")


def ordered_model_names(names: Iterable[str]) -> list[str]:
    order_index = {name: idx for idx, name in enumerate(CANONICAL_MODEL_ORDER)}
    unique = {str(name).strip().lower() for name in names if str(name).strip()}
    return sorted(unique, key=lambda name: (order_index.get(name, len(CANONICAL_MODEL_ORDER)), name))


def infer_model_names_from_columns(columns: Iterable[str]) -> list[str]:
    models: set[str] = set()
    for column in columns:
        name = str(column)
        if name.startswith(("var_", "es_", "exc_")):
            models.add(name.split("_", 1)[1])
    return ordered_model_names(models)
