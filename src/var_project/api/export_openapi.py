from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from var_project.api.app import create_app
from var_project.core.settings import find_repo_root


def _normalize_validation_error_schema(schema: dict[str, Any]) -> None:
    components = schema.get("components")
    if not isinstance(components, dict):
        return
    schemas = components.get("schemas")
    if not isinstance(schemas, dict):
        return
    validation_error = schemas.get("ValidationError")
    if not isinstance(validation_error, dict):
        return
    properties = validation_error.get("properties")
    if not isinstance(properties, dict):
        return

    # FastAPI/Pydantic versions can inject optional `input` / `ctx` fields here.
    # They are runtime diagnostics only and create noisy non-functional diffs
    # for generated frontend contracts across environments.
    properties.pop("input", None)
    properties.pop("ctx", None)

    required = validation_error.get("required")
    if isinstance(required, list):
        validation_error["required"] = [item for item in required if item not in {"input", "ctx"}]


def build_openapi_schema(repo_root: Path | None = None) -> dict[str, Any]:
    app = create_app(repo_root=(repo_root or find_repo_root()))
    schema = app.openapi()
    _normalize_validation_error_schema(schema)
    return schema


def export_openapi(output: Path | None = None, *, repo_root: Path | None = None) -> dict[str, Any]:
    schema = build_openapi_schema(repo_root=repo_root)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    return schema


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the FastAPI OpenAPI schema.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output JSON path.")
    parser.add_argument("--repo-root", type=Path, default=None, help="Optional repository root.")
    args = parser.parse_args()

    schema = export_openapi(args.output, repo_root=args.repo_root)
    if args.output is None:
        print(json.dumps(schema, indent=2))


if __name__ == "__main__":
    main()
