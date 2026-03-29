from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from var_project.api.app import create_app
from var_project.core.settings import find_repo_root


def build_openapi_schema(repo_root: Path | None = None) -> dict[str, Any]:
    app = create_app(repo_root=(repo_root or find_repo_root()))
    return app.openapi()


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
