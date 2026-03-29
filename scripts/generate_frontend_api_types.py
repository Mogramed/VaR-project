from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from var_project.api.export_openapi import build_openapi_schema


def _run_openapi_typescript(*, schema_path: Path, output_path: Path, cwd: Path) -> None:
    binary_name = "openapi-typescript.cmd" if sys.platform.startswith("win") else "openapi-typescript"
    local_bin = cwd / "node_modules" / ".bin" / binary_name
    if local_bin.exists():
        subprocess.run(
            [str(local_bin), str(schema_path), "--output", str(output_path)],
            cwd=cwd,
            check=True,
        )
        return

    npm_command = "npm.cmd" if sys.platform.startswith("win") else "npm"
    with tempfile.TemporaryDirectory(prefix="var_project_openapi_tool_") as tool_dir:
        tool_root = Path(tool_dir)
        (tool_root / "package.json").write_text('{"private": true}\n', encoding="utf-8")
        subprocess.run(
            [
                npm_command,
                "install",
                "--no-package-lock",
                "--no-save",
                "openapi-typescript@7.13.0",
            ],
            cwd=tool_root,
            check=True,
        )
        tool_bin = tool_root / "node_modules" / ".bin" / binary_name
        subprocess.run(
            [str(tool_bin), str(schema_path), "--output", str(output_path)],
            cwd=cwd,
            check=True,
        )


def _write_public_types(*, schema: dict[str, object], output_path: Path) -> None:
    schema_names = sorted(((schema.get("components") or {}).get("schemas") or {}).keys())
    lines = [
        "// Generated from the backend OpenAPI schema. Do not edit manually.",
        "",
        'import type { components } from "./generated-schema";',
        "",
        'export type ApiSchemas = components["schemas"];',
        "",
    ]
    for name in schema_names:
        lines.append(f'export type {name} = ApiSchemas["{name}"];')
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    repo_root = REPO_ROOT
    frontend_root = repo_root / "frontend"
    generated_schema_path = frontend_root / "src" / "lib" / "api" / "generated-schema.ts"
    public_types_path = frontend_root / "src" / "lib" / "api" / "types.ts"
    schema = build_openapi_schema(repo_root=repo_root)

    with tempfile.TemporaryDirectory(prefix="var_project_openapi_") as tmpdir:
        schema_path = Path(tmpdir) / "openapi.json"
        schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
        _run_openapi_typescript(schema_path=schema_path, output_path=generated_schema_path, cwd=frontend_root)
    _write_public_types(schema=schema, output_path=public_types_path)


if __name__ == "__main__":
    main()
