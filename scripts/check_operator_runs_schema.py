from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sqlalchemy import create_engine


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from var_project.core.settings import load_settings
from var_project.storage.schema_checks import validate_operator_runs_schema
from var_project.storage.settings import StorageSettings


def _resolve_database_url(repo_root: Path, explicit_url: str | None) -> str:
    if explicit_url:
        return str(explicit_url)
    raw_config = load_settings(repo_root) if (repo_root / "config" / "settings.yaml").exists() else {}
    return StorageSettings.from_root(repo_root, raw_config).database_url


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate operator_runs schema after Alembic migration.")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--database-url", default=None)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    database_url = _resolve_database_url(repo_root, args.database_url)
    engine = create_engine(database_url, future=True)
    try:
        issues = validate_operator_runs_schema(engine)
    finally:
        engine.dispose()

    if issues:
        print("operator_runs post-migration schema check failed:")
        for issue in issues:
            print(f"- {issue}")
        raise SystemExit(1)

    print("operator_runs post-migration schema check passed.")


if __name__ == "__main__":
    main()
