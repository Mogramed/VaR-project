from __future__ import annotations

import argparse
from dataclasses import replace
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from var_project.core.settings import load_settings
from var_project.storage.app_storage import AppStorage
from var_project.storage.settings import StorageSettings


def _resolve_settings(repo_root: Path, explicit_url: str | None) -> StorageSettings:
    raw_config = load_settings(repo_root) if (repo_root / "config" / "settings.yaml").exists() else {}
    settings = StorageSettings.from_root(repo_root, raw_config)
    if not explicit_url:
        return settings
    return replace(settings, database_url=str(explicit_url), database_path=None)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate critical DB schema and Alembic revision after migration."
    )
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--database-url", default=None)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    settings = _resolve_settings(repo_root, args.database_url)
    storage = AppStorage(settings, root=repo_root)
    report = storage.schema_status(strict_revision=True)
    try:
        storage.engine.dispose()
    except Exception:
        pass

    issues = list(report.get("issues") or [])
    if issues:
        print("database schema post-migration check failed:")
        print(f"- expected_revision: {report.get('expected_revision')}")
        print(f"- current_revision: {report.get('current_revision')}")
        for issue in issues:
            print(f"- {issue}")
        raise SystemExit(1)

    print(
        "database schema post-migration check passed "
        f"(revision={report.get('current_revision')})."
    )


if __name__ == "__main__":
    main()
