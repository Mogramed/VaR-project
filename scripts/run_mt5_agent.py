from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn


def _bootstrap_src_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return root


ROOT = _bootstrap_src_path()

from var_project.core.settings import find_repo_root  # noqa: E402
from var_project.execution.mt5_agent import create_mt5_agent_app  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Windows-side MT5 agent for the VaR Risk Desk.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    root = find_repo_root(ROOT)
    uvicorn.run(create_mt5_agent_app(root), host=str(args.host), port=int(args.port), reload=bool(args.reload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
