from __future__ import annotations

import argparse
import inspect
import logging
import logging.config
from pathlib import Path
from typing import Any

import yaml

from var_project.bootstrap import seed_demo_environment
from var_project.core.settings import find_repo_root
from var_project.storage import upgrade_database


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def setup_logging(root: Path) -> None:
    cfg_path = root / "config" / "logging.yaml"
    if not cfg_path.exists():
        logging.basicConfig(level=logging.INFO)
        return

    cfg = load_yaml(cfg_path)
    handlers = cfg.get("handlers", {})
    file_h = handlers.get("file")
    if isinstance(file_h, dict) and "filename" in file_h:
        rel = Path(str(file_h["filename"]))
        abs_path = (root / rel).resolve()
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        file_h["filename"] = str(abs_path)

    logging.config.dictConfig(cfg)


def main() -> None:
    root = find_repo_root()
    setup_logging(root)
    log = logging.getLogger("var_project.cli")

    parser = argparse.ArgumentParser(prog="var-project")
    sub = parser.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("api")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--workers", type=int, default=1)
    s.add_argument("--reload", action="store_true")
    s.add_argument("--timeout-worker-healthcheck", type=int, default=30)

    s = sub.add_parser("mt5-agent")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8010)
    s.add_argument("--reload", action="store_true")

    s = sub.add_parser("worker")
    s.add_argument("--once", action="store_true")

    sub.add_parser("operator-worker")

    s = sub.add_parser("seed-demo")
    s.add_argument("--portfolio-slug", default=None)

    s = sub.add_parser("demo-smoke")
    s.add_argument("--base-url", default="http://127.0.0.1:8000")
    s.add_argument("--portfolio-slug", default=None)
    s.add_argument("--timeout-seconds", type=float, default=5.0)
    s.add_argument("--max-wait-ms", type=int, default=1200)
    s.add_argument("--strict", action="store_true")

    s = sub.add_parser("db")
    db_sub = s.add_subparsers(dest="db_cmd", required=True)
    s = db_sub.add_parser("upgrade")
    s.add_argument("--revision", default="head")

    args = parser.parse_args()

    if args.cmd == "api":
        import uvicorn

        workers = max(int(getattr(args, "workers", 1) or 1), 1)
        if bool(args.reload):
            workers = 1
        timeout_worker_healthcheck = max(
            int(getattr(args, "timeout_worker_healthcheck", 30) or 30),
            1,
        )
        uvicorn_kwargs: dict[str, Any] = {
            "factory": True,
            "host": str(args.host),
            "port": int(args.port),
            "workers": workers,
            "reload": bool(args.reload),
        }
        if "timeout_worker_healthcheck" in inspect.signature(uvicorn.run).parameters:
            uvicorn_kwargs["timeout_worker_healthcheck"] = timeout_worker_healthcheck
        uvicorn.run(
            "var_project.api.app:create_app",
            **uvicorn_kwargs,
        )
        raise SystemExit(0)

    if args.cmd == "mt5-agent":
        import uvicorn

        from var_project.execution.mt5_agent import create_mt5_agent_app

        uvicorn.run(
            create_mt5_agent_app(root),
            host=str(args.host),
            port=int(args.port),
            reload=bool(args.reload),
        )
        raise SystemExit(0)

    if args.cmd == "worker":
        from var_project.jobs import JobRunner

        runner = JobRunner(root, bootstrap_storage=False)
        runner.run_forever(once=bool(args.once))
        raise SystemExit(0)

    if args.cmd == "operator-worker":
        from var_project.jobs.operator_queue import run_operator_worker

        run_operator_worker()
        raise SystemExit(0)

    if args.cmd == "seed-demo":
        result = seed_demo_environment(root, portfolio_slug=args.portfolio_slug)
        log.info(
            "seed-demo complete for %s portfolio(s) against %s",
            result["portfolio_count"],
            result["database_url"],
        )
        raise SystemExit(0)

    if args.cmd == "demo-smoke":
        from var_project.demo_smoke import format_demo_smoke_report, run_demo_smoke

        result = run_demo_smoke(
            base_url=str(args.base_url),
            portfolio_slug=args.portfolio_slug,
            timeout_seconds=float(args.timeout_seconds),
            max_wait_ms=int(args.max_wait_ms),
            strict=bool(args.strict),
        )
        for line in format_demo_smoke_report(result):
            print(line)
        raise SystemExit(0 if bool(result.get("ok", False)) else 1)

    if args.cmd == "db" and args.db_cmd == "upgrade":
        result = upgrade_database(root, revision=str(args.revision))
        log.info("database upgraded to %s using %s", result["revision"], result["database_url"])
        raise SystemExit(0)

    raise SystemExit(2)


if __name__ == "__main__":
    main()
