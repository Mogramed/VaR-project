from __future__ import annotations

import logging.config
from pathlib import Path

from var_project.cli import setup_logging


def _write_logging_config(root: Path) -> None:
    config_dir = root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "logging.yaml").write_text(
        """
version: 1
disable_existing_loggers: false

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    stream: ext://sys.stdout
  file:
    class: logging.handlers.RotatingFileHandler
    level: INFO
    filename: reports/logs/test.log
    maxBytes: 1024
    backupCount: 1

root:
  level: INFO
  handlers: [console, file]

loggers:
  var_project:
    level: INFO
    handlers: [console, file]
    propagate: false
""".strip(),
        encoding="utf-8",
    )


def test_setup_logging_applies_level_overrides_and_normalizes_file_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_logging_config(tmp_path)
    captured: dict[str, object] = {}

    monkeypatch.setenv("VAR_PROJECT_LOG_LEVEL", "warning")
    monkeypatch.setenv("VAR_PROJECT_LOG_FILE_LEVEL", "debug")

    def _fake_dict_config(cfg):
        captured["cfg"] = cfg

    monkeypatch.setattr(logging.config, "dictConfig", _fake_dict_config)

    setup_logging(tmp_path)

    cfg = captured["cfg"]
    assert isinstance(cfg, dict)
    assert cfg["root"]["level"] == "WARNING"
    assert cfg["loggers"]["var_project"]["level"] == "WARNING"
    assert cfg["handlers"]["console"]["level"] == "WARNING"
    assert cfg["handlers"]["file"]["level"] == "DEBUG"
    assert Path(str(cfg["handlers"]["file"]["filename"])).is_absolute()


def test_setup_logging_ignores_invalid_level_override(tmp_path: Path, monkeypatch) -> None:
    _write_logging_config(tmp_path)
    captured: dict[str, object] = {}

    monkeypatch.setenv("VAR_PROJECT_LOG_LEVEL", "verbose-mode")

    def _fake_dict_config(cfg):
        captured["cfg"] = cfg

    monkeypatch.setattr(logging.config, "dictConfig", _fake_dict_config)

    setup_logging(tmp_path)

    cfg = captured["cfg"]
    assert isinstance(cfg, dict)
    assert cfg["root"]["level"] == "INFO"
    assert cfg["loggers"]["var_project"]["level"] == "INFO"
    assert cfg["handlers"]["console"]["level"] == "INFO"
    assert cfg["handlers"]["file"]["level"] == "INFO"
