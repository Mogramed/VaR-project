from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class LiveLoopConfig:
    interval_seconds: int = 30
    once: bool = False
    max_consecutive_errors: int = 5
    error_sleep_seconds: int = 5


def run_live_loop(step_fn: Callable[[], None], cfg: LiveLoopConfig) -> None:
    """
    step_fn: fetch latest data -> compute risk -> print/save
    """
    consecutive_errors = 0

    while True:
        try:
            step_fn()
            consecutive_errors = 0
        except KeyboardInterrupt:
            print("\n[LIVE] Stopped by user (Ctrl+C).")
            return
        except Exception as e:
            consecutive_errors += 1
            print(f"[LIVE][ERROR] {type(e).__name__}: {e}")
            if consecutive_errors >= cfg.max_consecutive_errors:
                print("[LIVE] Too many consecutive errors -> stopping.")
                return
            time.sleep(cfg.error_sleep_seconds)

        if cfg.once:
            return

        time.sleep(cfg.interval_seconds)
