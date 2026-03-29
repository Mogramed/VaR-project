"""
VaR-project: mini risk toolkit (VaR/ES + backtests) with MT5 ingestion.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("var-project")
except PackageNotFoundError:  # running from source without install
    __version__ = "0.0.0"

__all__ = ["__version__"]
