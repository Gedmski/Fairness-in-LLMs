"""Local import shim for running ``python -m fair_mia.cli`` without installation."""

from pathlib import Path

__version__ = "0.1.0"

_src_pkg = Path(__file__).resolve().parent.parent / "src" / "fair_mia"
if _src_pkg.exists():
    __path__.append(str(_src_pkg))  # type: ignore[name-defined]

