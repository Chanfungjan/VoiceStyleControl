"""Compatibility exports for shared ark/scp utilities."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.ark import (
    BINARY_MARKER as B_,
    INT32_MARKER as FOUR_,
    ZERO as ZERO_,
    ArkWriter,
    pack_header,
    recover_write_offset,
)

__all__ = [
    "ArkWriter",
    "B_",
    "FOUR_",
    "ZERO_",
    "pack_header",
    "recover_write_offset",
]
