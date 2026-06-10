"""Kaldi-style ark/scp writing utilities with resume support."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
from typing import BinaryIO, TextIO


ZERO = 0x0
BINARY_MARKER = 0x42
INT32_MARKER = 0x4
HEADER_FORMAT = "<3BI"


def pack_header(size: int) -> bytes:
    return struct.pack(HEADER_FORMAT, ZERO, BINARY_MARKER, INT32_MARKER, size)


def _record_end_offset(ark_path: Path, offset: int) -> int:
    with ark_path.open("rb") as stream:
        stream.seek(offset)
        raw_header = stream.read(struct.calcsize(HEADER_FORMAT))
        zero, marker, int_marker, size = struct.unpack(HEADER_FORMAT, raw_header)
        if (zero, marker, int_marker) != (ZERO, BINARY_MARKER, INT32_MARKER):
            raise ValueError(f"invalid ark header at {ark_path}:{offset}")
        return offset + len(raw_header) + int(size)


def recover_write_offset(ark_path: str | Path, scp_path: str | Path) -> int:
    """Recover the next write offset from the newest valid scp entry."""
    ark = Path(ark_path)
    scp = Path(scp_path)
    if not ark.exists() or not scp.exists():
        return 0

    lines = scp.read_text(encoding="utf-8").splitlines()
    for line in reversed(lines[-8:]):
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            return _record_end_offset(ark, int(parts[1]))
        except (OSError, struct.error, ValueError):
            continue
    return 0


@dataclass
class ArkWriter:
    """Append records to one ark/scp pair while preserving resume offsets."""

    ark_path: str | Path
    scp_path: str | Path

    def __post_init__(self) -> None:
        self.ark_path = Path(self.ark_path)
        self.scp_path = Path(self.scp_path)
        self.offset = recover_write_offset(self.ark_path, self.scp_path)
        self._ark_stream: BinaryIO | None = None
        self._scp_stream: TextIO | None = None

    def __enter__(self) -> "ArkWriter":
        self.ark_path.parent.mkdir(parents=True, exist_ok=True)
        self.scp_path.parent.mkdir(parents=True, exist_ok=True)
        self._ark_stream = self.ark_path.open("ab+")
        self._scp_stream = self.scp_path.open("a+", encoding="utf-8")
        self._ark_stream.seek(self.offset)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def write(self, key: str | int, payload: bytes) -> int:
        if self._ark_stream is None or self._scp_stream is None:
            raise RuntimeError("ArkWriter must be used as a context manager")

        start_offset = self.offset
        header = pack_header(len(payload))
        self._ark_stream.seek(start_offset)
        self._ark_stream.write(header)
        self._ark_stream.write(payload)
        self._scp_stream.write(f"{key}    {start_offset}\n")
        self.offset += len(header) + len(payload)
        return start_offset

    def flush(self) -> None:
        if self._ark_stream is not None:
            self._ark_stream.flush()
        if self._scp_stream is not None:
            self._scp_stream.flush()

    def close(self) -> None:
        self.flush()
        if self._ark_stream is not None:
            self._ark_stream.close()
            self._ark_stream = None
        if self._scp_stream is not None:
            self._scp_stream.close()
            self._scp_stream = None

