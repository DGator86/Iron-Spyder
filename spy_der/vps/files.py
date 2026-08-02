"""Atomic JSON writers for VPS state files.

A crash mid-write must leave the previous file intact, not a truncated one the
dashboard would then fail to parse. ``tempfile`` + ``os.replace`` is the usual
pattern; the temp file is created in the target directory so the rename stays
on one filesystem.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

__all__ = ["atomic_write_json"]


def atomic_write_json(path: str | Path, payload: dict[str, Any], *, mode: int = 0o644) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_path, mode)
        os.replace(tmp_path, target)
    except Exception:
        with contextlib.suppress(OSError):
            tmp_path.unlink(missing_ok=True)
        raise
    return target
