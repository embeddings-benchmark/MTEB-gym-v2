"""Reproducibility helpers for MTEB Gym experiment results.

This module intentionally contains no experiment logic.  It only turns a run's
configuration into stable JSON metadata and a deterministic short config hash,
and records the exact git revision that produced the result.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def _jsonable(value: Any) -> Any:
    """Convert common config values into stable JSON-serializable objects."""
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {k: _jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def canonical_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a recursively JSON-safe, deterministically ordered config."""
    return _jsonable(config)


def config_hash(config: dict[str, Any], length: int = 8) -> str:
    """Stable short SHA-256 hash of the complete experiment configuration."""
    payload = json.dumps(
        canonical_config(config),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def git_revision(repo: str | Path | None = None) -> str | None:
    """Return the exact git commit for the checkout, or None outside git."""
    cwd = Path(repo) if repo is not None else Path(__file__).resolve().parents[1]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
