"""Standard result metadata and paths for reproducible MTEB Gym runs."""

from __future__ import annotations

import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .repro import config_hash, git_revision


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def runtime_versions() -> dict[str, str | None]:
    """Versions needed to reproduce a result."""
    return {
        "mteb_version": _package_version("mteb"),
        "gym_version": _package_version("mteb-gym"),
        "gym_revision": git_revision(),
    }


def task_identity(task_name: str) -> dict[str, Any]:
    """Read task identity/provenance directly from installed MTEB metadata."""
    import mteb

    task = mteb.get_tasks(tasks=[task_name])[0]
    meta = task.metadata
    dataset = meta.dataset or {}

    return {
        "task_name": meta.name,
        "dataset_path": dataset.get("path"),
        "dataset_revision": dataset.get("revision"),
        "task_type": meta.type,
        "main_score": meta.main_score,
        "eval_splits": list(meta.eval_splits or []),
        "eval_langs": list(meta.eval_langs or []),
        "mteb_prompt": meta.prompt,
    }


def _slug(model: str | None) -> str:
    """Filesystem-safe but readable model identifier."""
    if not model:
        return "none"
    value = model.strip().replace("/", "-")
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "none"


def experiment_config_hash(config: dict[str, Any]) -> str:
    """Public name for the stable experiment configuration hash."""
    return config_hash(config)


def result_directory(
    output_folder: str | Path,
    judge_model: str,
    generator_model: str | None,
    config: dict[str, Any],
) -> Path:
    """Return results/<judge>__<generator>/<config-hash>/."""
    arm = f"{_slug(judge_model)}__{_slug(generator_model)}"
    return Path(output_folder) / arm / experiment_config_hash(config)
