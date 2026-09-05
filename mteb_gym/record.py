"""The result record: one JSON per run with ratings, the resolved configuration,
judge diagnostics and versions, so a reported ranking can be reproduced offline."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


def config_hash(config: dict[str, Any], length: int = 8) -> str:
    """Stable short hash over the experiment-defining config (never over itself)."""
    payload = {k: v for k, v in config.items() if k != "config_hash"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     default=str).encode()).hexdigest()[:length]


def _version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def git_revision() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _slug(model: str | None) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", (model or "").replace("/", "-")).strip("-") or "none"


def result_directory(root: Path, experiment: dict[str, Any]) -> Path:
    """root/<judge>__<generator or human-queries>/<config-hash>/"""
    second = "human-queries" if experiment["arm"] == "human" else _slug(experiment["generator_model"])
    return Path(root) / f"{_slug(experiment['judge_model'])}__{second}" / experiment["config_hash"]


def verdict_diagnostics(verdicts: list[Any]) -> dict[str, Any]:
    """Judge diagnostics from persisted verdicts, so cached and resumed runs report the truth."""
    n = len(verdicts)
    identical = sum(v.raw == ["identical"] for v in verdicts)
    ties = sum(float(v.score_a) == 0.5 for v in verdicts)
    asks = failures = first = decisive = 0
    for v in verdicts:
        asks += len(v.parsed_ok)
        failures += sum(not ok for ok in v.parsed_ok)
        for w in v.raw:
            if w in ("A", "B"):
                decisive += 1
                first += w == "A"
    return {
        "judge_calls": asks,
        "n_comparisons": n,
        "commit_rate": (n - ties) / n if n else None,
        "tie_rate": ties / n if n else None,
        "a_first_rate": first / decisive if decisive else None,
        "parse_failure_rate": failures / asks if asks else None,
        "identical_retrieval_rate": identical / n if n else None,
    }


def build(corpus, experiment: dict[str, Any], ratings: list[Any], verdicts: list[Any],
          evaluation_time: float, revisions: dict[str, str | None] | None = None) -> dict[str, Any]:
    dataset = getattr(corpus.metadata, "dataset", None) or {}
    return {
        "task_name": corpus.name,
        "corpus_fingerprint": corpus.fingerprint,
        "dataset": {"path": dataset.get("path"), "revision": dataset.get("revision")},
        "mteb_version": _version("mteb"),
        "gym_version": _version("mteb-gym"),
        "gym_revision": git_revision(),
        "evaluation_time": float(evaluation_time),
        "config": experiment,
        "diagnostics": verdict_diagnostics(verdicts),
        "ratings": [{"model": m.name, "revision": (revisions or {}).get(m.name), "rating": m.rating,
                     "ci_low": m.ci_low, "ci_high": m.ci_high, "wins": m.wins, "losses": m.losses,
                     "ties": m.ties, "n": m.n} for m in ratings],
    }   # rank_agreement adds "agreement" when official scores are available
