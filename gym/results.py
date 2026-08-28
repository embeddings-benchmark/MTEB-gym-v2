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
    """Stable hash over experiment-defining config, never over the hash itself."""
    payload = dict(config)
    payload.pop("config_hash", None)
    return config_hash(payload)


def result_directory(
    output_folder: str | Path,
    judge_model: str,
    generator_model: str | None,
    config: dict[str, Any],
) -> Path:
    """Return results/<judge>__<generator-or-arm>/<config-hash>/."""
    arm_name = str(config.get("arm", "synthetic"))
    second = (
        "human-queries"
        if arm_name == "human"
        else _slug(generator_model)
    )
    arm = f"{_slug(judge_model)}__{second}"
    return Path(output_folder) / arm / experiment_config_hash(config)


def client_model_id(client: Any) -> str | None:
    """Best-effort stable identifier for the model behind an LLM client."""
    if client is None:
        return None
    return str(getattr(client, "model", type(client).__name__))


def judge_instruction_metadata(task_name: str) -> dict[str, Any]:
    """Describe the instruction source using the same resolution as Judge."""
    import os

    from .judge import REGISTRY_VERSION, resolve_task_instruction

    override = os.environ.get("GYM_JUDGE_INSTR_OVERRIDE") or None
    if override is not None:
        instruction = override
        source = "env:GYM_JUDGE_INSTR_OVERRIDE"
    else:
        instruction = resolve_task_instruction(task_name)
        source = "mteb_task_prompt_registry" if instruction is not None else None

    return {
        "instruction": instruction,
        "instruction_source": source,
        "registry_version": REGISTRY_VERSION,
    }


def verdict_diagnostics(verdicts: list[Any]) -> dict[str, Any]:
    """Diagnostics computed only from persisted Verdict fields."""
    n = len(verdicts)
    identical = sum(
        1 for v in verdicts
        if list(getattr(v, "raw", []) or []) == ["identical"]
    )
    ties = sum(1 for v in verdicts if float(v.score_a) == 0.5)

    # Verdicts written before parsed_ok existed cannot tell us how many judge
    # calls were made. Report those diagnostics as unknown rather than as zero
    # or as a misleading partial rate on a mixed old/new cache.
    judged = [
        v for v in verdicts
        if list(getattr(v, "raw", []) or []) != ["identical"]
    ]
    missing_call_metadata = any(
        not (getattr(v, "parsed_ok", []) or []) for v in judged
    )

    asks = failures = first = decisive = 0
    for v in verdicts:
        for ok in getattr(v, "parsed_ok", []) or []:
            asks += 1
            if not ok:
                failures += 1
        for winner in getattr(v, "raw", []) or []:
            if winner in ("A", "B"):
                decisive += 1
                if winner == "A":
                    first += 1

    return {
        "judge_calls": None if missing_call_metadata else asks,
        "n_comparisons": n,
        "commit_rate": ((n - ties) / n) if n else None,
        "tie_rate": (ties / n) if n else None,
        "a_first_rate": (first / decisive) if decisive else None,
        "parse_failure_rate": (
            None if missing_call_metadata
            else ((failures / asks) if asks else None)
        ),
        "identical_retrieval_rate": (identical / n) if n else None,
    }

def build_result_record(
    *,
    cfg: Any,
    judge_client: Any,
    generator_client: Any,
    models: list[str],
    ratings: list[Any],
    verdicts: list[Any],
    n_queries_generated: int | None,
    n_queries: int,
    hf_subset: str,
    evaluation_time: float,
    corpus_cap: int | None,
    inject_qrels_docs: str | None,
    judge_system: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build an MTEB-like result record plus its hash-defining config."""
    import os

    task = task_identity(cfg.task_name)
    instruction = judge_instruction_metadata(cfg.task_name)
    diag = verdict_diagnostics(verdicts)

    experiment_config = {
        "arm": cfg.arm,
        "judge_model": client_model_id(judge_client),
        "generator_model": (
            client_model_id(generator_client)
            if cfg.arm == "synthetic" else None
        ),
        **instruction,
        "judge_system": judge_system,
        "n_queries_generated": n_queries_generated,
        "n_queries": n_queries,
        "docs_per_query": cfg.docs_per_query,
        "gen_overshoot": cfg.gen_overshoot,
        "min_query_chars": cfg.min_query_chars,
        "max_query_chars": cfg.max_query_chars,
        "filter": cfg.filter_queries,
        "filter_min_score": cfg.filter_min_score,
        "dedup_threshold": cfg.dedup_threshold,
        "top_k": cfg.top_k,
        "seed": cfg.seed,
        "method": cfg.method,
        "bootstrap_samples": cfg.bootstrap_samples,
        "flip_positions": cfg.flip_positions,
        "use_mteb_models": cfg.use_mteb_models,
        "elo_base": cfg.elo_base,
        "elo_scale": cfg.elo_scale,
        "corpus_split": cfg.corpus_split,
        "corpus_cap": corpus_cap,
        "inject_qrels_docs": inject_qrels_docs,
        "max_doc_chars": int(os.environ.get("GYM_MAX_DOC_CHARS", "0") or 0),
        "max_seq": int(os.environ.get("GYM_MAX_SEQ", "4096")),
        "models": list(models),
    }
    experiment_config["config_hash"] = experiment_config_hash(experiment_config)

    score_row = {
        "hf_subset": hf_subset,
        "languages": task["eval_langs"],
        "main_score": None,
        "spearman": None,
        "spearman_ci_low": None,
        "spearman_ci_high": None,
        "spearman_top10": None,
        "kendall": None,
        "kendall_ap": None,
        "p_permutation": None,
        "n_models": len(ratings),
        **{k: diag[k] for k in (
            "commit_rate",
            "tie_rate",
            "a_first_rate",
            "parse_failure_rate",
            "identical_retrieval_rate",
            "n_comparisons",
        )},
        # judge_ prefix = verdict-level (judge vs qrels, human-query arms only)
        "judge_kappa_committed": None,
        "judge_committed_agreement": None,
        "judge_clear_winner_agreement": None,
    }

    versions = runtime_versions()
    record = {
        "task_name": task["task_name"],
        "dataset_path": task["dataset_path"],
        "dataset_revision": task["dataset_revision"],
        "mteb_version": versions["mteb_version"],
        "gym_version": versions["gym_version"],
        "gym_revision": versions["gym_revision"],
        "evaluation_time": float(evaluation_time),
        "judge_calls": diag["judge_calls"],
        "config": experiment_config,
        "scores": {
            cfg.corpus_split: [score_row],
        },
        "ratings": [
            {
                "model": m.name,
                "rating": m.rating,
                "ci_low": m.ci_low,
                "ci_high": m.ci_high,
                "wins": m.wins,
                "losses": m.losses,
                "ties": m.ties,
                "n": m.n,
            }
            for m in ratings
        ],
    }

    return record, experiment_config
