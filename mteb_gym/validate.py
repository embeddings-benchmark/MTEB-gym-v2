"""
Validation: how well the gym ranking agrees with the official MTEB ranking on
the same task. Official scores come from the MTEB results repository through
mteb's result cache. Models without one are skipped, or, with
evaluate_missing=True, evaluated by mteb on the real task (real qrels) and
stored in that same cache.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def fetch_truth(models: list[str], task: str, *, evaluate_missing: bool = False) -> tuple[dict[str, float], dict[str, str]]:
    """Official main score (nDCG@10, x100) per model on `task`, and per model whether
    it was "official" or "self-run"."""
    import mteb

    cache = mteb.ResultCache()
    try:
        cache.download_from_remote()
    except Exception:  # noqa: BLE001 - offline: use the local copy
        logger.warning("could not refresh the MTEB results cache; using the local copy")
    task_obj = mteb.get_task(task)
    scores: dict[str, float] = {}
    source: dict[str, str] = {}
    for name in models:
        meta = mteb.get_model_meta(name)
        result = cache.load_task_result(task, meta)
        if result is not None:
            source[name] = "official"
        elif evaluate_missing:
            logger.info("no official result for %s on %s; evaluating with mteb", name, task)
            res = mteb.evaluate(meta, task_obj, cache=cache, overwrite_strategy="only-missing", show_progress_bar=False)
            result = res.task_results[0] if res.task_results else None
            source[name] = "self-run"
        if result is None:
            logger.warning("no official result for %s on %s; skipped (evaluate_missing=True to run it)", name, task)
            continue
        scores[name] = round(float(result.get_score()) * 100, 2)
    return scores, source


def _tau_ap(g: np.ndarray, t: np.ndarray) -> float:
    """AP rank correlation (Yilmaz et al. 2008): Kendall's tau weighted toward the
    top of the reference ranking t. 1 = identical order, -1 = reversed."""
    g_ord = g[np.argsort(-t)]
    n = len(g_ord)
    total = sum(float(np.sum(g_ord[:i] > g_ord[i])) / i for i in range(1, n))
    return float(2.0 * total / (n - 1) - 1.0)


def correlate(gym_ratings: dict[str, float], ground_truth: dict[str, float],
              bootstrap: int = 1000, seed: int = 0) -> dict:
    """Rank agreement over the models present in both dicts."""
    from scipy.stats import kendalltau, spearmanr

    shared = [m for m in gym_ratings if m in ground_truth]
    if len(shared) < 3:
        return {"error": f"need >=3 shared models, have {len(shared)}", "shared": shared}
    g = np.array([gym_ratings[m] for m in shared])
    t = np.array([ground_truth[m] for m in shared])
    rho, p_rho = spearmanr(g, t)
    tau, p_tau = kendalltau(g, t)

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(bootstrap):
        idx = rng.integers(0, len(shared), len(shared))
        if len(set(idx)) < 3:
            continue
        r, _ = spearmanr(g[idx], t[idx])
        if not np.isnan(r):
            boots.append(r)
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) if boots else (None, None)

    top10 = None
    if len(shared) >= 12:   # among the officially top-10 models only: where a selection decision is made
        top = np.argsort(-t)[:10]
        r10, _ = spearmanr(g[top], t[top])
        top10 = None if np.isnan(r10) else float(r10)

    return {
        "n_models": len(shared), "models": shared,
        "spearman_rho": float(rho), "spearman_p": float(p_rho),
        "kendall_tau": float(tau), "kendall_p": float(p_tau),
        "spearman_top10": top10, "kendall_ap": _tau_ap(g, t), "spearman_ci95": ci,
        "gym_ranking": sorted(shared, key=lambda m: -gym_ratings[m]),
        "truth_ranking": sorted(shared, key=lambda m: -ground_truth[m]),
    }


def rank_agreement(results_dir: str | Path, *, evaluate_missing: bool = False,
                   bootstrap: int = 1000, seed: int = 0) -> dict[str, dict]:
    """Agreement with the official MTEB ranking for every gym result record under
    `results_dir` (or one record file); written into each record as "agreement"."""
    root = Path(results_dir)
    paths = [root] if root.is_file() else sorted(root.rglob("*.json"))
    out: dict[str, dict] = {}
    for path in paths:
        data = json.loads(path.read_text())
        if not {"task_name", "ratings", "config"} <= set(data):
            continue   # only gym result records; other JSON files in the tree are ignored
        ratings = {r["model"]: r["rating"] for r in data["ratings"]}
        truth, source = fetch_truth(list(ratings), data["task_name"], evaluate_missing=evaluate_missing)
        agreement = correlate(ratings, truth, bootstrap=bootstrap, seed=seed)
        agreement["truth_source"] = source
        out[str(path)] = data["agreement"] = agreement
        path.write_text(json.dumps(data, indent=2))
    return out
