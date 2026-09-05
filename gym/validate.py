"""
Validation: how well the gym ranking agrees with a trusted ranking.

Reports Spearman rho and Kendall tau with bootstrap CIs and permutation
p-values -- correlations over ~25 systems carry wide intervals, so
significance is reported rather than assumed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# nDCG@10 anchors for entrants with no entry in the official results repo:
# bm25 self-run with mteb/baseline-bm25s (test split); colbert = published
# ColBERTv2 BEIR number. Extend as tasks are added.
BASELINE_NDCG = {
    "bm25": {"NFCorpus": 32.1, "SciFact": 68.63, "FiQA2018": 25.14, "ArguAna": 49.29},
    "colbert": {"NFCorpus": 33.8},
}


def fetch_truth(
    models: list[str],
    task: str = "NFCorpus",
    split: str = "test",
) -> dict[str, float]:
    """Load official MTEB nDCG@10 scores, evaluating cache misses via MTEB."""
    import mteb

    out: dict[str, float] = {}

    for m in models:
        if m in BASELINE_NDCG and task in BASELINE_NDCG[m]:
            out[m] = BASELINE_NDCG[m][task]

    # Mirror the normal MTEB evaluation path: populate the local cache from the
    # official results repository, then evaluate only task/model cache misses.
    cache = mteb.ResultCache()
    try:
        cache.download_from_remote()
    except Exception:
        # A remote-cache failure should not prevent using an existing local
        # cache or evaluating the model directly.
        pass

    task_obj = next(iter(mteb.get_tasks(tasks=[task])), None)
    if task_obj is None:
        return out

    for model in models:
        if "/" not in model:  # baselines are anchored above
            continue

        try:
            # ModelMeta lets mteb.evaluate inspect the cache before loading
            # model weights. Fall back to get_model for unregistered models.
            try:
                model_ref = mteb.get_model_meta(model)
            except Exception:
                model_ref = mteb.get_model(model)

            result = mteb.evaluate(
                model_ref,
                task_obj,
                cache=cache,
                overwrite_strategy="only-missing",
                show_progress_bar=False,
            )

            task_result = next(
                (r for r in result.task_results if r.task_name == task),
                None,
            )
            if task_result is None or not task_result.scores:
                continue

            if split in task_result.scores:
                score_split = split
            elif "test" in task_result.scores:
                score_split = "test"
            else:
                score_split = next(iter(task_result.scores))

            score = task_result.get_score(
                splits=[score_split],
                getter=lambda scores: scores["ndcg_at_10"],
            )
            out[model] = round(float(score) * 100, 2)
        except Exception:
            continue

    return out



def _exact_permutation_p(g: np.ndarray, t: np.ndarray, max_n: int = 8) -> float | None:
    """Two-tailed exact permutation p-value for Spearman rho.

    With few models the asymptotic p from scipy is an approximation; here the
    permutation distribution is small enough to enumerate exactly (n=7 ->
    5040 permutations). Returns None for n > max_n.
    """
    from itertools import permutations

    from scipy.stats import spearmanr

    n = len(g)
    if n > max_n:
        return None
    rho_obs, _ = spearmanr(g, t)
    if np.isnan(rho_obs):
        return None
    hits = 0
    total = 0
    for perm in permutations(range(n)):
        r, _ = spearmanr(g, t[list(perm)])
        if abs(r) >= abs(rho_obs) - 1e-12:
            hits += 1
        total += 1
    return hits / total


def _tau_ap(g: np.ndarray, t: np.ndarray) -> float:
    """AP rank correlation (Yilmaz et al. 2008): Kendall's tau weighted toward
    the top of the reference ranking. Reference = official scores t; candidate
    = gym ratings g. 1 = identical order, -1 = reversed."""
    order = np.argsort(-t)          # reference ranking, best first
    g_ord = g[order]
    n = len(g_ord)
    total = 0.0
    for i in range(1, n):
        above = g_ord[:i]
        total += float(np.sum(above > g_ord[i])) / i
    return float(2.0 * total / (n - 1) - 1.0)


def correlate(gym_ratings: dict[str, float],
              ground_truth: dict[str, float],
              bootstrap: int = 1000, seed: int = 0) -> dict:
    """gym_ratings / ground_truth: {model_name: score}. Aligns on shared models."""
    from scipy.stats import kendalltau, spearmanr

    shared = [m for m in gym_ratings if m in ground_truth]
    if len(shared) < 3:
        return {"error": f"need >=3 shared models, have {len(shared)}", "shared": shared}

    g = np.array([gym_ratings[m] for m in shared])
    t = np.array([ground_truth[m] for m in shared])

    rho, p_rho = spearmanr(g, t)
    tau, p_tau = kendalltau(g, t)
    p_exact = _exact_permutation_p(g, t)

    # bootstrap CI on rho by resampling models
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

    # Restricted to the OFFICIALLY top-k models: does the ranking still hold
    # among the good candidates, where a selection decision is actually made,
    # or does the aggregate rho come from separating strong from weak?
    top10 = None
    if len(shared) >= 12:
        top = np.argsort(-t)[:10]
        r10, _ = spearmanr(g[top], t[top])
        if not np.isnan(r10):
            top10 = float(r10)

    return {
        "n_models": len(shared),
        "models": shared,
        "spearman_rho": float(rho), "spearman_p": float(p_rho),
        "spearman_p_exact": p_exact,  # exact permutation p, None if n > 8
        "kendall_tau": float(tau), "kendall_p": float(p_tau),
        "spearman_top10": top10,      # None when fewer than 12 anchored models
        "kendall_ap": _tau_ap(g, t),  # top-weighted, official ranking as reference
        "spearman_ci95": ci,
        "gym_ranking": [m for m, _ in sorted(gym_ratings.items(), key=lambda x: -x[1]) if m in shared],
        "truth_ranking": [m for m, _ in sorted(ground_truth.items(), key=lambda x: -x[1]) if m in shared],
    }



def rank_agreement(
    results_dir: str | Path,
    *,
    bootstrap: int = 1000,
    seed: int = 0,
) -> dict[str, dict]:
    """Compare standardized Gym rankings with official MTEB nDCG@10 scores."""
    root = Path(results_dir)

    if root.is_file():
        paths = [root]
    else:
        paths = sorted(root.rglob("*.json"))

    out: dict[str, dict] = {}

    for result_path in paths:
        data = json.loads(result_path.read_text())

        # Skip legacy/cache JSON files; this API operates on standardized results.
        if "task_name" not in data or "ratings" not in data or "scores" not in data:
            continue

        task = data["task_name"]
        ratings = {m["model"]: m["rating"] for m in data["ratings"]}
        models = list(ratings)

        score_splits = data.get("scores", {})
        if not score_splits:
            continue

        split = next(iter(score_splits))
        rows = score_splits.get(split) or []
        if not rows:
            continue

        truth = fetch_truth(models, task=task, split=split)
        agreement = correlate(
            ratings,
            truth,
            bootstrap=bootstrap,
            seed=seed,
        )

        if "error" in agreement:
            out[str(result_path)] = agreement
            continue

        lo, hi = agreement["spearman_ci95"]
        row = rows[0]
        row["main_score"] = agreement["spearman_rho"]
        row["spearman"] = agreement["spearman_rho"]
        row["spearman_ci_low"] = lo
        row["spearman_ci_high"] = hi
        row["kendall"] = agreement["kendall_tau"]
        row["p_permutation"] = agreement["spearman_p_exact"]
        row["n_models"] = agreement["n_models"]

        row["spearman_top10"] = agreement.get("spearman_top10")
        row["kendall_ap"] = agreement.get("kendall_ap")

        result_path.write_text(json.dumps(data, indent=2))
        out[str(result_path)] = agreement

    return out
