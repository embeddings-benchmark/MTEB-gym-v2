"""
Validation = how well does the gym ranking agree with a trusted ranking?

This is the whole point of the synthetic-query experiment: if gym rankings from
synthetic queries correlate with MTEB / Arena rankings as well as Rohan's real
queries did (rho ~ 0.86), synthetic queries are a viable, scalable substitute.

Reports Spearman rho and Kendall tau, with a bootstrap CI on the correlation so
you can say whether 0.71 and 0.86 are actually distinguishable given few models
(they often aren't — exactly Rohan's "noisy, big p-values" caveat).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_RESULTS_API = "https://api.github.com/repos/embeddings-benchmark/results/contents/results"
_RESULTS_RAW = "https://raw.githubusercontent.com/embeddings-benchmark/results/main/results"

BM25_NDCG = {
    "NFCorpus": 32.1,
    "SciFact": 68.63,
    "FiQA2018": 25.14,
    "ArguAna": 49.29,
}


def fetch_truth(
    models: list[str],
    task: str = "NFCorpus",
    split: str = "test",
) -> dict[str, float]:
    """Fetch official MTEB nDCG@10 scores for the requested models."""
    import urllib.request

    def _get_json(url: str):
        with urllib.request.urlopen(url, timeout=30) as response:
            return json.load(response)

    out: dict[str, float] = {}

    if "bm25" in models and task in BM25_NDCG:
        out["bm25"] = BM25_NDCG[task]

    for model in models:
        if "/" not in model:
            continue

        slug = model.replace("/", "__")
        try:
            revisions = [
                entry["name"]
                for entry in _get_json(f"{_RESULTS_API}/{slug}")
                if entry["name"] != "model_meta.json"
            ]
            revisions.sort(
                key=lambda revision: (
                    revision in ("external", "no_revision_available"),
                    revision,
                )
            )

            for revision in revisions:
                try:
                    data = _get_json(
                        f"{_RESULTS_RAW}/{slug}/{revision}/{task}.json"
                    )
                    scores = data.get("scores", {})
                    rows = (
                        scores.get(split)
                        or scores.get("test")
                        or next(iter(scores.values()), None)
                    )
                    if not rows:
                        continue

                    out[model] = round(rows[0]["ndcg_at_10"] * 100, 2)
                    break
                except Exception:
                    continue
        except Exception:
            continue

    return out


def _load_gym(path: str) -> dict[str, float]:
    """Load ratings from either legacy leaderboard.json or standardized results."""
    data = json.loads(Path(path).read_text())

    if "ratings" in data:
        return {m["model"]: m["rating"] for m in data["ratings"]}

    return {m["name"]: m["rating"] for m in data["models"]}


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

    return {
        "n_models": len(shared),
        "models": shared,
        "spearman_rho": float(rho), "spearman_p": float(p_rho),
        "spearman_p_exact": p_exact,  # exact permutation p, None if n > 8
        "kendall_tau": float(tau), "kendall_p": float(p_tau),
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

        # Definitions pending confirmation from Adnan.
        row.setdefault("spearman_top10", None)
        row.setdefault("kendall_ap", None)

        result_path.write_text(json.dumps(data, indent=2))
        out[str(result_path)] = agreement

    return out


def report(gym_path: str, ground_truth: dict[str, float], **kw) -> dict:
    return correlate(_load_gym(gym_path), ground_truth, **kw)
