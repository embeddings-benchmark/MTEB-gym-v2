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


def _load_gym(path: str) -> dict[str, float]:
    data = json.loads(Path(path).read_text())
    return {m["name"]: m["rating"] for m in data["models"]}


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
        "kendall_tau": float(tau), "kendall_p": float(p_tau),
        "spearman_ci95": ci,
        "gym_ranking": [m for m, _ in sorted(gym_ratings.items(), key=lambda x: -x[1]) if m in shared],
        "truth_ranking": [m for m, _ in sorted(ground_truth.items(), key=lambda x: -x[1]) if m in shared],
    }


def report(gym_path: str, ground_truth: dict[str, float], **kw) -> dict:
    return correlate(_load_gym(gym_path), ground_truth, **kw)
