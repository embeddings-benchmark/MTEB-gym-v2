"""
Rating from pairwise verdicts.

Verdicts carry a fractional `score_a` in [0, 1] (see judge.py). They are folded
into a win matrix W (W[i][j] = fractional wins of model i over j) and fit with
Bradley-Terry via the MM algorithm (Hunter 2004), reported on an Elo-style
scale, as Chatbot Arena and AlpacaEval do. Confidence intervals come from
bootstrap resampling of queries (clusters of correlated verdicts).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ModelRating:
    name: str
    rating: float
    ci_low: float
    ci_high: float
    wins: float
    losses: float
    ties: float
    n: int

    @property
    def ci(self) -> float:
        return (self.ci_high - self.ci_low) / 2


def _index_models(verdicts) -> list[str]:
    names = []
    for v in verdicts:
        for m in (v.model_a, v.model_b):
            if m not in names:
                names.append(m)
    return sorted(names)


def _win_matrix(verdicts, names) -> np.ndarray:
    idx = {n: i for i, n in enumerate(names)}
    W = np.zeros((len(names), len(names)), dtype=np.float64)
    for v in verdicts:
        a, b = idx[v.model_a], idx[v.model_b]
        W[a, b] += v.score_a
        W[b, a] += (1.0 - v.score_a)
    return W


def _bradley_terry(W: np.ndarray, iters: int = 200, tol: float = 1e-9,
                   smoothing: float = 0.1) -> np.ndarray:
    """MM algorithm. Returns positive strengths normalised to geometric mean 1.

    `smoothing` adds a small symmetric pseudo-win to every pair. Without it a
    model with zero total wins never gets its MM update and stays pinned at the
    initial strength (mid-table after normalisation) instead of converging to
    the bottom; it also keeps bootstrap resamples from going degenerate.
    """
    n = W.shape[0]
    if smoothing:
        W = W + smoothing * (1.0 - np.eye(n))
    p = np.ones(n, dtype=np.float64)
    wins = W.sum(axis=1)                      # total wins per model
    games = W + W.T                           # symmetric pair counts
    for _ in range(iters):
        p_old = p.copy()
        for i in range(n):
            denom = 0.0
            for j in range(n):
                if j == i:
                    continue
                gij = games[i, j]
                if gij > 0:
                    denom += gij / (p[i] + p[j])
            if denom > 0:
                p[i] = wins[i] / denom
        # normalise to geometric mean 1 for identifiability
        gm = np.exp(np.mean(np.log(np.clip(p, 1e-12, None))))
        p = p / gm
        if np.max(np.abs(p - p_old)) < tol:
            break
    return p


def _bt_to_elo(p: np.ndarray, base: float, scale: float) -> np.ndarray:
    r = scale * np.log10(np.clip(p, 1e-12, None))
    return r - r.mean() + base


def _tally(verdicts, names):
    idx = {n: i for i, n in enumerate(names)}
    w = np.zeros(len(names)); l = np.zeros(len(names)); t = np.zeros(len(names)); n = np.zeros(len(names))
    for v in verdicts:
        a, b = idx[v.model_a], idx[v.model_b]
        n[a] += 1; n[b] += 1
        if v.score_a > 0.5:
            w[a] += 1; l[b] += 1
        elif v.score_a < 0.5:
            l[a] += 1; w[b] += 1
        else:
            t[a] += 1; t[b] += 1
    return w, l, t, n


def rate(verdicts, base=1000.0, scale=400.0, bootstrap=1000, seed=0) -> list[ModelRating]:
    if not verdicts:
        return []
    names = _index_models(verdicts)

    def _ratings(vs) -> np.ndarray:
        return _bt_to_elo(_bradley_terry(_win_matrix(vs, names)), base, scale)

    point = _ratings(verdicts)

    # Bootstrap CIs: resample QUERIES (clusters of verdicts), not individual
    # verdicts. In a round-robin every query produces one verdict per model
    # pair, so verdicts sharing a qid are correlated (same query difficulty,
    # same retrieved sets); resampling them independently understates the CI
    # width. Chatbot Arena resamples battles because each battle is an
    # independent prompt — our sampling unit is the query.
    rng = np.random.default_rng(seed)
    if bootstrap and len(verdicts) > 1:
        by_qid: dict[str, list] = {}
        for v in verdicts:
            by_qid.setdefault(v.qid, []).append(v)
        qids = np.array(list(by_qid), dtype=object)
        rows = []
        skipped = 0
        for _ in range(bootstrap):
            sample: list = []
            for q in rng.choice(qids, size=len(qids), replace=True):
                sample.extend(by_qid[q])
            try:
                rows.append(_ratings(sample))
            except Exception:  # noqa: BLE001 - degenerate resample
                skipped += 1   # excluded from the CI rather than silently
                continue       # replaced by the point estimate
        if skipped:
            import logging
            logging.getLogger(__name__).warning(
                "bootstrap: %d/%d resamples were degenerate and excluded",
                skipped, bootstrap,
            )
        if rows:
            boot = np.vstack(rows)
            lo = np.percentile(boot, 2.5, axis=0)
            hi = np.percentile(boot, 97.5, axis=0)
        else:
            lo = hi = point
    else:
        lo = hi = point

    w, l, t, n = _tally(verdicts, names)
    out = [
        ModelRating(name=names[i], rating=float(point[i]),
                    ci_low=float(lo[i]), ci_high=float(hi[i]),
                    wins=float(w[i]), losses=float(l[i]), ties=float(t[i]), n=int(n[i]))
        for i in range(len(names))
    ]
    out.sort(key=lambda m: m.rating, reverse=True)
    return out


def format_leaderboard(ratings: list[ModelRating]) -> str:
    lines = ["=" * 78,
             f"{'Rank':<5}{'Model':<38}{'Rating':>8}{'CI±':>7}{'W':>5}{'L':>5}{'T':>5}",
             "-" * 78]
    for i, m in enumerate(ratings, 1):
        short = m.name.split("/")[-1][:36]
        lines.append(f"{i:<5}{short:<38}{m.rating:>8.0f}{m.ci:>7.0f}"
                     f"{m.wins:>5.0f}{m.losses:>5.0f}{m.ties:>5.0f}")
    lines.append("=" * 78)
    return "\n".join(lines)
