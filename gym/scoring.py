"""
Rating from pairwise verdicts.

Verdicts carry a fractional `score_a` in [0, 1] (see judge.py). We fold these
into a win matrix W where W[i][j] is the total fractional wins of model i over j,
then fit ratings two ways:

  bradley_terry : MLE strengths via the MM algorithm (Hunter 2004), converted to
                  an Elo-style scale. This is the principled choice for a fixed
                  batch of comparisons and is what Chatbot Arena / AlpacaEval use.
  elo           : classic online Elo, order-dependent, kept for parity/debugging.

Both report bootstrap confidence intervals by resampling verdicts with
replacement, which is the honest way to show how much the ranking can be trusted
given the number of comparisons (the thing Rohan warned was thin).
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


def _bradley_terry(W: np.ndarray, iters: int = 200, tol: float = 1e-9) -> np.ndarray:
    """MM algorithm. Returns positive strengths normalised to geometric mean 1."""
    n = W.shape[0]
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
            if denom > 0 and wins[i] > 0:
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


def _online_elo(verdicts, names, base, scale, k=32) -> np.ndarray:
    idx = {n: i for i, n in enumerate(names)}
    r = np.full(len(names), base, dtype=np.float64)
    for v in verdicts:
        a, b = idx[v.model_a], idx[v.model_b]
        ea = 1.0 / (1.0 + 10 ** ((r[b] - r[a]) / scale))
        r[a] += k * (v.score_a - ea)
        r[b] += k * ((1 - v.score_a) - (1 - ea))
    return r


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


def rate(verdicts, method="bradley_terry", base=1000.0, scale=400.0,
         bootstrap=1000, seed=0) -> list[ModelRating]:
    if not verdicts:
        return []
    names = _index_models(verdicts)

    def _ratings(vs) -> np.ndarray:
        if method == "elo":
            return _online_elo(vs, names, base, scale)
        return _bt_to_elo(_bradley_terry(_win_matrix(vs, names)), base, scale)

    point = _ratings(verdicts)

    # bootstrap CIs over resampled verdicts
    rng = np.random.default_rng(seed)
    if bootstrap and len(verdicts) > 1:
        boot = np.zeros((bootstrap, len(names)))
        v_arr = np.array(verdicts, dtype=object)
        for s in range(bootstrap):
            sample = list(rng.choice(v_arr, size=len(v_arr), replace=True))
            try:
                boot[s] = _ratings(sample)
            except Exception:  # noqa: BLE001 - degenerate resample
                boot[s] = point
        lo = np.percentile(boot, 2.5, axis=0)
        hi = np.percentile(boot, 97.5, axis=0)
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
