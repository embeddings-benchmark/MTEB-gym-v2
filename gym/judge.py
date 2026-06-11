"""
Pairwise judge.

What changed and why it matters for the tie-rate problem
--------------------------------------------------------
The old judge ran each pair twice (A first, B first) and, on any disagreement,
*downgraded to a tie*. On NFCorpus that produced ~85% ties — almost all the
signal got thrown away, so the ELO gap was tiny and noisy.

Two models are rarely "equal"; more often the judge is order-sensitive. So
instead of collapsing disagreements to ties, we record a FRACTIONAL outcome per
query:

    score_A = mean over the two orders of {win:1, tie:0.5, loss:0}

A clean win in both orders -> 1.0. A split (A wins one order, loses the flipped
one) -> 0.5: a *soft* tie that still carries the magnitude into Bradley-Terry,
rather than a hard tie that contributes nothing. Across many queries these
fractions integrate into a clean ranking even when any single pair is ambiguous.

We also surface POSITION BIAS as a first-class diagnostic (Rohan's Fig-5 issue):
`a_first_rate` is how often the judge picks whichever system was shown first.
0.5 is unbiased; the team saw ~0.67 at depth 1. Flipping positions cancels it in
the score; the metric just lets you see how bad it is for a given judge.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .query_generator import _extract_json

logger = logging.getLogger(__name__)

_JUDGE_SYSTEM = (
    "You compare two retrieval systems. Given a query and two ranked result sets "
    "(System A and System B), decide which set better satisfies the query, judging "
    "relevance, coverage of the information need, and ranking quality. Be decisive "
    "when one set is clearly better; only answer 'tie' when they are genuinely "
    "indistinguishable in usefulness. "
    "Reply with strict JSON: {\"winner\": \"A\"|\"B\"|\"tie\", "
    "\"confidence\": \"low\"|\"medium\"|\"high\", \"reasoning\": \"one sentence\"}"
)

_OUTCOME = {"A": 1.0, "tie": 0.5, "B": 0.0}


@dataclass
class Verdict:
    qid: str
    query: str
    model_a: str
    model_b: str
    score_a: float          # 0..1 fractional, after position averaging
    raw: list[str] = field(default_factory=list)   # per-order winners, for audit
    reasoning: str = ""
    note: str = ""


def _format_results(r) -> str:
    return "\n".join(f"  {i+1}. {t[:300]}" for i, t in enumerate(r.doc_texts))


class Judge:
    def __init__(self, client, flip_positions: bool = True, note: str = "",
                 workers: int = 1):
        self.client = client
        self.flip = flip_positions
        self.note = note
        self.workers = max(1, workers)
        # diagnostics (lock-guarded: judge_all may run verdicts concurrently)
        self._lock = threading.Lock()
        self._first_picks = 0
        self._decisive = 0
        self._asks = 0
        self._parse_failures = 0
        self._identical = 0

    def _ask(self, query: str, first, second) -> tuple[str, str, bool]:
        """One judgement. Returns (winner_in_AB_space, reasoning, parsed_ok).

        parsed_ok is False when the model's output had no usable verdict; we
        still score it as a tie, but it is counted separately so a misbehaving
        judge (refusals, truncation, thinking-mode preambles) cannot silently
        flatten the leaderboard the way v0.1's hard ties did.
        """
        msg = [
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content":
                f"Query: {query}\n\nSystem A results:\n{_format_results(first)}\n\n"
                f"System B results:\n{_format_results(second)}\n\nReply as JSON."},
        ]
        out = _extract_json(self.client.chat(msg, temperature=0.0))
        raw_winner = out.get("winner")
        parsed_ok = raw_winner in ("A", "B", "tie")
        winner = raw_winner if parsed_ok else "tie"
        with self._lock:
            self._asks += 1
            if not parsed_ok:
                self._parse_failures += 1
        return winner, out.get("reasoning", ""), parsed_ok

    def judge_pair(self, ra, rb, model_a: str, model_b: str) -> Verdict:
        """
        ra, rb: Retrieved objects for the same query from model_a, model_b.
        Runs both orders (if flip), averages to a fractional score for A.
        Identical result sets are scored 0.5 without spending LLM calls.
        """
        if (list(ra.doc_ids) == list(rb.doc_ids)
                and list(ra.doc_texts) == list(rb.doc_texts)):
            with self._lock:
                self._identical += 1
            return Verdict(
                qid=ra.qid, query=ra.query, model_a=model_a, model_b=model_b,
                score_a=0.5, raw=["identical"],
                reasoning="identical result sets; tied without judging",
                note=self.note,
            )

        raws: list[str] = []
        scores: list[float] = []
        reasonings: list[str] = []

        # Order 1: A shown first.
        w1, why1, _ok1 = self._ask(ra.query, ra, rb)
        raws.append(w1); reasonings.append(why1)
        scores.append(_OUTCOME[w1])
        if w1 != "tie":
            with self._lock:
                self._decisive += 1
                if w1 == "A":        # picked the first-shown system
                    self._first_picks += 1

        if self.flip:
            # Order 2: B shown first -> map its 'A'/'B' back to our A/B space.
            w2, why2, _ok2 = self._ask(ra.query, rb, ra)
            raws.append(w2); reasonings.append(why2)
            # In flipped order, "A" means model_b won.
            mapped = {"A": 0.0, "B": 1.0, "tie": 0.5}[w2]
            scores.append(mapped)
            if w2 != "tie":
                with self._lock:
                    self._decisive += 1
                    if w2 == "A":    # again picked the first-shown system
                        self._first_picks += 1

        score_a = sum(scores) / len(scores)
        return Verdict(
            qid=ra.qid, query=ra.query, model_a=model_a, model_b=model_b,
            score_a=score_a, raw=raws,
            reasoning=" | ".join(r for r in reasonings if r), note=self.note,
        )

    def judge_all(self, retr_a: list, retr_b: list, model_a: str, model_b: str) -> list[Verdict]:
        by_qid = {r.qid: r for r in retr_b}
        pairs = [(ra, by_qid[ra.qid]) for ra in retr_a if ra.qid in by_qid]

        def _one(pair):
            ra, rb = pair
            return self.judge_pair(ra, rb, model_a, model_b)

        if self.workers <= 1 or len(pairs) <= 1:
            verdicts = []
            for i, pair in enumerate(pairs, 1):
                verdicts.append(_one(pair))
                if i % 25 == 0:
                    logger.info("%s vs %s: %d/%d queries judged",
                                model_a, model_b, i, len(pairs))
            return verdicts

        # Thread pool: clients are thread-safe HTTP callers, and a vLLM server
        # only reaches its throughput under concurrent requests. map() preserves
        # input order, so verdict files stay deterministic.
        verdicts = []
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for i, v in enumerate(pool.map(_one, pairs), 1):
                verdicts.append(v)
                if i % 25 == 0:
                    logger.info("%s vs %s: %d/%d queries judged",
                                model_a, model_b, i, len(pairs))
        return verdicts

    @property
    def a_first_rate(self) -> float | None:
        """Fraction of decisive verdicts that went to the first-shown system."""
        return (self._first_picks / self._decisive) if self._decisive else None

    @property
    def parse_failure_rate(self) -> float | None:
        """Fraction of judge calls whose output had no usable verdict."""
        return (self._parse_failures / self._asks) if self._asks else None

# ---------------------------------------------------------------------------
# Pointwise judge (ablation vs pairwise)
# ---------------------------------------------------------------------------
# Instead of comparing A vs B directly, we score each model's result set
# independently on a 1-5 scale, then rank models by average score.
#
# Cost: O(n * m) calls vs O(n * m^2) for pairwise — linear in queries,
# not quadratic in model pairs. Tejas's lit review says pointwise matches
# or beats pairwise on system-ranking correlation, which is the hypothesis
# we're testing.
#
# Output is a dict {model_name: [scores]} — one score per query — which
# feeds directly into PointwiseRanker.rank() to produce a leaderboard.
# ---------------------------------------------------------------------------

_POINTWISE_SYSTEM = (
    "You evaluate a single retrieval result set for a given query. "
    "Score how well the retrieved documents satisfy the information need, "
    "considering relevance, coverage, and ranking quality. "
    "5 = excellent: top results are highly relevant, directly address the query. "
    "4 = good: mostly relevant with minor gaps. "
    "3 = mediocre: some relevant documents but significant gaps or noise. "
    "2 = poor: mostly irrelevant, query barely addressed. "
    "1 = useless: results are entirely off-topic. "
    "Be decisive and use the full scale — do not default to 3. "
    "Reply with strict JSON: {\"score\": 1-5, \"reasoning\": \"one sentence\"}"
)


@dataclass
class PointwiseScore:
    qid: str
    query: str
    model: str
    score: float          # 1-5
    reasoning: str = ""
    note: str = ""


class PointwiseJudge:
    """
    Scores each model's retrieved results independently (1-5) rather than
    comparing two models head-to-head. Use PointwiseRanker.rank() to convert
    per-query scores into a final leaderboard.

    Usage:
        judge = PointwiseJudge(client)
        scores = judge.score_all(retrieved, model_name)   # list[PointwiseScore]

    Run score_all() for each model, then pass all scores to PointwiseRanker.
    """

    def __init__(self, client, note: str = ""):
        self.client = client
        self.note = note

    def _ask(self, query: str, retrieved) -> tuple[float, str]:
        """Score one (query, result_set) pair. Returns (score 1-5, reasoning)."""
        msg = [
            {"role": "system", "content": _POINTWISE_SYSTEM},
            {"role": "user", "content":
                f"Query: {query}\n\nRetrieved results:\n{_format_results(retrieved)}\n\n"
                f"Reply as JSON."},
        ]
        out = _extract_json(self.client.chat(msg, temperature=0.0))
        try:
            score = float(out.get("score", 3))
            score = max(1.0, min(5.0, score))   # clamp to valid range
        except (TypeError, ValueError):
            score = 3.0                          # neutral fallback
        return score, out.get("reasoning", "")

    def score_one(self, retrieved, model: str) -> PointwiseScore:
        """Score a single model's results for a single query."""
        score, reasoning = self._ask(retrieved.query, retrieved)
        return PointwiseScore(
            qid=retrieved.qid, query=retrieved.query,
            model=model, score=score, reasoning=reasoning, note=self.note,
        )

    def score_all(self, retrieved_list: list, model: str) -> list[PointwiseScore]:
        """Score all queries for one model. Call once per model."""
        return [self.score_one(r, model) for r in retrieved_list]


class PointwiseRanker:
    """
    Converts per-query pointwise scores into a ranked leaderboard.

    Usage:
        ranker = PointwiseRanker()
        for model, retrieved in model_results.items():
            scores = judge.score_all(retrieved, model)
            ranker.add(scores)
        leaderboard = ranker.rank()   # list of (model, mean_score, std)
    """

    def __init__(self):
        self._scores: dict[str, list[float]] = {}

    def add(self, scores: list[PointwiseScore]) -> None:
        for s in scores:
            self._scores.setdefault(s.model, []).append(s.score)

    def rank(self) -> list[dict]:
        """Returns list of dicts sorted by mean score descending."""
        out = []
        for model, scores in self._scores.items():
            arr = scores
            mean = sum(arr) / len(arr)
            variance = sum((x - mean) ** 2 for x in arr) / len(arr)
            std = variance ** 0.5
            out.append({
                "model": model,
                "mean_score": round(mean, 3),
                "std": round(std, 3),
                "n_queries": len(arr),
            })
        out.sort(key=lambda x: x["mean_score"], reverse=True)
        for i, row in enumerate(out, 1):
            row["rank"] = i
        return out

    def format_leaderboard(self) -> str:
        rows = self.rank()
        lines = [
            "=" * 60,
            f"{'Rank':<5}{'Model':<38}{'Score':>8}{'Std':>6}",
            "-" * 60,
        ]
        for r in rows:
            short = r["model"].split("/")[-1][:36]
            lines.append(f"{r['rank']:<5}{short:<38}{r['mean_score']:>8.3f}{r['std']:>6.3f}")
        lines.append("=" * 60)
        return "\n".join(lines)
