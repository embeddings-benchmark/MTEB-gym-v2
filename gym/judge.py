"""
Pairwise judge.

Each pair is judged in both presentation orders and scored fractionally:

    score_A = mean over the two orders of {win: 1, tie: 0.5, loss: 0}

A split across orders becomes 0.5 -- a soft tie that still carries magnitude
into Bradley-Terry instead of discarding the signal as a hard tie. Position
bias is surfaced as `a_first_rate` (how often the judge picks whichever system
was shown first; 0.5 is unbiased). Flipping orders cancels it in the score;
the metric only makes it visible.
"""

from __future__ import annotations
import os

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

def task_prompt(task_name: str) -> str | None:
    """The task's own mteb prompt text, for passing as cfg.judge_instruction
    when a run should judge by the task's stated criterion. None when the
    task defines no prompt. Beware: some task families (e.g. BRIGHT) carry
    encoder prefixes here, not task definitions."""
    import mteb

    prompt = mteb.get_tasks(tasks=[task_name])[0].metadata.prompt
    return prompt if isinstance(prompt, str) else (prompt or {}).get("query")


def judge_system(instruction: str | None = None) -> str:
    """Generic judge prompt, with a task instruction injected when given.
    Resolution lives in results.judge_instruction_metadata (env override >
    config text > mteb task prompt > generic); the env override exists so
    control arms (placebo / cross-corpus text) run without config edits. The
    verdict-cache signature hashes the RESOLVED prompt, so any instruction
    gets its own cache namespace and can never collide with a generic run."""
    instr = os.environ.get("GYM_JUDGE_INSTR_OVERRIDE") or instruction
    if not instr:
        return _JUDGE_SYSTEM
    return (
        "You compare two retrieval systems. The retrieval task is: "
        + instr.rstrip(".") + ". "
        "Given a query and two ranked result sets "
        "(System A and System B), decide which set better satisfies this task for "
        "the query, judging task fit, coverage of the information need, and ranking "
        "quality. Be decisive "
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
    # per-order judge-parse success, so parse_failure_rate can be recomputed
    # from disk on cached/resumed runs (empty on verdicts written before this
    # field existed, and on "identical" short-circuits that make no judge call)
    parsed_ok: list[bool] = field(default_factory=list)


def _format_results(r) -> str:
    return "\n".join(f"  {i+1}. {t[:300]}" for i, t in enumerate(r.doc_texts))


def _parse_response(raw: str) -> tuple[str, str, bool]:
    """(winner, reasoning, parsed_ok) from a judge response. EnsembleClient.chat
    returns a JSON array of member responses — majority-vote those."""
    try:
        members = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        members = None
    # A real ensemble response is a JSON array of member-response STRINGS
    # (EnsembleClient.chat packs them that way). A lone judge that happens to
    # emit a top-level JSON array of objects must NOT be voted as an ensemble:
    # that path drops every dict element and silently returns a parse-failure
    # tie (the tie-flattening failure mode). Require all-string elements, else fall
    # through to single-object parsing below.
    if isinstance(members, list) and members and all(isinstance(m, str) for m in members):
        from .clients import EnsembleClient
        outs = [_extract_json(m) for m in members]
        winners = [o["winner"] for o in outs if o.get("winner") in ("A", "B", "tie")]
        if not winners:
            return "tie", "", False
        winner, _ = EnsembleClient.vote(winners)
        reasoning = " || ".join(o.get("reasoning", "") for o in outs if o.get("reasoning"))
        return winner, reasoning, True
    out = _extract_json(raw)
    raw_winner = out.get("winner")
    parsed_ok = raw_winner in ("A", "B", "tie")
    return (raw_winner if parsed_ok else "tie"), out.get("reasoning", ""), parsed_ok


class Judge:
    def __init__(self, client, flip_positions: bool = True, note: str = "",
                 workers: int = 1, instruction: str | None = None):
        self.system = judge_system(instruction)
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
        flatten the leaderboard into ties.
        """
        msg = [
            {"role": "system", "content": self.system},
            {"role": "user", "content":
                f"Query: {query}\n\nSystem A results:\n{_format_results(first)}\n\n"
                f"System B results:\n{_format_results(second)}\n\nReply as JSON."},
        ]
        winner, reasoning, parsed_ok = _parse_response(self.client.chat(msg, temperature=0.0))
        with self._lock:
            self._asks += 1
            if not parsed_ok:
                self._parse_failures += 1
        return winner, reasoning, parsed_ok

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
        parsed: list[bool] = []

        # Order 1: A shown first.
        w1, why1, ok1 = self._ask(ra.query, ra, rb)
        raws.append(w1); reasonings.append(why1); parsed.append(ok1)
        scores.append(_OUTCOME[w1])
        if w1 != "tie":
            with self._lock:
                self._decisive += 1
                if w1 == "A":        # picked the first-shown system
                    self._first_picks += 1

        if self.flip:
            # Order 2: B shown first -> map its 'A'/'B' back to our A/B space.
            w2, why2, ok2 = self._ask(ra.query, rb, ra)
            raws.append(w2); reasonings.append(why2); parsed.append(ok2)
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
            parsed_ok=parsed,
        )

    def _early_failure_guard(self, n_done: int, n_failed: int,
                             n_identical: int = 0) -> None:
        """Abort loudly when the judge endpoint is effectively dead.

        A dead endpoint / bad API key makes every verdict a parse-failure tie;
        without this guard that surfaces as a flat leaderboard instead of an
        error (observed: 12k/12k silent ties on a broken judge URL). Checked
        after the first GYM_EARLY_FAIL_WINDOW verdicts (default 30): if more
        than GYM_MAX_EARLY_PARSE_FAIL (default 0.5) of them failed to parse,
        raise instead of continuing.
        """
        import os
        window = int(os.environ.get("GYM_EARLY_FAIL_WINDOW", "30"))
        if n_done != window:
            return
        max_rate = float(os.environ.get("GYM_MAX_EARLY_PARSE_FAIL", "0.5"))
        rate = n_failed / max(n_done, 1)
        if rate > max_rate:
            raise RuntimeError(
                f"judge parse-failure rate {rate:.0%} over the first {n_done} "
                f"verdicts (> {max_rate:.0%}): the judge endpoint is likely dead "
                "or the API key invalid. Aborting instead of silently scoring "
                "ties. Set GYM_MAX_EARLY_PARSE_FAIL=1.0 to override.")
        # A structurally distinct failure the rate above is blind to: the judge is
        # never CALLED because every pair's retrieved lists are byte-identical, so
        # there are no parse failures to count. Observed on CUREv1, whose corpus
        # loaded as 11 empty documents (a multi-subset dataset whose subset names
        # were read as documents): 21k/21k identical-list ties, every model left on
        # the default rating, Spearman nan, parse-failure rate 0.0.
        # Genuine judge ties carry parsed_ok entries; identical-list short-circuits
        # carry raw == ["identical"] with an empty parsed_ok, so the two are cleanly
        # separable and this cannot fire on a merely tie-heavy corpus (our real ones
        # sit at 0.26-0.66 ties). A high rate here always means degenerate
        # retrieval: an empty or mis-loaded corpus, or a corpus no larger than
        # top_k, where every model necessarily returns the same set.
        max_ident = float(os.environ.get("GYM_MAX_EARLY_IDENTICAL", "0.95"))
        irate = n_identical / max(n_done, 1)
        if irate > max_ident:
            raise RuntimeError(
                f"{irate:.0%} of the first {n_done} pairs had byte-identical "
                f"retrieved lists (over {max_ident:.0%}), so the judge was never "
                "called and every verdict is a tie. The corpus is almost certainly "
                "empty, mis-loaded, or smaller than top_k; check the corpus loader "
                "for this task before spending judge budget. Set "
                "GYM_MAX_EARLY_IDENTICAL=1.0 to override.")

    def judge_all(self, retr_a: list, retr_b: list, model_a: str, model_b: str,
                  on_verdict=None) -> list[Verdict]:
        """Judge every shared query. `on_verdict`, if given, is called once per
        verdict as it completes (from any worker thread), letting the caller
        persist incrementally; the callback does its own locking."""
        by_qid = {r.qid: r for r in retr_b}
        pairs = [(ra, by_qid[ra.qid]) for ra in retr_a if ra.qid in by_qid]

        import threading
        _efg_lock = threading.Lock()
        _efg = {"done": 0, "failed": 0, "identical": 0}

        def _one(pair):
            ra, rb = pair
            v = self.judge_pair(ra, rb, model_a, model_b)
            # early dead-endpoint guard: a verdict whose every judge call failed
            # to parse (parsed_ok all-False; empty = no call made, not a failure)
            failed = bool(getattr(v, "parsed_ok", None)) and not any(v.parsed_ok)
            identical = list(getattr(v, "raw", []) or []) == ["identical"]
            with _efg_lock:
                _efg["done"] += 1
                if failed:
                    _efg["failed"] += 1
                if identical:
                    _efg["identical"] += 1
                self._early_failure_guard(_efg["done"], _efg["failed"],
                                          _efg["identical"])
            if on_verdict is not None:
                on_verdict(v)
            return v

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
# Pointwise judge (ablation vs pairwise): score each model's result set
# independently on a 1-5 scale, rank by average. O(n*m) calls vs O(n*m^2);
# output {model: [scores]} feeds PointwiseRanker.rank().
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
