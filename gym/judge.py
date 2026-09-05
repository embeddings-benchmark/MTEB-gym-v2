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

import logging
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
    """The task's own criterion, verbatim from mteb TaskMetadata.prompt. None
    when the task has no prompt or it is an encoder prefix rather than a task
    statement ("Represent this post for searching passages: ", BRIGHT)."""
    import mteb

    p = mteb.get_tasks(tasks=[task_name])[0].metadata.prompt
    p = (p if isinstance(p, str) else (p or {}).get("query") or "").strip()
    if not p or p.endswith(":") or p.lower().startswith("represent "):
        return None
    return p


def judge_system(instruction: str | None = None) -> str:
    """Generic judge prompt, or the same prompt with the task's criterion stated
    (see results.judge_instruction_metadata). The verdict cache keys on the
    resolved prompt, so instructed and generic runs never share verdicts."""
    if not instruction:
        return _JUDGE_SYSTEM
    return (
        "You compare two retrieval systems. The retrieval task is: "
        + instruction.rstrip(".") + ". "
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
    parsed_ok: list[bool] = field(default_factory=list)  # per order; empty = no judge call


def _format_results(r) -> str:
    return "\n".join(f"  {i+1}. {t[:300]}" for i, t in enumerate(r.doc_texts))


def _parse_response(raw: str) -> tuple[str, str, bool]:
    """(winner, reasoning, parsed_ok). An unparseable response scores as a tie
    but is flagged, so a misbehaving judge cannot silently flatten the board."""
    out = _extract_json(raw)
    winner = out.get("winner")
    ok = winner in ("A", "B", "tie")
    return (winner if ok else "tie"), out.get("reasoning", ""), ok


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
        """One judgement: (winner in A/B space, reasoning, parsed_ok)."""
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

    # Dead-endpoint guards, checked once after the first EARLY_FAIL_WINDOW verdicts.
    EARLY_FAIL_WINDOW = 30
    MAX_EARLY_PARSE_FAIL = 0.5   # above this fraction unparseable: dead judge or bad API key
    MAX_EARLY_IDENTICAL = 0.95   # above this fraction identical lists: empty or mis-loaded corpus

    def _early_failure_guard(self, n_done: int, n_failed: int, n_identical: int) -> None:
        """Abort loudly instead of scoring a whole run as ties."""
        if n_done != self.EARLY_FAIL_WINDOW:
            return
        if n_failed / n_done > self.MAX_EARLY_PARSE_FAIL:
            raise RuntimeError(
                f"judge parse-failure rate {n_failed / n_done:.0%} over the first {n_done} "
                "verdicts: the judge endpoint is likely dead or the API key invalid.")
        if n_identical / n_done > self.MAX_EARLY_IDENTICAL:
            raise RuntimeError(
                f"{n_identical / n_done:.0%} of the first {n_done} pairs had identical retrieved "
                "lists, so the judge was never called: the corpus is likely empty, "
                "mis-loaded, or smaller than top_k.")

    def judge_all(self, retr_a: list, retr_b: list, model_a: str, model_b: str,
                  on_verdict=None) -> list[Verdict]:
        """Judge every shared query. `on_verdict`, if given, is called once per
        verdict as it completes (from any worker thread), letting the caller
        persist incrementally; the callback does its own locking."""
        by_qid = {r.qid: r for r in retr_b}
        pairs = [(ra, by_qid[ra.qid]) for ra in retr_a if ra.qid in by_qid]

        guard_lock = threading.Lock()
        counts = {"done": 0, "failed": 0, "identical": 0}

        def _one(pair):
            v = self.judge_pair(*pair, model_a, model_b)
            with guard_lock:
                counts["done"] += 1
                counts["failed"] += bool(v.parsed_ok) and not any(v.parsed_ok)
                counts["identical"] += v.raw == ["identical"]
                self._early_failure_guard(counts["done"], counts["failed"], counts["identical"])
            if on_verdict is not None:
                on_verdict(v)
            return v

        def _collect(results):
            out = []
            for i, v in enumerate(results, 1):
                out.append(v)
                if i % 25 == 0:
                    logger.info("%s vs %s: %d/%d queries judged", model_a, model_b, i, len(pairs))
            return out

        if self.workers <= 1 or len(pairs) <= 1:
            return _collect(map(_one, pairs))
        with ThreadPoolExecutor(max_workers=self.workers) as pool:   # map keeps input order
            return _collect(pool.map(_one, pairs))

    @property
    def a_first_rate(self) -> float | None:
        """Fraction of decisive verdicts that went to the first-shown system."""
        return (self._first_picks / self._decisive) if self._decisive else None

    @property
    def parse_failure_rate(self) -> float | None:
        """Fraction of judge calls whose output had no usable verdict."""
        return (self._parse_failures / self._asks) if self._asks else None
