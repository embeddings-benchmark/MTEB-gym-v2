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
import re
from dataclasses import dataclass, field

from .query_generator import _extract_json

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
    def __init__(self, client, flip_positions: bool = True, note: str = ""):
        self.client = client
        self.flip = flip_positions
        self.note = note
        # diagnostics
        self._first_picks = 0
        self._decisive = 0

    def _ask(self, query: str, first, second) -> tuple[str, str]:
        """One judgement. Returns (winner_in_AB_space, reasoning)."""
        msg = [
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content":
                f"Query: {query}\n\nSystem A results:\n{_format_results(first)}\n\n"
                f"System B results:\n{_format_results(second)}\n\nReply as JSON."},
        ]
        out = _extract_json(self.client.chat(msg, temperature=0.0))
        winner = out.get("winner", "tie")
        if winner not in ("A", "B", "tie"):
            winner = "tie"
        return winner, out.get("reasoning", "")

    def judge_pair(self, ra, rb, model_a: str, model_b: str) -> Verdict:
        """
        ra, rb: Retrieved objects for the same query from model_a, model_b.
        Runs both orders (if flip), averages to a fractional score for A.
        """
        raws: list[str] = []
        scores: list[float] = []
        reasonings: list[str] = []

        # Order 1: A shown first.
        w1, why1 = self._ask(ra.query, ra, rb)
        raws.append(w1); reasonings.append(why1)
        scores.append(_OUTCOME[w1])
        if w1 != "tie":
            self._decisive += 1
            if w1 == "A":            # picked the first-shown system
                self._first_picks += 1

        if self.flip:
            # Order 2: B shown first -> map its 'A'/'B' back to our A/B space.
            w2, why2 = self._ask(ra.query, rb, ra)
            raws.append(w2); reasonings.append(why2)
            # In flipped order, "A" means model_b won.
            mapped = {"A": 0.0, "B": 1.0, "tie": 0.5}[w2]
            scores.append(mapped)
            if w2 != "tie":
                self._decisive += 1
                if w2 == "A":        # again picked the first-shown system
                    self._first_picks += 1

        score_a = sum(scores) / len(scores)
        return Verdict(
            qid=ra.qid, query=ra.query, model_a=model_a, model_b=model_b,
            score_a=score_a, raw=raws,
            reasoning=" | ".join(r for r in reasonings if r), note=self.note,
        )

    def judge_all(self, retr_a: list, retr_b: list, model_a: str, model_b: str) -> list[Verdict]:
        by_qid = {r.qid: r for r in retr_b}
        verdicts = []
        for ra in retr_a:
            rb = by_qid.get(ra.qid)
            if rb is None:
                continue
            verdicts.append(self.judge_pair(ra, rb, model_a, model_b))
        return verdicts

    @property
    def a_first_rate(self) -> float | None:
        """Fraction of decisive verdicts that went to the first-shown system."""
        return (self._first_picks / self._decisive) if self._decisive else None
