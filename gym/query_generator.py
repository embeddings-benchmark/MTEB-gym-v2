"""
Synthetic query generation + filtering.

The generation idea (unchanged): show the LLM a few random corpus docs and ask
for a realistic search query that is *answerable from the corpus* but *not a
copy* of the seeded docs, so retrieval is non-trivial.

The new part is the FILTER. Rohan found that the single biggest lever on
correlation with ground truth was throwing away bad synthetic queries. A query
is "bad" if it's too short/long, not a real information need, trivially answered
by keyword overlap, or a near-duplicate of another query. We apply three gates:

  1. heuristic   — length + obvious degeneracy, free
  2. LLM quality — 1-5 score for "is this a good retrieval query?", keep >= min
  3. dedup       — drop near-duplicates by cosine on a cheap encoder

Order matters: cheap gates first so we only spend LLM calls on plausible queries.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass

import numpy as np

from .config import GymConfig


@dataclass
class Query:
    qid: str
    text: str
    seed_doc_ids: list[str]
    quality: int | None = None


_GEN_SYSTEM = (
    "You write realistic search queries for evaluating retrieval systems. "
    "Given a few documents from a corpus, produce ONE natural query a real user "
    "might type that is answerable using the corpus but is NOT a restatement of "
    "any single shown document. Vary phrasing and specificity. "
    "Reply with strict JSON: {\"query\": \"...\"}"
)

_FILTER_SYSTEM = (
    "You rate the quality of search queries for benchmarking retrieval models. "
    "A 5 is a clear, specific, genuinely answerable information need that would "
    "separate good retrievers from bad ones. A 1 is vague, trivial, malformed, "
    "or answerable by exact keyword match. "
    "Reply with strict JSON: {\"score\": 1-5, \"reason\": \"...\"}"
)


def _extract_json(text: str) -> dict:
    """Tolerant JSON extraction — models love to add prose or fences."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {}


class QueryGenerator:
    def __init__(self, client, cfg: GymConfig):
        self.client = client
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)

    # ---------------------------------------------------------------- generate
    def _one(self, doc_ids: list[str], docs: dict[str, str], idx: int) -> Query | None:
        snippet = "\n\n".join(f"[{i+1}] {docs[d][:600]}" for i, d in enumerate(doc_ids))
        msg = [
            {"role": "system", "content": _GEN_SYSTEM},
            {"role": "user", "content": f"Documents:\n{snippet}\n\nWrite one query as JSON."},
        ]
        try:
            out = _extract_json(self.client.chat(msg, temperature=0.7))
        except Exception:  # noqa: BLE001 - one flaky call shouldn't kill the run
            return None
        q = (out.get("query") or "").strip()
        if not q:
            return None
        return Query(qid=f"q{idx}", text=q, seed_doc_ids=doc_ids)

    def generate(self, corpus: dict[str, str]) -> list[Query]:
        doc_ids = list(corpus.keys())
        target = int(self.cfg.n_queries * (self.cfg.gen_overshoot if self.cfg.filter_queries else 1.0))
        out: list[Query] = []
        i = 0
        attempts = 0
        while len(out) < target and attempts < target * 3:
            attempts += 1
            seeds = self.rng.sample(doc_ids, min(self.cfg.docs_per_query, len(doc_ids)))
            q = self._one(seeds, corpus, i)
            if q and self._heuristic_ok(q.text):
                out.append(q)
                i += 1
        return out

    # ----------------------------------------------------------------- filters
    def _heuristic_ok(self, text: str) -> bool:
        n = len(text)
        if n < self.cfg.min_query_chars or n > self.cfg.max_query_chars:
            return False
        if text.count(" ") < 2:                 # one or two words: too trivial
            return False
        if not re.search(r"[A-Za-z]", text):     # must contain letters
            return False
        return True

    def _llm_quality(self, queries: list[Query]) -> None:
        failures = 0

        def _score(q: Query) -> None:
            nonlocal failures
            msg = [
                {"role": "system", "content": _FILTER_SYSTEM},
                {"role": "user", "content": f"Query: {q.text}\nReply as JSON."},
            ]
            try:
                out = _extract_json(self.client.chat(msg, temperature=0.0))
                if "score" not in out:
                    failures += 1
                q.quality = int(out.get("score", 3))
            except Exception:  # noqa: BLE001
                failures += 1
                q.quality = 3   # neutral on failure rather than dropping signal

        workers = max(1, getattr(self.cfg, "judge_workers", 1))
        if workers <= 1 or len(queries) <= 1:
            for q in queries:
                _score(q)
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(_score, queries))
        if failures:
            import logging
            logging.getLogger(__name__).warning(
                "quality filter: %d/%d scores defaulted to 3 (parse/call failure); "
                "a misbehaving filter model can silently disable filtering",
                failures, len(queries),
            )

    def _dedup(self, queries: list[Query]) -> list[Query]:
        """Drop near-duplicate queries. Uses a tiny encoder if available; else skips."""
        try:
            from .encoders import Encoder
            enc = Encoder("sentence-transformers/all-MiniLM-L6-v2")
            vecs = enc.encode_queries([q.text for q in queries])
        except Exception:  # noqa: BLE001 - no ML deps / offline -> skip dedup
            return queries
        keep: list[int] = []
        for i in range(len(queries)):
            dup = False
            for j in keep:
                if float(vecs[i] @ vecs[j]) >= self.cfg.dedup_threshold:
                    dup = True
                    break
            if not dup:
                keep.append(i)
        return [queries[i] for i in keep]

    def filter(self, queries: list[Query]) -> list[Query]:
        if not self.cfg.filter_queries:
            return queries[: self.cfg.n_queries]
        self._llm_quality(queries)
        good = [q for q in queries if (q.quality or 0) >= self.cfg.filter_min_score]
        good = self._dedup(good)
        # Re-id so downstream files are clean and stable.
        for k, q in enumerate(good[: self.cfg.n_queries]):
            q.qid = f"q{k}"
        return good[: self.cfg.n_queries]

    # ------------------------------------------------------------------ public
    def run(self, corpus: dict[str, str]) -> list[Query]:
        raw = self.generate(corpus)
        kept = self.filter(raw)
        return kept
