"""
Synthetic query generation + filtering.

Generation: show the LLM a few random corpus docs and ask for a realistic
search query answerable from the corpus but not a copy of the seeded docs.

The biggest lever on ground-truth correlation is FILTERING bad queries (too
short/long, not a real information need, trivially keyword-matched,
near-duplicate). Three gates, cheapest first so LLM calls are spent only on
plausible queries:

  1. heuristic   -- length + obvious degeneracy, free
  2. LLM quality -- 1-5 "is this a good retrieval query?", keep >= min
  3. dedup       -- near-duplicates by cosine on a cheap encoder
"""

from __future__ import annotations

import json
import logging
import random
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from .config import GymConfig

logger = logging.getLogger(__name__)


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
    """Tolerant JSON extraction — models add prose, fences, or reasoning traces.

    Returns the last object found, since reasoning models answer last.
    """
    text = re.sub(r"```(?:json)?", "", text or "")
    decoder = json.JSONDecoder()
    found: dict = {}
    i = text.find("{")
    while i != -1:
        try:
            obj, end = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            i = text.find("{", i + 1)
            continue
        if isinstance(obj, dict):
            found = obj
        i = text.find("{", end)
    return found


class QueryGenerator:
    def __init__(self, client, cfg: GymConfig):
        self.client = client
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        # Number of successfully generated, heuristic-valid queries before
        # quality filtering/dedup. Used only for reproducibility metadata.
        self.last_generated_count: int | None = None

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
        """Generate queries in order-preserving waves.

        Determinism contract: the shared RNG draws one doc sample per attempt,
        in the main thread, exactly as the old sequential loop did. The LLM
        calls fan out over a thread pool, but map() preserves input order and
        acceptance is evaluated in attempt order, so the generated query list
        (texts, seed docs, qids) is identical for any worker count — cache
        files and downstream hashes are unchanged.
        """
        doc_ids = list(corpus.keys())
        target = int(self.cfg.n_queries * (self.cfg.gen_overshoot if self.cfg.filter_queries else 1.0))
        cap = target * 3
        workers = max(1, getattr(self.cfg, "gen_workers", 1))
        out: list[Query] = []
        attempts = 0
        while len(out) < target and attempts < cap:
            # One wave = the number of queries still missing. Failed/rejected
            # attempts are retried by the next wave, never silently dropped.
            wave = min(target - len(out), cap - attempts)
            seeds = [self.rng.sample(doc_ids, min(self.cfg.docs_per_query, len(doc_ids)))
                     for _ in range(wave)]
            base = attempts
            attempts += wave
            if workers <= 1 or wave <= 1:
                results = [self._one(s, corpus, base + k) for k, s in enumerate(seeds)]
            else:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    results = list(pool.map(
                        lambda ks: self._one(ks[1], corpus, base + ks[0]),
                        enumerate(seeds)))
            for q in results:
                if q and self._heuristic_ok(q.text):
                    q.qid = f"q{len(out)}"   # qid = acceptance order, as before
                    out.append(q)
            logger.info("query generation: %d/%d kept (%d attempts)",
                        len(out), target, attempts)
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
        def _score(q: Query) -> int:
            """Score one query in place; return 1 if its score had to be
            defaulted (parse/call failure), 0 otherwise. Returning a count
            instead of mutating a shared `failures` avoids a data race when the
            pool runs _score across worker threads with no lock."""
            msg = [
                {"role": "system", "content": _FILTER_SYSTEM},
                {"role": "user", "content": f"Query: {q.text}\nReply as JSON."},
            ]
            try:
                out = _extract_json(self.client.chat(msg, temperature=0.0))
                q.quality = int(out.get("score", 3))
                return 0 if "score" in out else 1
            except Exception:  # noqa: BLE001
                q.quality = 3   # neutral on failure rather than dropping signal
                return 1

        workers = max(1, getattr(self.cfg, "judge_workers", 1))
        if workers <= 1 or len(queries) <= 1:
            failures = sum(_score(q) for q in queries)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                failures = sum(pool.map(_score, queries))
        if failures:
            logger.warning(
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
        self.last_generated_count = len(raw)
        kept = self.filter(raw)
        return kept
