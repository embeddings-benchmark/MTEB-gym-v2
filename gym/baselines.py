"""
BM25 baseline.

Rohan flagged BM25 as an easy, high-value addition: a lexical anchor every dense
model should beat. If a dense model loses to BM25 in the gym, that's a strong
smell test that something is wrong (bad prefixes, broken encoding, junk queries).

Implements Okapi BM25 with no heavy deps (pure numpy + a tokenizer regex), so it
behaves like just another "model" in a tournament.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from .retrieval_harness import Retrieved

_TOKEN = re.compile(r"[A-Za-z0-9]+")


def _tok(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Retriever:
    """Quacks like an Encoder+Harness pair: .name and .retrieve(corpus, queries)."""

    model_name = "bm25"

    def __init__(self, top_k: int = 10, k1: float = 1.5, b: float = 0.75):
        self.top_k = top_k
        self.k1 = k1
        self.b = b

    def retrieve(self, corpus: dict[str, str], queries: list) -> list[Retrieved]:
        doc_ids = list(corpus.keys())
        docs = [_tok(t) for t in corpus.values()]
        texts = list(corpus.values())
        N = len(docs)
        lengths = np.array([len(d) for d in docs], dtype=np.float32)
        avgdl = float(lengths.mean()) if N else 1.0

        df: Counter = Counter()
        for d in docs:
            for term in set(d):
                df[term] += 1
        idf = {t: math.log(1 + (N - n + 0.5) / (n + 0.5)) for t, n in df.items()}
        tfs = [Counter(d) for d in docs]

        out: list[Retrieved] = []
        k = min(self.top_k, N)
        for q in queries:
            qterms = _tok(q.text)
            scores = np.zeros(N, dtype=np.float32)
            for term in qterms:
                if term not in idf:
                    continue
                w = idf[term]
                for di in range(N):
                    f = tfs[di].get(term, 0)
                    if f == 0:
                        continue
                    denom = f + self.k1 * (1 - self.b + self.b * lengths[di] / avgdl)
                    scores[di] += w * (f * (self.k1 + 1)) / denom
            idx = np.argpartition(-scores, k - 1)[:k]
            idx = idx[np.argsort(-scores[idx])]
            out.append(Retrieved(
                qid=q.qid, query=q.text,
                doc_ids=[doc_ids[j] for j in idx],
                doc_texts=[texts[j] for j in idx],
            ))
        return out
