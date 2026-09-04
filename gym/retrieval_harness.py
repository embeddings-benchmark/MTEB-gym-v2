"""
Retrieval harness: encodes the corpus once per model, caches it on disk keyed by
model + corpus fingerprint, and retrieves with one matmul over unit vectors.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .encoders import cache_key


@dataclass
class Retrieved:
    """Top-k doc ids + texts a single model returned for a single query."""
    qid: str
    query: str
    doc_ids: list[str]
    doc_texts: list[str]


def _corpus_fingerprint(corpus: dict[str, str]) -> str:
    """Content hash over every doc id and text, in order: a corpus revision or a
    different title+text assembly can never reuse stale embeddings, and matching
    hashes imply cached rows align with the live corpus dict."""
    h = hashlib.sha256()
    for did, text in corpus.items():
        h.update(did.encode())
        h.update(b"\x00")
        h.update(text.encode())
        h.update(b"\x01")
    return h.hexdigest()[:12]


class RetrievalHarness:
    def __init__(self, cache_dir: Path, top_k: int = 10):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.top_k = top_k

    def _corpus_cache_path(self, model_name: str, fp: str) -> Path:
        return self.cache_dir / f"corpus_{cache_key(model_name)}_{fp}.npy"

    def encode_corpus(self, encoder, corpus: dict[str, str]) -> np.ndarray:
        fp = _corpus_fingerprint(corpus)
        path = self._corpus_cache_path(encoder.cache_name, fp)
        if path.exists():
            return np.load(str(path))
        embs = encoder.encode_documents(list(corpus.values()))
        # atomic write: a crash mid-save must not leave a truncated .npy that later runs trust
        tmp = str(path) + ".tmp"
        with open(tmp, "wb") as fh:
            np.save(fh, embs)
        os.replace(tmp, str(path))
        return embs

    def retrieve(self, encoder, corpus: dict[str, str], queries: list) -> list[Retrieved]:
        """`queries` is a list of objects with .qid and .text."""
        doc_ids = list(corpus.keys())
        doc_texts = list(corpus.values())
        corpus_embs = self.encode_corpus(encoder, corpus)          # (N, d), unit norm
        q_embs = encoder.encode_queries([q.text for q in queries])  # (Q, d), unit norm

        scores = q_embs @ corpus_embs.T                             # cosine, no division
        k = min(self.top_k, len(doc_ids))
        # argpartition for top-k then sort just those k (fast on big corpora)
        top_idx = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        out: list[Retrieved] = []
        for i, q in enumerate(queries):
            idx = top_idx[i]
            idx = idx[np.argsort(-scores[i, idx])]
            out.append(Retrieved(
                qid=q.qid, query=q.text,
                doc_ids=[doc_ids[j] for j in idx],
                doc_texts=[doc_texts[j] for j in idx],
            ))
        return out
