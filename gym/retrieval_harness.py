"""
Retrieval harness.

Encodes the corpus once per model and caches it to disk (keyed by a filesystem
safe model id + a corpus fingerprint, so the '/' in 'intfloat/...' no longer
creates phantom subdirectories). Retrieval is a single dense matmul on
L2-normalised vectors — no per-query division, no overflow warnings.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .encoders import Encoder, cache_key


@dataclass
class Retrieved:
    """Top-k doc ids + texts a single model returned for a single query."""
    qid: str
    query: str
    doc_ids: list[str]
    doc_texts: list[str]


def _corpus_fingerprint(corpus: dict[str, str]) -> str:
    """Content hash over every doc id and text.

    Hashing anything less (e.g. length + first ids) lets a corpus revision or
    a change to title+text assembly silently reuse stale embeddings. Matching hashes also imply matching iteration order, so the
    cached matrix rows stay aligned with the live corpus dict. Prompt-variant
    isolation is handled separately by the encoder's cache_name.
    """
    h = hashlib.sha256()
    for did, text in corpus.items():
        h.update(did.encode())
        h.update(b"\x00")
        h.update(text.encode())
        h.update(b"\x01")
    return h.hexdigest()[:12]


def _doc_cap_suffix() -> str:
    """Non-default GYM_MAX_DOC_CHARS / GYM_MAX_SEQ change embeddings for long
    documents, so they key the cache. Empty at defaults: existing caches stay valid."""
    bits = []
    chars = int(os.environ.get("GYM_MAX_DOC_CHARS", "0") or 0)
    if chars:
        bits.append(f"chars{chars}")
    seq = int(os.environ.get("GYM_MAX_SEQ", "4096"))
    if seq != 4096:
        bits.append(f"seq{seq}")
    return ("_" + "-".join(bits)) if bits else ""


class RetrievalHarness:
    def __init__(self, cache_dir: Path, top_k: int = 10):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.top_k = top_k

    def _corpus_cache_path(self, model_name: str, fp: str) -> Path:
        return self.cache_dir / (
            f"corpus_{cache_key(model_name)}{_doc_cap_suffix()}_{fp}.npy")

    def encode_corpus(self, encoder: Encoder, corpus: dict[str, str]) -> np.ndarray:
        fp = _corpus_fingerprint(corpus)
        # cache_name (if present) distinguishes prompt variants of the same model
        path = self._corpus_cache_path(getattr(encoder, "cache_name", encoder.model_name), fp)
        if path.exists():
            return np.load(str(path))
        embs = encoder.encode_documents(list(corpus.values()))
        # Atomic write: a crash or SLURM preemption mid-save would otherwise
        # leave a truncated .npy that every future run silently trusts via
        # np.load. Write to a temp file (a file handle, so np.save does not
        # re-append .npy to the name), then os.replace -- atomic on the same
        # filesystem -- so the cache path only ever appears complete.
        tmp = str(path) + ".tmp"
        with open(tmp, "wb") as fh:
            np.save(fh, embs)
        os.replace(tmp, str(path))
        return embs

    def retrieve(self, encoder: Encoder, corpus: dict[str, str],
                 queries: list) -> list[Retrieved]:
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
