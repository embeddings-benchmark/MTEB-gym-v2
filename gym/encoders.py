"""
Model-aware encoding.

Two things every previous version got wrong, and why they mattered:

1. PREFIXES. Many embedding models are *asymmetric*: they were trained with
   distinct query and document prefixes, and they silently underperform without
   them. e5 wants "query: " / "passage: ", bge wants a query instruction, nomic
   wants "search_query:" / "search_document:". If you encode an e5 model with no
   prefix, its retrieval quality collapses — which is exactly the kind of thing
   that makes a *stronger* model (e5-small) lose to a weaker one (MiniLM) in a
   head-to-head. MiniLM/mpnet are symmetric and need no prefix, so they were
   never penalised. That asymmetry was almost certainly the cause of the
   "gym says MiniLM > e5-small but MTEB says the opposite" mismatch.

2. NORMALISATION. Cosine similarity = dot product of L2-normalised vectors.
   Doing the division at query time (q . d / (|q||d|)) on un-normalised vectors
   invites divide-by-zero / overflow on degenerate rows. We normalise once at
   encode time and then only ever take dot products. No runtime division, no
   warnings, and retrieval is a single clean matmul.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np


@dataclass
class PromptTemplate:
    """How a given model wants queries and documents wrapped before encoding."""

    query: str = "{text}"      # e.g. "query: {text}"
    document: str = "{text}"   # e.g. "passage: {text}"

    def wrap_query(self, text: str) -> str:
        return self.query.format(text=text)

    def wrap_document(self, text: str) -> str:
        return self.document.format(text=text)


# Pattern -> template. First regex match wins. Patterns are matched against the
# lowercased model id, so "intfloat/multilingual-e5-small" hits the e5 rule.
_REGISTRY: list[tuple[str, PromptTemplate]] = [
    # --- e5 family (symmetric prefixes) ---
    (r"e5", PromptTemplate(query="query: {text}", document="passage: {text}")),
    # --- nomic ---
    (r"nomic", PromptTemplate(query="search_query: {text}", document="search_document: {text}")),
    # --- bge en v1.5 (query instruction, bare passages) ---
    (r"bge.*en", PromptTemplate(
        query="Represent this sentence for searching relevant passages: {text}",
        document="{text}")),
    # --- mxbai (same instruction style as bge) ---
    (r"mxbai", PromptTemplate(
        query="Represent this sentence for searching relevant passages: {text}",
        document="{text}")),
    # --- gte-Qwen instruct style ---
    (r"gte-qwen", PromptTemplate(
        query="Instruct: Given a search query, retrieve relevant passages\nQuery: {text}",
        document="{text}")),
    # --- explicitly symmetric models: no prefix ---
    (r"(all-minilm|all-mpnet|jina|sentence-t5)", PromptTemplate()),
]

# A safe default for unknown models: no prefix. We log so the user can add a rule.
_DEFAULT = PromptTemplate()


def resolve_template(model_name: str) -> PromptTemplate:
    name = model_name.lower()
    for pattern, template in _REGISTRY:
        if re.search(pattern, name):
            return template
    return _DEFAULT


def cache_key(model_name: str) -> str:
    """Filesystem-safe key for a model id (handles the '/' that broke caching)."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", model_name)


class Encoder:
    """
    Thin wrapper over a SentenceTransformer that applies the right prompt
    template and always returns L2-normalised float32 vectors.

    The heavy import (sentence_transformers / torch) is lazy so the rest of the
    package — query gen, judging, scoring, tests — runs with zero ML deps.
    """

    def __init__(self, model_name: str, device: str | None = None, batch_size: int = 64):
        self.model_name = model_name
        self.batch_size = batch_size
        self.template = resolve_template(model_name)
        self._model = None
        self._device = device

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self._device)
        return self._model

    def _encode(self, texts: list[str]) -> np.ndarray:
        model = self._ensure_model()
        embs = model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,   # <- cosine becomes a plain dot product
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(embs, dtype=np.float32)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.template.wrap_query(t) for t in texts])

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.template.wrap_document(t) for t in texts])
