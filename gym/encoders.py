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

import logging
import re
from dataclasses import dataclass, field

import numpy as np


def _shim_transformers5_for_jina() -> None:
    """transformers 5.x compat shims for jina-embeddings-v2's legacy remote code.

    jina-bert's trust_remote_code modeling files were written against
    transformers 4.x and hit three import-time failures on 5.x. Each shim
    below activates only when the symbol is actually missing, and the whole
    function is a no-op on transformers < 5:

    1. transformers.onnx was removed, but jina-bert still imports OnnxConfig
       from it (only as a base class for an export config, never used at
       inference). Register a stub so the model loads.
    2. find_pruneable_heads_and_indices was dropped from pytorch_utils;
       jina's modeling code imports it (head pruning is never invoked at
       inference, but the import must resolve). Faithful v4 copy.
    3. PretrainedConfig no longer defaults several attributes that v4-era
       remote code reads (jina's BertSelfAttention checks config.is_decoder).
       Restore them as class-level fallbacks only where absent; these are the
       v4 defaults, so encoder-model semantics are unchanged.

    NOTE: these fix only the three import-time failures. jina-embeddings-v2
    STILL fails at encode time under transformers 5.x: its remote code builds
    tensors in __init__ (ALiBi slopes), which crashes under 5.x's meta-device
    initialization. That fourth incompatibility is NOT fixed here.
    """
    import transformers

    if int(transformers.__version__.split(".")[0]) < 5:
        return  # 4.x still ships all three symbols; nothing to shim

    import sys

    # --- shim 1: stub for the removed transformers.onnx module ---
    try:
        import transformers.onnx  # noqa: F401
    except ImportError:
        import types

        stub = types.ModuleType("transformers.onnx")

        class OnnxConfig:  # minimal stand-in for the removed base class
            pass

        stub.OnnxConfig = OnnxConfig
        sys.modules["transformers.onnx"] = stub

    # --- shim 2: backfill find_pruneable_heads_and_indices ---
    try:
        from transformers.pytorch_utils import find_pruneable_heads_and_indices  # noqa: F401
    except ImportError:
        import torch
        import transformers.pytorch_utils as _pu

        def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
            mask = torch.ones(n_heads, head_size)
            heads = set(heads) - already_pruned_heads
            for head in heads:
                head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
                mask[head] = 0
            mask = mask.view(-1).contiguous().eq(1)
            index = torch.arange(len(mask))[mask].long()
            return heads, index

        _pu.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices

    # --- shim 3: restore v4 PretrainedConfig defaults ---
    from transformers import PretrainedConfig
    for _attr, _default in (("is_decoder", False), ("add_cross_attention", False),
                            ("chunk_size_feed_forward", 0), ("tie_encoder_decoder", False)):
        if not hasattr(PretrainedConfig, _attr):
            setattr(PretrainedConfig, _attr, _default)


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
    # --- e5 instruct family (instruction on queries, BARE passages) ---
    # mteb encodes these with "Instruct: {task}\nQuery: " and
    # apply_instruction_to_passages=False; query:/passage: is wrong for them.
    # Used when MTEBEncoder is off/unavailable (it gets prompts from ModelMeta).
    (r"e5.*instruct", PromptTemplate(
        query="Instruct: Given a web search query, retrieve relevant passages "
              "that answer the query\nQuery: {text}",
        document="{text}")),
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
    logging.getLogger(__name__).warning(
        "No prompt template rule for %s; encoding without prefixes. "
        "If this model is asymmetric, add a rule to gym.encoders._REGISTRY.",
        model_name,
    )
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
            if "jina" in self.model_name.lower():
                # import-time shims only; jina still crashes at encode on
                # transformers 5.x (meta-device init) — see the shim docstring.
                _shim_transformers5_for_jina()
            from sentence_transformers import SentenceTransformer
            # trust_remote_code: nomic / gte-Qwen / jina ship custom modeling
            # code and fail to load without it (mteb passes it for these too).
            self._model = SentenceTransformer(
                self.model_name, device=self._device, trust_remote_code=True
            )
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


class MTEBEncoder:
    """Encode via mteb 2.x, so prompts, pooling and dtype come from the model's
    ModelMeta (as an official MTEB run would) instead of _REGISTRY above."""

    def __init__(self, model_name: str, task_name: str, split: str = "test",
                 subset: str = "default", batch_size: int = 32):
        self.model_name = model_name
        self.task_name = task_name
        self.split = split
        self.subset = subset
        self.batch_size = batch_size
        self._model = None
        self._task_meta = None

    @property
    def cache_name(self) -> str:
        # same model encoded with different prompts must not share a cache entry
        return f"{self.model_name}+mteb"

    def _ensure_model(self):
        if self._model is None:
            if "jina" in self.model_name.lower():
                _shim_transformers5_for_jina()
            import mteb
            self._model = mteb.get_model(self.model_name)
            self._task_meta = mteb.get_tasks(tasks=[self.task_name])[0].metadata

    def _encode(self, texts: list[str], is_query: bool) -> np.ndarray:
        self._ensure_model()
        from datasets import Dataset
        from mteb._create_dataloaders import create_dataloader  # no public equivalent yet
        from mteb.types import PromptType

        prompt_type = PromptType.query if is_query else PromptType.document
        # "id" is required by mteb's corpus standardizer; ignored for queries.
        ds = Dataset.from_dict({"id": [str(i) for i in range(len(texts))], "text": texts})
        loader = create_dataloader(ds, task_metadata=self._task_meta,
                                   prompt_type=prompt_type, input_column="text",
                                   batch_size=self.batch_size)
        embs = self._model.encode(loader, task_metadata=self._task_meta,
                                  hf_split=self.split, hf_subset=self.subset,
                                  prompt_type=prompt_type)
        embs = np.asarray(embs, dtype=np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        return embs / np.clip(norms, 1e-12, None)   # harness expects unit norm

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, is_query=True)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, is_query=False)


def make_encoder(model_name: str, task_name: str, split: str = "test",
                 use_mteb: bool = True) -> "Encoder | MTEBEncoder":
    """MTEBEncoder when mteb 2.x knows the model, else the builtin Encoder."""
    if use_mteb:
        try:
            import mteb
            import mteb._create_dataloaders  # noqa: F401 — absent on mteb 1.x
            mteb.get_model_meta(model_name)  # raises if unknown to the registry
            return MTEBEncoder(model_name, task_name, split=split)
        except Exception as e:  # noqa: BLE001
            logging.getLogger(__name__).warning(
                "mteb encoder unavailable for %s (%s); using builtin prompt registry.",
                model_name, e)
    return Encoder(model_name)
