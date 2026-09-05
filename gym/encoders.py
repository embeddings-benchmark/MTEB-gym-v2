"""
Encoding through mteb, so prompts, pooling, dtype and truncation are exactly
what an official MTEB run uses. Vectors are L2-normalised once here, so
retrieval is a single matmul.
"""

from __future__ import annotations

import re

import numpy as np


def cache_key(model_name: str) -> str:
    """Filesystem-safe key for a model id."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", model_name)


def _shim_transformers5_for_jina() -> None:
    """Import-time shims for jina's transformers-4-era remote code under
    transformers 5 (removed transformers.onnx, pruneable-heads helper, and
    PretrainedConfig defaults). No-op on transformers < 5."""
    import transformers

    if int(transformers.__version__.split(".")[0]) < 5:
        return
    import sys

    try:
        import transformers.onnx  # noqa: F401
    except ImportError:
        import types

        stub = types.ModuleType("transformers.onnx")

        class OnnxConfig:
            pass

        stub.OnnxConfig = OnnxConfig
        sys.modules["transformers.onnx"] = stub

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

    from transformers import PretrainedConfig
    for attr, default in (("is_decoder", False), ("add_cross_attention", False),
                          ("chunk_size_feed_forward", 0), ("tie_encoder_decoder", False)):
        if not hasattr(PretrainedConfig, attr):
            setattr(PretrainedConfig, attr, default)


class MTEBEncoder:
    """Encodes with mteb's own model loader: ModelMeta prompts, pooling, dtype."""

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
        return f"{self.model_name}+mteb"   # unchanged so existing embedding caches stay valid

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
        ds = Dataset.from_dict({"id": [str(i) for i in range(len(texts))], "text": texts})
        loader = create_dataloader(ds, task_metadata=self._task_meta, prompt_type=prompt_type,
                                   input_column="text", batch_size=self.batch_size)
        embs = self._model.encode(loader, task_metadata=self._task_meta, hf_split=self.split,
                                  hf_subset=self.subset, prompt_type=prompt_type)
        embs = np.asarray(embs, dtype=np.float32)
        return embs / np.clip(np.linalg.norm(embs, axis=1, keepdims=True), 1e-12, None)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, is_query=True)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, is_query=False)
