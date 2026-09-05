"""Retrieval is mteb.evaluate on the gym task; the gym only reads mteb's prediction file."""

from __future__ import annotations

import gc
import json
import re
from dataclasses import dataclass
from pathlib import Path

from .corpus import Corpus


@dataclass
class Ranked:
    """One model's top-k for one query."""
    qid: str
    query: str
    doc_ids: list[str]
    doc_texts: list[str]


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def predict(model_name: str, task, folder: Path, *, batch_size: int = 32) -> Path:
    """Run `model_name` on `task` through mteb, writing mteb's prediction file into
    `folder`; skipped when the file already exists. Dense, sparse and
    late-interaction models all go through the same call."""
    path = folder / f"{task.metadata.name}_predictions.json"
    if not path.exists():
        import mteb

        mteb.evaluate(mteb.get_model(model_name), task, prediction_folder=folder,
                      cache=None, encode_kwargs={"batch_size": batch_size}, show_progress_bar=False)
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()   # free GPU memory between models
        except ImportError:
            pass
    return path


def revision(path: Path) -> str | None:
    """The model revision mteb recorded in its prediction file."""
    return (json.loads(path.read_text()).get("mteb_model_meta") or {}).get("revision")


def top_k(path: Path, corpus: Corpus, queries: dict[str, str], k: int) -> list[Ranked]:
    hits = json.loads(path.read_text())["default"]["test"]
    out = []
    for qid, text in queries.items():
        scores = hits.get(qid, {})
        ids = sorted(scores, key=scores.get, reverse=True)[:k]
        out.append(Ranked(qid, text, ids, [corpus.docs[d] for d in ids]))
    return out
