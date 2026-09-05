"""
A corpus is {doc_id: "title text"} plus the mteb task metadata that tells models
how to encode it (prompts, language). Loaded from an mteb retrieval task by name,
or from a local directory of .txt/.md files or a .jsonl with id/text fields.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Corpus:
    name: str
    docs: dict[str, str]
    metadata: object                                   # mteb TaskMetadata
    queries: dict[str, str] | None = None              # the task's own queries (human-query arm)
    qrels: dict[str, dict[str, int]] | None = None     # the task's own relevance labels
    fingerprint: str = field(init=False)               # content hash over ids and texts, in order

    def __post_init__(self):
        h = hashlib.sha256()
        for did, text in self.docs.items():
            h.update(did.encode()); h.update(b"\x00"); h.update(text.encode()); h.update(b"\x01")
        self.fingerprint = h.hexdigest()[:12]


def load(spec: str | Path, *, cap: int | None = None, seed: int = 0, keep_judged: bool = False) -> Corpus:
    """`spec` is an mteb task name, or a path to a directory / .jsonl. `cap` subsamples
    giant corpora; `keep_judged` keeps the task's qrels-judged documents in the
    subsample (human-query arm)."""
    path = Path(spec)
    if path.exists():
        return _load_local(path, cap, seed)
    return _load_task(str(spec), cap, seed, keep_judged)


def _text(row) -> str:
    return ((row.get("title") or "") + " " + (row.get("text") or "")).strip()


def _load_task(name: str, cap: int | None, seed: int, keep_judged: bool) -> Corpus:
    import mteb

    task = mteb.get_tasks(tasks=[name])[0]
    task.load_data()
    subsets = task.dataset
    subset = "default" if subsets.get("default") else next(iter(subsets))   # an empty "default" falls through
    data = subsets[subset].get("test") or subsets[subset][next(iter(subsets[subset]))]
    corpus_ds = data["corpus"]
    # seeded subsample for giant corpora; select() never materializes the full set
    kept = corpus_ds.shuffle(seed=seed).select(range(cap)) if cap and len(corpus_ds) > cap else corpus_ds
    docs = {row["id"]: _text(row) for row in kept}
    if cap and keep_judged:
        # a capped corpus must keep the judged documents, else every model scores zero on the real queries
        need = {d for ds in data["relevant_docs"].values() for d in ds} - set(docs)
        for row in corpus_ds:
            if row["id"] in need:
                docs[row["id"]] = _text(row)
                need.discard(row["id"])
                if not need:
                    break
        logger.info("kept the qrels-judged documents in the capped corpus")

    meta = task.metadata
    langs = meta.eval_langs[subset] if isinstance(meta.eval_langs, dict) else meta.eval_langs
    metadata = meta.model_copy(update={"eval_splits": ["test"], "eval_langs": list(langs)})   # one subset, one split
    queries = {row["id"]: row["text"] for row in data["queries"]} if "queries" in data else None
    return Corpus(name=name, docs=docs, metadata=metadata, queries=queries, qrels=data.get("relevant_docs"))


def _load_local(root: Path, cap: int | None, seed: int) -> Corpus:
    import random

    from mteb.abstasks.task_metadata import TaskMetadata

    if root.is_file() and root.suffix == ".jsonl":
        rows = (json.loads(l) for l in root.read_text().splitlines() if l.strip())
        docs = {str(r["id"]): str(r.get("text") or "") for r in rows}
    else:
        files = sorted(p for p in root.rglob("*") if p.suffix in (".txt", ".md"))
        docs = {str(p.relative_to(root)): p.read_text(errors="ignore") for p in files}
    if not docs:
        raise ValueError(f"no documents found in {root}")
    if cap and len(docs) > cap:
        keep = random.Random(seed).sample(sorted(docs), cap)
        docs = {k: docs[k] for k in keep}
    corpus = Corpus(name=root.stem, docs=docs, metadata=None)
    # no benchmark task behind a local corpus: minimal metadata, no task prompt
    corpus.metadata = TaskMetadata(
        name=root.stem, description=f"Local corpus {root}", type="Retrieval", category="t2t",
        modalities=["text"], eval_splits=["test"], eval_langs=["eng-Latn"], main_score="ndcg_at_10",
        dataset={"path": str(root), "revision": corpus.fingerprint},
    )
    return corpus
