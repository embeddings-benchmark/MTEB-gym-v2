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


def load(spec: str | Path, *, split: str = "test", cap: int | None = None, seed: int = 0,
         inject_qrels_split: str | None = None) -> Corpus:
    """`spec` is an mteb task name, or a path to a directory / .jsonl."""
    path = Path(spec)
    if path.exists():
        return _load_local(path, cap, seed)
    return _load_task(str(spec), split, cap, seed, inject_qrels_split)


def _text(row) -> str:
    return ((row.get("title") or "") + " " + (row.get("text") or "")).strip()


def _load_task(name: str, split: str, cap: int | None, seed: int, inject_qrels_split: str | None) -> Corpus:
    import mteb

    task = mteb.get_tasks(tasks=[name])[0]
    task.load_data()
    subsets = task.dataset
    subset = "default" if subsets.get("default") else next(iter(subsets))   # an empty "default" falls through
    data = subsets[subset].get(split) or subsets[subset][next(iter(subsets[subset]))]
    corpus_ds = data["corpus"]
    # seeded subsample for giant corpora; select() never materializes the full set
    kept = corpus_ds.shuffle(seed=seed).select(range(cap)) if cap and len(corpus_ds) > cap else corpus_ds
    docs = {row["id"]: _text(row) for row in kept}
    if cap and inject_qrels_split:
        # Human-query arms on a capped corpus need the judged documents present, else every model scores zero.
        rel = subsets[subset].get(inject_qrels_split) or data
        need = {d for ds in rel["relevant_docs"].values() for d in ds} - set(docs)
        for row in corpus_ds:
            if row["id"] in need:
                docs[row["id"]] = _text(row)
                need.discard(row["id"])
                if not need:
                    break
        logger.info("injected qrels-judged documents into the capped corpus")

    meta = task.metadata
    langs = meta.eval_langs[subset] if isinstance(meta.eval_langs, dict) else meta.eval_langs
    metadata = meta.model_copy(update={"eval_splits": ["test"], "eval_langs": list(langs)})   # one subset, one split
    queries = {row["id"]: row["text"] for row in data["queries"]} if "queries" in data else None
    return Corpus(name=name, docs=docs, metadata=metadata, queries=queries, qrels=data.get("relevant_docs"))


def _load_local(root: Path, cap: int | None, seed: int) -> Corpus:
    import random

    import mteb

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
    # No benchmark task behind a local corpus: a generic English retrieval task's
    # metadata stands in, named after the corpus and without a task prompt.
    base = mteb.get_tasks(tasks=["NFCorpus"])[0].metadata
    metadata = base.model_copy(update={"name": root.stem, "prompt": None, "eval_splits": ["test"],
                                       "eval_langs": ["eng-Latn"],
                                       "dataset": {"path": str(root), "revision": "local"}})
    return Corpus(name=root.stem, docs=docs, metadata=metadata)
