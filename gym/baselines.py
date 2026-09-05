"""
Non-dense entrants (BM25, ColBERT, learned sparse) run through mteb's search
protocol (index, then search), so the gym scores the same implementations mteb
does. Short names map to mteb model ids; anything mteb marks non-dense works.
"""

from __future__ import annotations

from .retrieval_harness import Retrieved

ALIASES = {"bm25": "mteb/baseline-bm25s", "colbert": "colbert-ir/colbertv2.0"}


class MTEBSearchRetriever:
    """Retrieves with an mteb model that implements index() / search()."""

    def __init__(self, model_name: str, task_metadata, split: str = "test",
                 subset: str = "default", top_k: int = 10):
        self.model_name = ALIASES.get(model_name, model_name)
        self.task_metadata = task_metadata
        self.split = split
        self.subset = subset
        self.top_k = top_k

    def retrieve(self, corpus: dict[str, str], queries: list) -> list[Retrieved]:
        import mteb
        from datasets import Dataset

        model = mteb.get_model(self.model_name)
        ids = list(corpus)
        docs = Dataset.from_dict({"id": ids, "title": [""] * len(ids), "text": [corpus[i] for i in ids]})
        qs = Dataset.from_dict({"id": [q.qid for q in queries], "text": [q.text for q in queries]})
        kw = dict(task_metadata=self.task_metadata, hf_split=self.split,
                  hf_subset=self.subset, encode_kwargs={})
        model.index(docs, **kw)
        hits = model.search(qs, top_k=self.top_k, **kw)
        out = []
        for q in queries:
            ranked = sorted(hits.get(q.qid, {}).items(), key=lambda kv: -kv[1])[:self.top_k]
            out.append(Retrieved(qid=q.qid, query=q.text, doc_ids=[d for d, _ in ranked],
                                 doc_texts=[corpus[d] for d, _ in ranked]))
        return out
