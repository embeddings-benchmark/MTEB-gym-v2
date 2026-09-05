"""
The gym hands mteb a retrieval task and lets mteb run every model on it. The
task is the corpus plus the gym's queries; the qrels are each synthetic query's
seed documents (a diagnostic, not a label) or, in the human-query arm, the
task's own labels.
"""

from __future__ import annotations

from .corpus import Corpus
from .queries import Query


def build(corpus: Corpus, queries: list[Query] | None):
    """An mteb AbsTaskRetrieval over `corpus`. `queries=None` uses the task's own
    queries and qrels (human-query arm)."""
    from datasets import Dataset
    from mteb.abstasks.retrieval import AbsTaskRetrieval

    if queries is None:
        if not corpus.queries or not corpus.qrels:
            raise ValueError(f"{corpus.name} has no queries/qrels for a human-query arm")
        texts, qrels = corpus.queries, corpus.qrels
    else:
        texts = {q.qid: q.text for q in queries}
        qrels = {q.qid: {d: 1 for d in q.seed_doc_ids} for q in queries}

    ids = list(corpus.docs)
    data = {"default": {"test": {
        "corpus": Dataset.from_dict({"id": ids, "title": [""] * len(ids), "text": [corpus.docs[i] for i in ids]}),
        "queries": Dataset.from_dict({"id": list(texts), "text": list(texts.values())}),
        "relevant_docs": qrels,
        "top_ranked": None,
    }}}

    def load_data(self, **kwargs):
        self.dataset = data
        self.data_loaded = True

    return type("GymTask", (AbsTaskRetrieval,), {"metadata": corpus.metadata, "load_data": load_data})()
