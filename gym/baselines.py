"""
BM25 baseline.

Rohan flagged BM25 as an easy, high-value addition: a lexical anchor every dense
model should beat. If a dense model loses to BM25 in the gym, that's a strong
smell test that something is wrong (bad prefixes, broken encoding, junk queries).

Implements Okapi BM25 with no heavy deps (pure numpy + a tokenizer regex), so it
behaves like just another "model" in a tournament.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from .retrieval_harness import Retrieved

_TOKEN = re.compile(r"[A-Za-z0-9]+")


def _tok(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Retriever:
    """Quacks like an Encoder+Harness pair: .name and .retrieve(corpus, queries)."""

    model_name = "bm25"

    def __init__(self, top_k: int = 10, k1: float = 1.5, b: float = 0.75):
        self.top_k = top_k
        self.k1 = k1
        self.b = b

    def retrieve(self, corpus: dict[str, str], queries: list) -> list[Retrieved]:
        doc_ids = list(corpus.keys())
        docs = [_tok(t) for t in corpus.values()]
        texts = list(corpus.values())
        N = len(docs)
        lengths = np.array([len(d) for d in docs], dtype=np.float32)
        avgdl = float(lengths.mean()) if N else 1.0

        df: Counter = Counter()
        for d in docs:
            for term in set(d):
                df[term] += 1
        idf = {t: math.log(1 + (N - n + 0.5) / (n + 0.5)) for t, n in df.items()}
        tfs = [Counter(d) for d in docs]

        out: list[Retrieved] = []
        k = min(self.top_k, N)
        for q in queries:
            qterms = _tok(q.text)
            scores = np.zeros(N, dtype=np.float32)
            for term in qterms:
                if term not in idf:
                    continue
                w = idf[term]
                for di in range(N):
                    f = tfs[di].get(term, 0)
                    if f == 0:
                        continue
                    denom = f + self.k1 * (1 - self.b + self.b * lengths[di] / avgdl)
                    scores[di] += w * (f * (self.k1 + 1)) / denom
            idx = np.argpartition(-scores, k - 1)[:k]
            idx = idx[np.argsort(-scores[idx])]
            out.append(Retrieved(
                qid=q.qid, query=q.text,
                doc_ids=[doc_ids[j] for j in idx],
                doc_texts=[texts[j] for j in idx],
            ))
        return out


# --------------------------------------------------------------------------- #
# ColBERT (late interaction) baseline
# --------------------------------------------------------------------------- #
#
# ColBERT is the non-dense, non-lexical third axis: every token gets its own
# vector and relevance is the sum of per-query-token max similarities (MaxSim),
# not a single dot product. It is the natural stress test for the gym's "does
# it under-rank anything that isn't one dense vector?" question that bm25 first
# raised.
#
# pylate (1.6.0) gives us the official colbert-ir/colbertv2.0 checkpoint plus a
# MaxSim scorer. For a few-thousand-doc corpus (NFCorpus is 3633) we skip the
# Voyager/PLAID index entirely and brute-force MaxSim over every document.
#
# We pad the corpus token embeddings ONCE into a single (N, max_doc_tok, dim)
# tensor + mask, then score queries in small chunks against it. pylate's own
# rank.rerank() re-pads the whole corpus inside its per-query loop (rebuilding
# the full padded tensor N_queries times); scoring against a corpus padded once
# is the same MaxSim math without that quadratic repad. We keep the query chunk
# small so the (chunk, N, q_tok, d_tok) einsum intermediate stays bounded.
#
# Heavy imports (pylate / torch) stay lazy so this module keeps importing with
# zero ML deps, exactly like BM25Retriever above.


class ColBERTRetriever:
    """Late-interaction retriever. Quacks like an Encoder+Harness pair:
    `.model_name` and `.retrieve(corpus, queries) -> list[Retrieved]`.

    Encodes the whole corpus once into per-document token-embedding tensors,
    then scores each query against every document with MaxSim (brute force, no
    ANN index — fine for the few-thousand-doc tasks the gym runs on).
    """

    model_name = "colbert"

    def __init__(self, top_k: int = 10,
                 checkpoint: str = "colbert-ir/colbertv2.0",
                 doc_batch_size: int = 256,
                 query_batch_size: int = 128,
                 query_score_chunk: int = 16,
                 device: str | None = None):
        self.top_k = top_k
        self.checkpoint = checkpoint
        self.doc_batch_size = doc_batch_size
        self.query_batch_size = query_batch_size
        self.query_score_chunk = query_score_chunk
        self.device = device

    def _resolve_device(self) -> str:
        if self.device is not None:
            return self.device
        try:
            import torch  # noqa: PLC0415
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001
            return "cpu"

    def _load_model(self, device: str):
        # Lazy import: keeps pylate/torch out of the import path for the
        # dep-free parts of the package (query gen, judging, scoring, tests).
        from pylate import models  # noqa: PLC0415
        return models.ColBERT(model_name_or_path=self.checkpoint, device=device)

    @staticmethod
    def _pad(token_tensors: list, device: str):
        """Pad a list of (n_tok, dim) tensors -> ((B, max_tok, dim), (B, max_tok))."""
        import torch  # noqa: PLC0415
        padded = torch.nn.utils.rnn.pad_sequence(
            token_tensors, batch_first=True, padding_value=0.0
        )
        lengths = torch.tensor([t.shape[0] for t in token_tensors], device=padded.device)
        max_tok = padded.shape[1]
        mask = (torch.arange(max_tok, device=padded.device)[None, :] < lengths[:, None]).float()
        return padded.to(device), mask.to(device)

    def retrieve(self, corpus: dict[str, str], queries: list) -> list[Retrieved]:
        import torch  # noqa: PLC0415
        from pylate.scores import colbert_scores  # noqa: PLC0415

        device = self._resolve_device()
        model = self._load_model(device)

        doc_ids = list(corpus.keys())
        doc_texts = list(corpus.values())

        # (N) list of (n_tok, dim) tensors -> padded once into one corpus tensor.
        doc_embs = model.encode(
            doc_texts, is_query=False, batch_size=self.doc_batch_size,
            show_progress_bar=False, convert_to_tensor=True,
        )
        doc_pad, doc_mask = self._pad(list(doc_embs), device)   # (N, Dtok, dim), (N, Dtok)

        q_embs = model.encode(
            [q.text for q in queries], is_query=True, batch_size=self.query_batch_size,
            show_progress_bar=False, convert_to_tensor=True,
        )

        N = len(doc_ids)
        k = min(self.top_k, N)
        out: list[Retrieved] = []
        chunk = max(1, self.query_score_chunk)

        with torch.no_grad():
            for start in range(0, len(queries), chunk):
                q_slice = list(q_embs[start:start + chunk])
                q_pad, q_mask = self._pad(q_slice, device)       # (b, Qtok, dim), (b, Qtok)
                # (b, N) MaxSim scores: every query in the chunk vs the full corpus.
                scores = colbert_scores(
                    queries_embeddings=q_pad,
                    documents_embeddings=doc_pad,
                    queries_mask=q_mask,
                    documents_mask=doc_mask,
                )
                topk = torch.topk(scores, k=k, dim=1).indices.cpu().tolist()
                for row, idxs in enumerate(topk):
                    q = queries[start + row]
                    out.append(Retrieved(
                        qid=q.qid, query=q.text,
                        doc_ids=[doc_ids[j] for j in idxs],
                        doc_texts=[doc_texts[j] for j in idxs],
                    ))
        return out
