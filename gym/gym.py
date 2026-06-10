"""
The orchestrator. One object that runs the whole pipeline and caches every
stage so a crash (or your laptop lid) never costs more than the current step.

    gym = Gym(cfg, judge_client=..., gen_client=...)
    corpus = gym.load_corpus()
    queries = gym.get_queries(corpus)          # generate + filter (cached)
    ratings = gym.tournament(models, corpus, queries)
    print(gym.leaderboard_str)
"""

from __future__ import annotations

import itertools
import json
from dataclasses import asdict
from pathlib import Path

from .baselines import BM25Retriever
from .config import GymConfig
from .encoders import Encoder
from .judge import Judge, Verdict
from .query_generator import Query, QueryGenerator
from .retrieval_harness import RetrievalHarness
from .scoring import format_leaderboard, rate


class Gym:
    def __init__(self, cfg: GymConfig, judge_client, gen_client=None):
        self.cfg = cfg
        cfg.ensure_dirs()
        self.judge_client = judge_client
        self.gen_client = gen_client or judge_client
        self.harness = RetrievalHarness(cfg.cache_dir, top_k=cfg.top_k)
        self.judge = Judge(judge_client, flip_positions=cfg.flip_positions,
                           note=cfg.judge_batch_note)
        self._retr_cache: dict[str, list] = {}
        self.leaderboard_str = ""

    # --------------------------------------------------------------- corpus
    def load_corpus(self) -> dict[str, str]:
        """Load an MTEB task corpus as {doc_id: 'title text'}.

        Handles both mteb 1.x (task.corpus[split] dict) and mteb 2.x, where
        load_data() populates task.dataset[subset][split]["corpus"] as a HF
        Dataset with id/title/text columns and task.corpus no longer exists.
        """
        import mteb
        task = mteb.get_tasks(tasks=[self.cfg.task_name])[0]
        task.load_data()

        legacy = getattr(task, "corpus", None)
        if legacy:                                         # mteb 1.x
            corpus = legacy.get(self.cfg.corpus_split) or legacy[next(iter(legacy))]
            out: dict[str, str] = {}
            for did, doc in corpus.items():
                if isinstance(doc, dict):                  # {title, text}
                    out[did] = (doc.get("title", "") + " " + doc.get("text", "")).strip()
                else:                                      # plain string
                    out[did] = str(doc).strip()
            return out

        # mteb 2.x
        subsets = task.dataset
        subset = subsets.get("default") or subsets[next(iter(subsets))]
        split_data = subset.get(self.cfg.corpus_split) or subset[next(iter(subset))]
        corpus_ds = split_data["corpus"]
        return {
            row["id"]: ((row.get("title") or "") + " " + (row.get("text") or "")).strip()
            for row in corpus_ds
        }

    # --------------------------------------------------------------- queries
    def _queries_path(self) -> Path:
        return self.cfg.output_dir / f"queries_{self.cfg.task_name}.json"

    def get_queries(self, corpus: dict[str, str]) -> list[Query]:
        path = self._queries_path()
        if path.exists():
            data = json.loads(path.read_text())
            return [Query(**q) for q in data]
        gen = QueryGenerator(self.gen_client, self.cfg)
        queries = gen.run(corpus)
        path.write_text(json.dumps([asdict(q) for q in queries], indent=2))
        return queries

    # --------------------------------------------------------------- retrieval
    def _retrieve(self, model_name: str, corpus, queries) -> list:
        if model_name in self._retr_cache:
            return self._retr_cache[model_name]
        if model_name == "bm25":
            retr = BM25Retriever(top_k=self.cfg.top_k).retrieve(corpus, queries)
        else:
            enc = Encoder(model_name)
            retr = self.harness.retrieve(enc, corpus, queries)
        self._retr_cache[model_name] = retr
        return retr

    # --------------------------------------------------------------- matchup
    def _matchup_path(self, a, b) -> Path:
        from .encoders import cache_key
        return self.cfg.output_dir / f"verdicts_{cache_key(a)}__{cache_key(b)}.json"

    def matchup(self, model_a, model_b, corpus, queries) -> list[Verdict]:
        path = self._matchup_path(model_a, model_b)
        if path.exists():
            data = json.loads(path.read_text())
            return [Verdict(**v) for v in data]
        ra = self._retrieve(model_a, corpus, queries)
        rb = self._retrieve(model_b, corpus, queries)
        verdicts = self.judge.judge_all(ra, rb, model_a, model_b)
        path.write_text(json.dumps([asdict(v) for v in verdicts], indent=2))
        return verdicts

    # --------------------------------------------------------------- tournament
    def tournament(self, models: list[str], corpus, queries):
        all_verdicts: list[Verdict] = []
        for a, b in itertools.combinations(models, 2):
            all_verdicts.extend(self.matchup(a, b, corpus, queries))

        ratings = rate(all_verdicts, method=self.cfg.method,
                       base=self.cfg.elo_base, scale=self.cfg.elo_scale,
                       bootstrap=self.cfg.bootstrap_samples, seed=self.cfg.seed)
        self.leaderboard_str = format_leaderboard(ratings)

        out = {
            "task": self.cfg.task_name,
            "n_queries": len(queries),
            "method": self.cfg.method,
            "a_first_rate": self.judge.a_first_rate,   # position-bias diagnostic
            "models": [
                {"name": m.name, "rating": m.rating, "ci_low": m.ci_low,
                 "ci_high": m.ci_high, "wins": m.wins, "losses": m.losses,
                 "ties": m.ties, "n": m.n}
                for m in ratings
            ],
        }
        (self.cfg.output_dir / "leaderboard.json").write_text(json.dumps(out, indent=2))
        return ratings

    # --------------------------------------------------------------- convenience
    def run(self, models: list[str]):
        corpus = self.load_corpus()
        queries = self.get_queries(corpus)
        return self.tournament(models, corpus, queries)
