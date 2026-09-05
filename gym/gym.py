"""
The orchestrator: corpus -> queries -> retrieval -> pairwise judging -> rating.
Every stage is cached on disk, so a crash costs at most the current step.

    gym = Gym(cfg)          # or Gym(cfg, judge_client=..., gen_client=...)
    gym.run()               # models from cfg.models
    print(gym.leaderboard_str)
"""

from __future__ import annotations

import gc
import hashlib
import itertools
import json
import logging
import os
import random
import threading
import time
from dataclasses import asdict
from pathlib import Path

from .baselines import ALIASES, MTEBSearchRetriever
from .clients import AnthropicClient, MockClient, OpenAICompatClient
from .config import GymConfig
from .encoders import MTEBEncoder, cache_key
from .judge import Judge, Verdict
from .query_generator import Query, QueryGenerator
from .results import build_result_record, judge_instruction_metadata, result_directory
from .retrieval_harness import RetrievalHarness
from .scoring import format_leaderboard, rate

logger = logging.getLogger(__name__)

_MODEL_FAMILIES = {
    "claude": ("claude", "haiku", "sonnet", "opus"),
    "gpt": ("gpt-", "o1", "o3", "o4"),
    "qwen": ("qwen",),
    "gemini": ("gemini", "gemma"),
    "llama": ("llama",),
    "mistral": ("mistral", "mixtral"),
    "deepseek": ("deepseek",),
}


def _model_family(client) -> str | None:
    mid = str(getattr(client, "model", "")).lower()
    return next((f for f, marks in _MODEL_FAMILIES.items() if any(m in mid for m in marks)), None)


def _free_gpu() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


class Gym:
    def __init__(self, cfg: GymConfig, judge_client=None, gen_client=None):
        stale = sorted(k for k in os.environ if k.startswith("GYM_"))
        if stale:
            raise RuntimeError(
                f"{', '.join(stale)}: GYM_* environment variables are no longer read; use "
                "GymConfig fields (corpus_cap, inject_qrels_docs, encode_batch_size, judge_instruction).")
        self.cfg = cfg
        cfg.ensure_dirs()

        def build_client(model: str | None, base_url: str | None):
            if not model:
                return None
            if model == "mock":
                return MockClient(seed=cfg.seed)
            if model.lower().startswith("claude"):
                return AnthropicClient(model=model)
            return OpenAICompatClient(model=model, base_url=base_url)

        judge_client = judge_client or build_client(cfg.judge, cfg.judge_base_url)
        if judge_client is None:
            raise ValueError("No judge configured: pass judge_client=... or set GymConfig(judge='...').")
        gen_client = gen_client or build_client(cfg.generator, cfg.generator_base_url or cfg.judge_base_url)

        self.judge_client = judge_client
        self.gen_client = gen_client or judge_client
        self.harness = RetrievalHarness(cfg.cache_dir, top_k=cfg.top_k)
        self.judge = Judge(judge_client, instruction=judge_instruction_metadata(cfg)["instruction"],
                           flip_positions=cfg.flip_positions, workers=cfg.judge_workers)
        self._retr_cache: dict[str, list] = {}
        self.leaderboard_str = ""
        self._corpus_subset = "default"                # hf_subset the corpus was read from
        self._task_meta = None
        self._n_queries_generated: int | None = None   # pre-filter count, for the record

        # LLM-written queries should not be judged by the family that wrote them
        # (self-preference bias, Wataoka et al. 2024).
        gen_fam, judge_fam = _model_family(self.gen_client), _model_family(judge_client)
        if gen_fam and gen_fam == judge_fam:
            logger.warning("generator and judge are both '%s' family; use a different "
                           "generator family for clean validation.", gen_fam)

    # --------------------------------------------------------------- corpus
    def _load_local_corpus(self) -> dict[str, str]:
        """{doc_id: text} from a directory of .txt/.md files or a .jsonl with id/text."""
        root = Path(self.cfg.corpus_path)
        if root.is_file() and root.suffix == ".jsonl":
            rows = (json.loads(l) for l in root.read_text().splitlines() if l.strip())
            out = {str(r["id"]): str(r.get("text") or "") for r in rows}
        else:
            files = sorted(p for p in root.rglob("*") if p.suffix in (".txt", ".md"))
            out = {str(p.relative_to(root)): p.read_text(errors="ignore") for p in files}
        if not out:
            raise ValueError(f"no documents found in {root}")
        cap = self.cfg.corpus_cap
        if cap and len(out) > cap:
            keep = random.Random(self.cfg.seed).sample(sorted(out), cap)
            out = {k: out[k] for k in keep}
        return out

    def load_corpus(self) -> dict[str, str]:
        """{doc_id: 'title text'} for the mteb task, or for cfg.corpus_path."""
        if self.cfg.corpus_path:
            return self._load_local_corpus()
        import mteb
        task = mteb.get_tasks(tasks=[self.cfg.task_name])[0]
        task.load_data()
        subsets = task.dataset
        # an empty-but-present "default" subset falls through to the first real one
        self._corpus_subset = "default" if subsets.get("default") else next(iter(subsets))
        subset = subsets[self._corpus_subset]
        split = subset.get(self.cfg.corpus_split) or subset[next(iter(subset))]
        corpus_ds = split["corpus"]
        cap = self.cfg.corpus_cap
        # seeded subsample for giant corpora; select() never materializes the full set
        capped = corpus_ds.shuffle(seed=self.cfg.seed).select(range(cap)) if cap and len(corpus_ds) > cap else corpus_ds

        def text(row):
            return ((row.get("title") or "") + " " + (row.get("text") or "")).strip()

        out = {row["id"]: text(row) for row in capped}
        # Human-query arms on a capped corpus need the qrels-judged documents present,
        # else every model scores zero; inject_qrels_docs names the split to take them from.
        if cap and self.cfg.inject_qrels_docs:
            rel = subset.get(self.cfg.inject_qrels_docs) or subset[next(iter(subset))]
            need = {d for docs in rel["relevant_docs"].values() for d in docs} - set(out)
            for row in corpus_ds:
                if row["id"] in need:
                    out[row["id"]] = text(row)
                    need.discard(row["id"])
                    if not need:
                        break
            logger.info("injected qrels-judged documents into the capped corpus")
        return out

    # --------------------------------------------------------------- queries
    def _controls_key(self) -> str:
        """Cache-key suffix for corpus controls; empty when uncapped so old keys hold."""
        c = self.cfg
        if not c.corpus_cap:
            return ""
        return f"|cap={c.corpus_cap}" + (f"|inject_qrels={c.inject_qrels_docs}" if c.inject_qrels_docs else "")

    def _query_config_hash(self) -> str:
        """Everything that changes which queries get generated."""
        c = self.cfg
        gen_id = getattr(self.gen_client, "model", type(self.gen_client).__name__)
        key = (f"{gen_id}|{c.task_name}|{c.n_queries}|{c.seed}|{c.docs_per_query}|"
               f"{c.gen_overshoot}|{c.filter_queries}|{c.filter_min_score}|"
               f"{c.dedup_threshold}|{c.min_query_chars}|{c.max_query_chars}")
        return hashlib.sha256((key + self._controls_key()).encode()).hexdigest()[:8]

    def _queries_path(self) -> Path:
        return self.cfg.output_dir / f"queries_{self.cfg.task_name}_{self._query_config_hash()}.json"

    def get_queries(self, corpus: dict[str, str]) -> list[Query]:
        path = self._queries_path()
        meta_path = path.with_suffix(".meta.json")
        if path.exists():
            data = json.loads(path.read_text())
            if data:
                self._n_queries_generated = None
                if meta_path.exists():
                    self._n_queries_generated = json.loads(meta_path.read_text()).get("n_queries_generated")
                return [Query(**q) for q in data]
            logger.warning("empty query cache at %s (crash artifact); regenerating", path)
            path.unlink()
        gen = QueryGenerator(self.gen_client, self.cfg)
        queries = gen.run(corpus)
        if not queries:
            raise RuntimeError(f"query generation returned 0 queries for {self.cfg.task_name}: "
                               "the generator endpoint is likely dead or the filter rejected everything.")
        path.write_text(json.dumps([asdict(q) for q in queries], indent=2))
        self._n_queries_generated = gen.last_generated_count
        meta_path.write_text(json.dumps({"n_queries_generated": self._n_queries_generated}, indent=2))
        return queries

    # --------------------------------------------------------------- retrieval
    def _task_metadata(self):
        """mteb TaskMetadata (prompts, language) for encoding and search. A local
        corpus has no task, so a generic English retrieval task's metadata stands
        in, named after the run and without a prompt."""
        if self._task_meta is None:
            import mteb
            if self.cfg.corpus_path:
                meta = mteb.get_tasks(tasks=["NFCorpus"])[0].metadata
                self._task_meta = meta.model_copy(update={"name": self.cfg.task_name, "prompt": None})
            else:
                self._task_meta = mteb.get_tasks(tasks=[self.cfg.task_name])[0].metadata
        return self._task_meta

    def _retrieve(self, model_name: str, corpus, queries) -> list:
        if model_name in self._retr_cache:
            return self._retr_cache[model_name]
        import mteb
        mteb_name = ALIASES.get(model_name, model_name)
        common = dict(split=self.cfg.corpus_split, subset=self._corpus_subset)
        if "dense" in (mteb.get_model_meta(mteb_name).model_type or ["dense"]):
            enc = MTEBEncoder(mteb_name, self._task_metadata(),
                              batch_size=self.cfg.encode_batch_size, **common)
            retr = self.harness.retrieve(enc, corpus, queries)
            del enc
            _free_gpu()   # 7B-class encoders otherwise accumulate on one GPU and OOM mid-tournament
        else:   # bm25, colbert, learned sparse: mteb's index/search, no embedding cache
            retr = MTEBSearchRetriever(mteb_name, self._task_metadata(), top_k=self.cfg.top_k,
                                       **common).retrieve(corpus, queries)
        self._retr_cache[model_name] = retr
        return retr

    # --------------------------------------------------------------- matchup
    def _verdict_config_hash(self, queries) -> str:
        """Everything that changes verdicts for a pair: judge model, resolved judge
        prompt (task instruction included), top_k, flipping, the query set."""
        judge_id = getattr(self.judge_client, "model", type(self.judge_client).__name__)
        prompt_sig = hashlib.sha256(self.judge.system.encode()).hexdigest()[:8]
        qsig = "||".join(f"{q.qid}:{q.text}" for q in queries)
        key = f"{judge_id}|{prompt_sig}|{self.cfg.top_k}|{self.cfg.flip_positions}|{qsig}"
        return hashlib.sha256((key + self._controls_key()).encode()).hexdigest()[:8]

    def _matchup_path(self, a, b, queries) -> Path:
        return self.cfg.output_dir / (
            f"verdicts_{cache_key(a)}__{cache_key(b)}_{self._verdict_config_hash(queries)}.json")

    def matchup(self, model_a, model_b, corpus, queries) -> list[Verdict]:
        path = self._matchup_path(model_a, model_b, queries)
        if path.exists():
            data = json.loads(path.read_text())
            logger.info("%s vs %s: %d cached verdicts", model_a, model_b, len(data))
            return [Verdict(**v) for v in data]

        # Verdicts stream to a JSONL sidecar as they complete; a crash mid-pair
        # costs only the in-flight queries and the rerun resumes from it.
        jsonl = path.with_suffix(".jsonl")
        done: dict[str, Verdict] = {}
        if jsonl.exists():
            for line in jsonl.read_text().splitlines():
                if line.strip():
                    v = Verdict(**json.loads(line))
                    done[v.qid] = v
            if done:
                logger.info("%s vs %s: resuming, %d verdicts on disk", model_a, model_b, len(done))

        t0 = time.time()
        ra = self._retrieve(model_a, corpus, queries)
        rb = self._retrieve(model_b, corpus, queries)
        todo = [r for r in ra if r.qid not in done]
        if todo:
            write_lock = threading.Lock()

            def _persist(v: Verdict) -> None:
                with write_lock, jsonl.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(asdict(v)) + "\n")

            new = self.judge.judge_all(todo, rb, model_a, model_b, on_verdict=_persist)
            done.update({v.qid: v for v in new})

        order = {q.qid: i for i, q in enumerate(queries)}
        verdicts = sorted(done.values(), key=lambda v: order.get(v.qid, 1 << 30))
        path.write_text(json.dumps([asdict(v) for v in verdicts], indent=2))
        logger.info("%s vs %s: %d verdicts in %.0fs", model_a, model_b, len(verdicts), time.time() - t0)
        return verdicts

    # --------------------------------------------------------------- tournament
    def tournament(self, models: list[str], corpus, queries, *,
                   output_folder: str | Path | None = None,
                   run_started_at: float | None = None):
        started = time.time() if run_started_at is None else run_started_at
        pairs = list(itertools.combinations(models, 2))
        verdicts: list[Verdict] = []
        for i, (a, b) in enumerate(pairs, 1):
            logger.info("pair %d/%d: %s vs %s", i, len(pairs), a, b)
            verdicts.extend(self.matchup(a, b, corpus, queries))

        ratings = rate(verdicts, base=self.cfg.elo_base, scale=self.cfg.elo_scale,
                       bootstrap=self.cfg.bootstrap_samples, seed=self.cfg.seed)
        self.leaderboard_str = format_leaderboard(ratings)

        record, experiment_config = build_result_record(
            cfg=self.cfg, judge_client=self.judge_client, generator_client=self.gen_client,
            models=models, ratings=ratings, verdicts=verdicts,
            n_queries_generated=self._n_queries_generated, n_queries=len(queries),
            hf_subset=self._corpus_subset, evaluation_time=time.time() - started,
            judge_system=self.judge.system,
        )
        root = Path(output_folder) if output_folder is not None else self.cfg.output_dir
        result_dir = result_directory(root, experiment_config["judge_model"],
                                      experiment_config["generator_model"], experiment_config)
        result_dir.mkdir(parents=True, exist_ok=True)
        self.result_path = result_dir / f"{self.cfg.task_name}.json"
        self.result_path.write_text(json.dumps(record, indent=2))
        return ratings

    def run(self, models: list[str] | None = None, *, output_folder: str | Path | None = None):
        models = list(models if models is not None else self.cfg.models)
        if not models:
            raise ValueError("No models: pass models to Gym.run(...) or set GymConfig(models=[...]).")
        started = time.time()
        corpus = self.load_corpus()
        queries = self.get_queries(corpus)
        return self.tournament(models, corpus, queries, output_folder=output_folder, run_started_at=started)
