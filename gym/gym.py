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

import hashlib
import itertools
import json
import logging
import os
import threading
import time
from dataclasses import asdict
from pathlib import Path

from .baselines import BM25Retriever, ColBERTRetriever
from .clients import AnthropicClient, MockClient, OpenAICompatClient
from .config import GymConfig
from .encoders import make_encoder
from .judge import Judge, Verdict
from .query_generator import Query, QueryGenerator
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
    """Best-effort family of the model behind a client, None if unknown."""
    mid = str(getattr(client, "model", "")).lower()
    for family, markers in _MODEL_FAMILIES.items():
        if any(m in mid for m in markers):
            return family
    return None


def _a_first_rate_from(verdicts: list[Verdict]) -> float | None:
    """Position-bias diagnostic recomputed from persisted per-order winners.

    Pure function over verdict files, so cached or resumed runs report the
    true rate instead of whatever the in-process judge counters happened to
    see. raw[0] is the A-first order (first-shown win = "A"); raw[1] is the
    flipped order (first-shown win = "A" there too). "identical" markers from
    the short-circuit carry no position information and are skipped.
    """
    first = decisive = 0
    for v in verdicts:
        for w in v.raw:
            if w in ("A", "B"):
                decisive += 1
                if w == "A":
                    first += 1
    return (first / decisive) if decisive else None


def _parse_failure_rate_from(verdicts: list[Verdict]) -> float | None:
    """Judge parse-failure rate recomputed from persisted per-order flags.

    Like _a_first_rate_from, a pure function over verdict files, so a cached or
    resumed run reports the true rate instead of the in-process judge counter
    (which only saw whatever queries this run actually judged). Verdicts written
    before parsed_ok existed carry an empty list and are skipped; "identical"
    short-circuits make no judge call and contribute nothing.
    """
    asks = failures = 0
    for v in verdicts:
        for ok in v.parsed_ok:
            asks += 1
            if not ok:
                failures += 1
    return (failures / asks) if asks else None


class Gym:
    def __init__(self, cfg: GymConfig, judge_client=None, gen_client=None):
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

        if judge_client is None:
            judge_client = build_client(cfg.judge, cfg.judge_base_url)
            if judge_client is None:
                raise ValueError(
                    "No judge configured. Pass judge_client=... "
                    "or set GymConfig(judge='...')."
                )

        if gen_client is None and cfg.generator:
            gen_client = build_client(
                cfg.generator,
                cfg.generator_base_url or cfg.judge_base_url,
            )

        self.judge_client = judge_client
        self.gen_client = gen_client or judge_client
        self.harness = RetrievalHarness(cfg.cache_dir, top_k=cfg.top_k)
        self.judge = Judge(judge_client, task_name=cfg.task_name,
                           flip_positions=cfg.flip_positions,
                           note=cfg.judge_batch_note, workers=cfg.judge_workers)
        self._retr_cache: dict[str, list] = {}
        self.leaderboard_str = ""
        # Resolved in load_corpus for mteb 2.x multi-subset tasks so the encoder
        # is told the same hf_subset the corpus was actually read from (#9).
        self._corpus_subset = "default"
        # Exact pre-filter generated-query count for result provenance.
        # None for legacy query caches that predate the metadata sidecar.
        self._n_queries_generated: int | None = None

        # Self-preference bias tracks the judge's own perplexity over the text
        # it rates (Wataoka et al. 2024), so LLM-generated queries should not
        # be judged by the same model family that wrote them.
        gen_fam, judge_fam = _model_family(self.gen_client), _model_family(judge_client)
        if gen_fam and gen_fam == judge_fam:
            logger.warning(
                "query generator and judge are both '%s' family; self-preference "
                "bias can favor models that retrieve generator-flavored text. "
                "Pass a gen_client from a different family for clean validation.",
                gen_fam,
            )

    # --------------------------------------------------------------- corpus
    def _corpus_cap(self) -> int | None:
        """Resolve corpus cap while preserving legacy environment behavior."""
        if self.cfg.corpus_cap is not None:
            return self.cfg.corpus_cap
        raw = os.environ.get("GYM_MAX_CORPUS_DOCS")
        return int(raw) if raw else None

    def _inject_qrels_docs(self) -> str | None:
        """Resolve qrels-injection split while preserving legacy env behavior."""
        if self.cfg.inject_qrels_docs is not None:
            return self.cfg.inject_qrels_docs
        return os.environ.get("GYM_INJECT_QRELS_DOCS")

    def load_corpus(self) -> dict[str, str]:
        """Load an MTEB task corpus as {doc_id: 'title text'}.

        Handles both mteb 1.x (task.corpus[split] dict) and mteb 2.x, where
        load_data() populates task.dataset[subset][split]["corpus"] as a HF
        Dataset with id/title/text columns and task.corpus no longer exists.
        """
        import mteb, random
        task = mteb.get_tasks(tasks=[self.cfg.task_name])[0]
        task.load_data()

        # Opt-in corpus cap (default OFF) so giant BEIR corpora (MSMARCO/DBPedia/NQ/HotpotQA/FEVER)
        # fit on one GPU. Seeded by cfg.seed for reproducibility. The gym generates its own queries
        # from whatever corpus this returns, so a seeded subsample is self-consistent; validation vs
        # the full-corpus official nDCG@10 anchor is an approximation, reported as a subsampled arm.
        _cap = self._corpus_cap()

        legacy = getattr(task, "corpus", None)
        if legacy:                                         # mteb 1.x
            corpus = legacy.get(self.cfg.corpus_split) or legacy[next(iter(legacy))]
            # Multilingual / multi-domain 1.x tasks (e.g. CUREv1) nest one level
            # deeper: corpus[lang] -> {subset_name: {doc_id: doc}}. Iterating that
            # directly turns subset NAMES into doc ids (CUREv1 loaded as 11
            # repr-text "docs" and every retrieval tied). Detect the extra level
            # (first value is itself a mapping of docs, not a doc) and descend,
            # preferring the 'all' union, then the split name, else merging.
            def _is_doc(x):
                return isinstance(x, str) or (isinstance(x, dict) and ("text" in x or "title" in x))
            if isinstance(corpus, dict) and corpus and not _is_doc(next(iter(corpus.values()))):
                inner = corpus.get("all") or corpus.get(self.cfg.corpus_split)
                if inner is None:
                    inner = {}
                    for _sub in corpus.values():
                        if isinstance(_sub, dict):
                            inner.update(_sub)
                corpus = inner
            out: dict[str, str] = {}
            for did, doc in corpus.items():
                if isinstance(doc, dict):                  # {title, text}
                    out[did] = (doc.get("title", "") + " " + doc.get("text", "")).strip()
                else:                                      # plain string
                    out[did] = str(doc).strip()
            if _cap and len(out) > _cap:
                keys = sorted(out)
                random.Random(self.cfg.seed).shuffle(keys)
                out = {k: out[k] for k in keys[:_cap]}
            return out

        # mteb 2.x
        subsets = task.dataset
        # Remember which subset KEY we read, so make_encoder gets the same
        # hf_subset (a hardcoded "default" mis-encodes multi-subset tasks) (#9).
        # Truthiness, not mere membership, to match the inner split line below:
        # an empty-but-present "default" subset must fall through to the first
        # real subset (else this picks {} and the next line StopIterations).
        subset_key = "default" if subsets.get("default") else next(iter(subsets))
        self._corpus_subset = subset_key
        subset = subsets[subset_key]
        split_data = subset.get(self.cfg.corpus_split) or subset[next(iter(subset))]
        corpus_ds = split_data["corpus"]
        _inject = self._inject_qrels_docs()
        if _cap and len(corpus_ds) > _cap:                 # RAM-safe: never materialize the full giant corpus
            capped = corpus_ds.shuffle(seed=self.cfg.seed).select(range(_cap))
        else:
            capped = corpus_ds
        out = {
            row["id"]: ((row.get("title") or "") + " " + (row.get("text") or "")).strip()
            for row in capped
        }
        # Human-query anchoring on a capped giant corpus is meaningless unless the
        # qrels-judged documents are present: a random 100k of 5.4M docs almost
        # never contains a real query's golds, every model scores 0, and
        # obj_decisive collapses to 0 (observed on ClimateFEVER). When
        # GYM_INJECT_QRELS_DOCS is set (value = split), stream the full corpus
        # once and add every judged doc for that split. Synthetic-query runs
        # leave this unset and are byte-identical to before. MUST be set
        # identically for preembed and judging so cache keys line up.
        if _cap and _inject:
            rel = task.dataset[subset_key].get(_inject) or task.dataset[subset_key][next(iter(task.dataset[subset_key]))]
            need = set()
            for docs in rel["relevant_docs"].values():
                need.update(docs.keys())
            need -= set(out)
            if need:
                added = 0
                for row in corpus_ds:
                    if row["id"] in need:
                        out[row["id"]] = ((row.get("title") or "") + " " + (row.get("text") or "")).strip()
                        added += 1
                        if added == len(need):
                            break
                print(f"[load_corpus] injected {added}/{len(need)} qrels-judged docs into capped corpus")
        return out

    # --------------------------------------------------------------- queries
    def _query_config_hash(self) -> str:
        """Cache key over everything that changes which queries get generated.

        Without this, rerunning with a different --n-queries or filter setting
        into the same output dir silently reuses the old query file.
        """
        c = self.cfg
        gen_id = getattr(self.gen_client, "model", type(self.gen_client).__name__)
        key = (f"{gen_id}|{c.task_name}|{c.n_queries}|{c.seed}|{c.docs_per_query}|"
               f"{c.gen_overshoot}|{c.filter_queries}|{c.filter_min_score}|"
               f"{c.dedup_threshold}|{c.min_query_chars}|{c.max_query_chars}")
        # Fold the corpus-subsample cap in: queries are generated from the (capped)
        # corpus, so toggling GYM_MAX_CORPUS_DOCS must not silently reuse queries
        # built at a different cap. Appended only when a cap is set, so uncapped
        # runs keep their existing cache keys unchanged.
        cap = self._corpus_cap()
        if cap:
            key += f"|cap={cap}"
            inject = self._inject_qrels_docs()
            if inject:
                key += f"|inject_qrels={inject}"
        # Encode-time doc caps change the retrievals behind every verdict.
        # Conditional on non-default values so existing caches stay valid.
        doc_chars = int(os.environ.get("GYM_MAX_DOC_CHARS", "0") or 0)
        if doc_chars:
            key += f"|doc_chars={doc_chars}"
        max_seq = int(os.environ.get("GYM_MAX_SEQ", "4096"))
        if max_seq != 4096:
            key += f"|max_seq={max_seq}"
        return hashlib.sha256(key.encode()).hexdigest()[:8]

    def _queries_path(self) -> Path:
        return self.cfg.output_dir / (
            f"queries_{self.cfg.task_name}_{self._query_config_hash()}.json")

    def _queries_meta_path(self) -> Path:
        """Sidecar for query-generation metadata without changing cache format."""
        return self._queries_path().with_suffix(".meta.json")

    def get_queries(self, corpus: dict[str, str]) -> list[Query]:
        path = self._queries_path()
        meta_path = self._queries_meta_path()
        if path.exists():
            data = json.loads(path.read_text())
            if data:
                self._n_queries_generated = None
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text())
                    self._n_queries_generated = meta.get("n_queries_generated")
                return [Query(**q) for q in data]
            # An empty query cache is a crash artifact (e.g. generation ran
            # against a dead generator and every call failed): treat it as a
            # miss and regenerate, never trust it.
            logger.warning("empty query cache at %s -- regenerating", path)
            path.unlink()
        gen = QueryGenerator(self.gen_client, self.cfg)
        queries = gen.run(corpus)
        if not queries:
            # Never persist or proceed with zero queries: downstream this
            # surfaces as an opaque size-0 IndexError inside the encoder
            # dataloader. A dead generator endpoint must fail loudly here.
            raise RuntimeError(
                f"query generation returned 0 retained queries for "
                f"{self.cfg.task_name}: the generator endpoint is likely dead "
                "or the filter rejected everything. Not writing a cache file.")
        path.write_text(json.dumps([asdict(q) for q in queries], indent=2))
        self._n_queries_generated = gen.last_generated_count
        meta_path.write_text(json.dumps({
            "n_queries_generated": self._n_queries_generated,
        }, indent=2))
        return queries

    # --------------------------------------------------------------- retrieval
    def _retrieve(self, model_name: str, corpus, queries) -> list:
        if model_name in self._retr_cache:
            return self._retr_cache[model_name]
        if model_name == "bm25":
            retr = BM25Retriever(top_k=self.cfg.top_k).retrieve(corpus, queries)
        # ColBERT (late interaction) and SPLADE (learned sparse) are additive,
        # non-dense entrants that go through their own retrievers instead of the
        # dense matmul harness. Matched by name so callers just pass "colbert".
        elif model_name == "colbert" or model_name.startswith("colbert"):
            retr = ColBERTRetriever(top_k=self.cfg.top_k).retrieve(corpus, queries)
        elif "splade" in model_name:
            # Learned-sparse (SPLADE) entrants are not implemented: baselines.py
            # ships BM25Retriever and ColBERTRetriever only. Fail fast with a clear
            # message instead of crashing the whole tournament with an ImportError
            # on a missing class. To add one, implement gym.baselines.SPLADERetriever
            # duck-typed like BM25Retriever (.model_name + .retrieve(corpus, queries)).
            raise NotImplementedError(
                f"SPLADE retriever not implemented (model={model_name!r}); "
                "remove it from the roster or implement gym.baselines.SPLADERetriever")
        else:
            enc = make_encoder(model_name, task_name=self.cfg.task_name,
                               split=self.cfg.corpus_split,
                               subset=getattr(self, "_corpus_subset", "default"),
                               use_mteb=self.cfg.use_mteb_models)
            retr = self.harness.retrieve(enc, corpus, queries)
            # Free this model's GPU memory before the next encoder loads. Only the
            # Retrieved lists (CPU) live in _retr_cache, so the encoder is safe to drop.
            # Without this, 7B-class encoders accumulate on one GPU and OOM mid-
            # tournament (observed on NFCorpus/FiQA at the 25-model roster, 2026-07-07).
            del enc
            try:
                import gc as _gc, torch as _torch
                _gc.collect()
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
            except Exception:
                pass
        self._retr_cache[model_name] = retr
        return retr

    # --------------------------------------------------------------- matchup
    def _verdict_config_hash(self, queries) -> str:
        """Cache key over everything that changes verdicts for a model pair.

        Judge model, judge prompt, top_k, position flipping, and the query set
        itself. This is what stops a --judge qwen3 rerun (or a judge-prompt
        edit) from silently reusing stale verdicts out of the same output dir.
        """
        from .judge import _JUDGE_SYSTEM
        judge_id = getattr(self.judge_client, "model", type(self.judge_client).__name__)
        from .judge import REGISTRY_VERSION
        # hash the judge's RESOLVED system prompt (per-task instruction included)
        # a task with a registry instruction hashes differently; generic tasks hash
        # byte-identically to before, so all existing verdict caches are preserved.
        prompt_sig = hashlib.sha256(
            self.judge.system.encode()).hexdigest()[:8]
        qsig = "||".join(f"{q.qid}:{q.text}" for q in queries)
        key = f"{judge_id}|{prompt_sig}|{self.cfg.top_k}|{self.cfg.flip_positions}|{qsig}"
        # Same cap guard as the query hash: a different corpus cap changes the
        # retrievals behind every verdict. Conditional so uncapped caches are
        # untouched. (qsig already shifts when the cap changes the query set,
        # but keying on the cap directly makes the dependency explicit.)
        cap = self._corpus_cap()
        if cap:
            key += f"|cap={cap}"
            inject = self._inject_qrels_docs()
            if inject:
                key += f"|inject_qrels={inject}"
        # Encode-time doc caps change the retrievals behind every verdict.
        # Conditional on non-default values so existing caches stay valid.
        doc_chars = int(os.environ.get("GYM_MAX_DOC_CHARS", "0") or 0)
        if doc_chars:
            key += f"|doc_chars={doc_chars}"
        max_seq = int(os.environ.get("GYM_MAX_SEQ", "4096"))
        if max_seq != 4096:
            key += f"|max_seq={max_seq}"
        return hashlib.sha256(key.encode()).hexdigest()[:8]

    def _matchup_path(self, a, b, queries) -> Path:
        from .encoders import cache_key
        return self.cfg.output_dir / (
            f"verdicts_{cache_key(a)}__{cache_key(b)}_"
            f"{self._verdict_config_hash(queries)}.json")

    def matchup(self, model_a, model_b, corpus, queries) -> list[Verdict]:
        path = self._matchup_path(model_a, model_b, queries)
        if path.exists():
            data = json.loads(path.read_text())
            logger.info("%s vs %s: %d cached verdicts", model_a, model_b, len(data))
            return [Verdict(**v) for v in data]

        # Verdicts stream to a JSONL sidecar as they complete, so a crash or
        # preemption mid-pair costs only the in-flight queries, not the whole
        # pair (~2x n_queries judge calls). On restart we resume from it.
        jsonl = path.with_suffix(".jsonl")
        done: dict[str, Verdict] = {}
        if jsonl.exists():
            for line in jsonl.read_text().splitlines():
                if line.strip():
                    v = Verdict(**json.loads(line))
                    done[v.qid] = v
            if done:
                logger.info("%s vs %s: resuming, %d verdicts already on disk",
                            model_a, model_b, len(done))

        t0 = time.time()
        ra = self._retrieve(model_a, corpus, queries)
        rb = self._retrieve(model_b, corpus, queries)
        todo = [r for r in ra if r.qid not in done]
        if todo:
            write_lock = threading.Lock()

            def _persist(v: Verdict) -> None:
                with write_lock:
                    with jsonl.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(asdict(v)) + "\n")

            new = self.judge.judge_all(todo, rb, model_a, model_b,
                                       on_verdict=_persist)
            done.update({v.qid: v for v in new})

        # Finalize in query order for deterministic files; the JSONL stays as
        # the raw artifact.
        order = {q.qid: i for i, q in enumerate(queries)}
        verdicts = sorted(done.values(), key=lambda v: order.get(v.qid, 1 << 30))
        path.write_text(json.dumps([asdict(v) for v in verdicts], indent=2))
        logger.info("%s vs %s: %d verdicts in %.0fs", model_a, model_b,
                    len(verdicts), time.time() - t0)
        return verdicts

    # --------------------------------------------------------------- tournament
    def tournament(
        self,
        models: list[str],
        corpus,
        queries,
        *,
        output_folder: str | Path | None = None,
        run_started_at: float | None = None,
    ):
        started_at = time.time() if run_started_at is None else run_started_at
        pairs = list(itertools.combinations(models, 2))
        all_verdicts: list[Verdict] = []
        for i, (a, b) in enumerate(pairs, 1):
            logger.info("pair %d/%d: %s vs %s", i, len(pairs), a, b)
            all_verdicts.extend(self.matchup(a, b, corpus, queries))

        ratings = rate(all_verdicts, method=self.cfg.method,
                       base=self.cfg.elo_base, scale=self.cfg.elo_scale,
                       bootstrap=self.cfg.bootstrap_samples, seed=self.cfg.seed)
        self.leaderboard_str = format_leaderboard(ratings)

        out = {
            "task": self.cfg.task_name,
            "n_queries": len(queries),
            "method": self.cfg.method,
            # position bias recomputed from the persisted verdicts, so cached
            # and resumed runs report the true rate
            "a_first_rate": _a_first_rate_from(all_verdicts),
            # recomputed from persisted per-order flags so cached/resumed runs
            # report the true rate, not the in-process counter (mirrors a_first_rate)
            "parse_failure_rate": _parse_failure_rate_from(all_verdicts),
            "models": [
                {"name": m.name, "rating": m.rating, "ci_low": m.ci_low,
                 "ci_high": m.ci_high, "wins": m.wins, "losses": m.losses,
                 "ties": m.ties, "n": m.n}
                for m in ratings
            ],
        }
        (self.cfg.output_dir / "leaderboard.json").write_text(json.dumps(out, indent=2))

        # Additive MTEB-style result artifact. Existing caches and
        # leaderboard.json remain unchanged for backwards compatibility.
        from .results import build_result_record, result_directory

        record, experiment_config = build_result_record(
            cfg=self.cfg,
            judge_client=self.judge_client,
            generator_client=self.gen_client,
            models=models,
            ratings=ratings,
            verdicts=all_verdicts,
            n_queries_generated=self._n_queries_generated,
            n_queries=len(queries),
            hf_subset=getattr(self, "_corpus_subset", "default"),
            evaluation_time=time.time() - started_at,
            corpus_cap=self._corpus_cap(),
            inject_qrels_docs=self._inject_qrels_docs(),
            judge_system=self.judge.system,
        )

        results_root = (
            Path(output_folder)
            if output_folder is not None
            else self.cfg.output_dir
        )
        result_dir = result_directory(
            results_root,
            experiment_config["judge_model"],
            experiment_config["generator_model"],
            experiment_config,
        )
        result_dir.mkdir(parents=True, exist_ok=True)
        result_path = result_dir / f"{self.cfg.task_name}.json"
        result_path.write_text(json.dumps(record, indent=2))
        self.result_path = result_path
        return ratings

    # --------------------------------------------------------------- convenience
    def run(
        self,
        models: list[str] | None = None,
        *,
        output_folder: str | Path | None = None,
    ):
        models = list(models if models is not None else self.cfg.models)
        if not models:
            raise ValueError(
                "No models provided. Pass models to Gym.run(...) "
                "or set GymConfig(models=[...])."
            )

        started_at = time.time()
        corpus = self.load_corpus()
        queries = self.get_queries(corpus)
        return self.tournament(
            models,
            corpus,
            queries,
            output_folder=output_folder,
            run_started_at=started_at,
        )
