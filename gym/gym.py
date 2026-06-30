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
    def __init__(self, cfg: GymConfig, judge_client, gen_client=None):
        self.cfg = cfg
        cfg.ensure_dirs()
        self.judge_client = judge_client
        self.gen_client = gen_client or judge_client
        self.harness = RetrievalHarness(cfg.cache_dir, top_k=cfg.top_k)
        self.judge = Judge(judge_client, flip_positions=cfg.flip_positions,
                           note=cfg.judge_batch_note, workers=cfg.judge_workers)
        self._retr_cache: dict[str, list] = {}
        self.leaderboard_str = ""
        # Resolved in load_corpus for mteb 2.x multi-subset tasks so the encoder
        # is told the same hf_subset the corpus was actually read from (#9).
        self._corpus_subset = "default"

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
        _cap = os.environ.get("GYM_MAX_CORPUS_DOCS")
        _cap = int(_cap) if _cap else None

        legacy = getattr(task, "corpus", None)
        if legacy:                                         # mteb 1.x
            corpus = legacy.get(self.cfg.corpus_split) or legacy[next(iter(legacy))]
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
        if _cap and len(corpus_ds) > _cap:                 # RAM-safe: never materialize the full giant corpus
            corpus_ds = corpus_ds.shuffle(seed=self.cfg.seed).select(range(_cap))
        return {
            row["id"]: ((row.get("title") or "") + " " + (row.get("text") or "")).strip()
            for row in corpus_ds
        }

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
        cap = int(os.environ.get("GYM_MAX_CORPUS_DOCS", 0) or 0)
        if cap:
            key += f"|cap={cap}"
        return hashlib.sha256(key.encode()).hexdigest()[:8]

    def _queries_path(self) -> Path:
        return self.cfg.output_dir / (
            f"queries_{self.cfg.task_name}_{self._query_config_hash()}.json")

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
        prompt_sig = hashlib.sha256(_JUDGE_SYSTEM.encode()).hexdigest()[:8]
        qsig = "||".join(f"{q.qid}:{q.text}" for q in queries)
        key = f"{judge_id}|{prompt_sig}|{self.cfg.top_k}|{self.cfg.flip_positions}|{qsig}"
        # Same cap guard as the query hash: a different corpus cap changes the
        # retrievals behind every verdict. Conditional so uncapped caches are
        # untouched. (qsig already shifts when the cap changes the query set,
        # but keying on the cap directly makes the dependency explicit.)
        cap = int(os.environ.get("GYM_MAX_CORPUS_DOCS", 0) or 0)
        if cap:
            key += f"|cap={cap}"
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
    def tournament(self, models: list[str], corpus, queries):
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
        return ratings

    # --------------------------------------------------------------- convenience
    def run(self, models: list[str]):
        corpus = self.load_corpus()
        queries = self.get_queries(corpus)
        return self.tournament(models, corpus, queries)
