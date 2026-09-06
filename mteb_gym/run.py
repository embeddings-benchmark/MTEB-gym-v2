"""
One call runs the pipeline and caches every stage on disk:

    corpus -> synthetic queries -> mteb retrieval per model -> pairwise judging -> Bradley-Terry -> record

    out/queries/      one file per (corpus, generator, parameters)
    out/predictions/  <model>/<query-set>/  mteb's prediction file
    out/verdicts/     one file per model pair, keyed on judge, prompt and both retrieved lists
    out/results/      <judge>__<generator>/<config-hash>/<corpus>.json
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from . import corpus as corpus_mod
from . import record, retrieval, validate
from . import task as task_mod
from .judge import Judge, Verdict, task_prompt
from .queries import Query, QueryGenerator
from .rank import ModelRating, format_leaderboard, rate
from .retrieval import Ranked, slug

logger = logging.getLogger(__name__)


def _model_id(client) -> str:
    return str(getattr(client, "model", type(client).__name__))


def _sha(*parts) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12]


@dataclass
class Result:
    ratings: list[ModelRating]
    leaderboard: str
    record_path: Path
    record: dict

    def agreement(self, **kwargs) -> dict:
        """Rank agreement with official MTEB scores (only meaningful for an MTEB task)."""
        return validate.rank_agreement(self.record_path, **kwargs)


def resolve_intent(intent: str | None, corpus) -> tuple[str | None, str | None]:
    """The task criterion given to the generator and the judge. "auto": the task's
    own mteb prompt (generic when it has none); None: generic; text: verbatim."""
    if intent == "auto":
        p = task_prompt(getattr(corpus.metadata, "prompt", None))
        return (p, "mteb:task_prompt") if p else (None, None)
    return (intent, "config:intent") if intent else (None, None)


def _cached_queries(path: Path, gen: QueryGenerator, docs: dict[str, str]) -> tuple[list[Query], int | None]:
    if path.exists():
        data = json.loads(path.read_text())
        if data["queries"]:
            return [Query(**q) for q in data["queries"]], data.get("n_generated")
        logger.warning("empty query cache at %s (crash artifact); regenerating", path)
    queries = gen.run(docs)
    if not queries:
        raise RuntimeError(
            "query generation returned 0 queries: the generator endpoint is likely dead "
            "or the filter rejected everything"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"n_generated": gen.n_generated, "queries": [asdict(q) for q in queries]}, indent=2))
    return queries, gen.n_generated


def verdict_key(
    judge: Judge, top_k: int, query_set: str, a: str, rev_a: str | None, b: str, rev_b: str | None, mteb_version: str
) -> str:
    """Identity of a pair's verdicts, the way mteb keys results: judge, resolved
    prompt, top_k, query set, both models at their revisions, mteb version."""
    return _sha(_model_id(judge.client), judge.system, top_k, query_set, f"{a}@{rev_a}", f"{b}@{rev_b}", mteb_version)


def judge_pair_cached(
    vdir: Path, judge: Judge, a: str, b: str, ra: list[Ranked], rb: list[Ranked], key: str
) -> list[Verdict]:
    """Verdicts for one pair under `key`; streamed to JSONL as they complete so a crash resumes."""
    path = vdir / f"{slug(a)}__{slug(b)}-{key}.json"
    if path.exists():
        return [Verdict(**v) for v in json.loads(path.read_text())]
    vdir.mkdir(parents=True, exist_ok=True)
    jsonl = path.with_suffix(".jsonl")
    done: dict[str, Verdict] = {}
    if jsonl.exists():
        for line in jsonl.read_text().splitlines():
            if line.strip():
                v = Verdict(**json.loads(line))
                done[v.qid] = v
        if done:
            logger.info("%s vs %s: resuming, %d verdicts on disk", a, b, len(done))
    todo = [r for r in ra if r.qid not in done]
    if todo:
        lock = threading.Lock()

        def persist(v: Verdict) -> None:
            with lock, jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(v)) + "\n")

        done.update({v.qid: v for v in judge.judge_all(todo, rb, a, b, on_verdict=persist)})
    order = {r.qid: i for i, r in enumerate(ra)}
    verdicts = sorted(done.values(), key=lambda v: order.get(v.qid, 1 << 30))
    path.write_text(json.dumps([asdict(v) for v in verdicts], indent=2))
    return verdicts


def run(
    corpus: str | Path,
    models: list[str],
    judge,
    generator=None,
    *,
    n_queries: int = 100,
    top_k: int = 10,
    seed: int = 0,
    out: str | Path = "results/run",
    arm: str = "synthetic",
    intent: str | None = "auto",
    filter: bool = True,
    batch_size: int = 32,
    workers: int = 8,
) -> Result:
    """Rank `models` on `corpus` (an mteb task name or a local path) with an LLM judge.

    `judge` and `generator` are LLM clients (see mteb_gym.llm); the generator defaults
    to the judge. `arm="human"` uses the task's own queries and qrels instead of
    synthetic queries. `intent` conditions both generation and judging: "auto" = the
    task's own criterion, None = generic relevance, or text. `batch_size` is the encode
    batch; `workers` the concurrent LLM calls."""
    started = time.time()
    out, models = Path(out), list(models)
    if not models:
        raise ValueError("no models given")
    gen_client = generator if generator is not None else judge
    if generator is None:
        logger.warning("no generator given: the judge also writes the queries (self-preference risk)")

    corp = corpus_mod.load(corpus)

    criterion, criterion_source = resolve_intent(intent, corp)
    gen = QueryGenerator(gen_client, intent=criterion, n_queries=n_queries, seed=seed, filter=filter, workers=workers)
    if arm == "human":
        if not corp.queries:
            raise ValueError(f"{corp.name} has no queries for a human-query arm")
        queries, n_generated, texts = None, None, corp.queries
        query_set = f"{slug(corp.id)}-human"
    else:
        query_set = f"{slug(corp.id)}-{slug(_model_id(gen_client))}-{_sha(sorted(gen.params.items()))}"
        queries, n_generated = _cached_queries(out / "queries" / f"{query_set}.json", gen, corp.docs)
        texts = {q.qid: q.text for q in queries}

    gym_task = task_mod.build(corp, queries)
    ranked: dict[str, list[Ranked]] = {}
    revisions: dict[str, str | None] = {}
    for m in models:
        path = retrieval.predict(m, gym_task, out / "predictions" / slug(m) / query_set, batch_size=batch_size)
        ranked[m] = retrieval.top_k(path, corp, texts, top_k)
        revisions[m] = retrieval.revision(path)

    import mteb

    jd = Judge(judge, instruction=criterion, workers=workers)
    verdicts: list[Verdict] = []
    for i, (a, b) in enumerate(itertools.combinations(models, 2), 1):
        logger.info("pair %d/%d: %s vs %s", i, len(models) * (len(models) - 1) // 2, a, b)
        key = verdict_key(jd, top_k, query_set, a, revisions[a], b, revisions[b], mteb.__version__)
        verdicts.extend(judge_pair_cached(out / "verdicts", jd, a, b, ranked[a], ranked[b], key))

    ratings = rate(verdicts, seed=seed)

    experiment = {
        "corpus_id": corp.id,
        "query_set": query_set,
        "arm": arm,
        "judge_model": _model_id(judge),
        "generator_model": _model_id(gen_client) if arm == "synthetic" else None,
        "intent": criterion,
        "intent_source": criterion_source,
        "judge_system": jd.system,
        "n_queries_generated": n_generated,
        "n_queries": len(texts),
        "top_k": top_k,
        "seed": seed,
        **{f"gen_{k}": v for k, v in gen.params.items()},
        "models": models,
    }
    experiment["config_hash"] = record.config_hash(experiment)
    rec = record.build(corp, experiment, ratings, verdicts, evaluation_time=time.time() - started, revisions=revisions)
    rdir = record.result_directory(out / "results", experiment)
    rdir.mkdir(parents=True, exist_ok=True)
    rpath = rdir / f"{corp.name}.json"
    rpath.write_text(json.dumps(rec, indent=2))
    return Result(ratings, format_leaderboard(ratings), rpath, rec)
