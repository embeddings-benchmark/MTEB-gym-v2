"""
One call runs the pipeline and caches every stage on disk:

    corpus -> queries -> mteb retrieval per model -> pairwise judging -> Bradley-Terry -> record

    out/queries/      generated queries, one file per (corpus, generator, parameters)
    out/predictions/  <model>@<revision>/<query set>/  mteb's prediction file
    out/verdicts/     one file per model pair
    out/records/      one record per run: ratings, config, diagnostics
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import threading
import time
from dataclasses import asdict
from pathlib import Path

from . import corpus as corpus_mod
from . import results, retrieval
from . import task as task_mod
from .judge import Judge, Verdict, task_prompt
from .queries import Query, QueryGenerator
from .rank import rate
from .results import Result
from .retrieval import Ranked, slug

logger = logging.getLogger(__name__)


def _model_id(client) -> str:
    return str(getattr(client, "model", type(client).__name__))


def _sha(*parts) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12]


def resolve_description(task_description: str | None, corpus) -> tuple[str | None, str | None]:
    """None: the task's own mteb prompt if it has one; text: verbatim."""
    if task_description is None:
        p = task_prompt(getattr(corpus.metadata, "prompt", None))
        return (p, "mteb:task_prompt") if p else (None, None)
    return task_description, "config:task_description"


def _cached_queries(path: Path, gen: QueryGenerator, docs: dict[str, str]) -> tuple[list[Query], int | None]:
    if path.exists():
        data = json.loads(path.read_text())
        if data["queries"]:
            return [Query(**q) for q in data["queries"]], data.get("n_generated")
        logger.warning("empty query cache at %s (crash artifact); regenerating", path)
    queries = gen.run(docs)
    if not queries:
        raise RuntimeError(
            "query generation returned 0 queries: the generator endpoint is likely dead or the filter rejected everything"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"n_generated": gen.n_generated, "queries": [asdict(q) for q in queries]}, indent=2))
    return queries, gen.n_generated


def own_queries(spec) -> list[Query]:
    """Your own queries: a list of strings, a .txt (one per line) or a .jsonl with id/text."""
    if isinstance(spec, (list, tuple)):
        return [Query(f"q{i}", str(t), []) for i, t in enumerate(spec)]
    path = Path(spec)
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in lines]
        return [Query(str(r.get("id", i)), str(r["text"]), []) for i, r in enumerate(rows)]
    return [Query(f"q{i}", line.strip(), []) for i, line in enumerate(lines)]


def verdict_key(judge: Judge, top_k: int, query_set: str, a: str, rev_a: str | None, b: str, rev_b: str | None) -> str:
    """Identity of a pair's verdicts: judge, resolved prompt, top_k, query set, both models at their revisions."""
    return _sha(_model_id(judge.client), judge.system, top_k, query_set, f"{a}@{rev_a}", f"{b}@{rev_b}")


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
    queries="synthetic",
    task_description: str | None = None,
    n_queries: int = 100,
    top_k: int = 10,
    seed: int = 0,
    filter: bool = True,
    out: str | Path = "results",
    batch_size: int = 32,
    workers: int = 8,
) -> Result:
    """Rank `models` on `corpus` (an mteb task name or a local path) with an LLM judge.

    `judge` and `generator` are clients from mteb_gym.llm. `queries`: "synthetic" (the
    generator writes them; defaults to the judge), "task" (the benchmark's own queries
    and labels, for validation), or your own as a path / list. `task_description` is
    one sentence on what counts as a good result, given to generator and judge; by
    default the task's own mteb prompt. `batch_size` is the encode batch; `workers`
    the concurrent LLM calls."""
    started = time.time()
    out, models = Path(out), list(models)
    if not models:
        raise ValueError("no models given")
    gen_client = generator if generator is not None else judge

    corp = corpus_mod.load(corpus)
    description, description_source = resolve_description(task_description, corp)
    gen = QueryGenerator(
        gen_client, task_description=description, n_queries=n_queries, seed=seed, filter=filter, workers=workers
    )

    if queries == "synthetic":
        if generator is None:
            logger.warning("no generator given: the judge also writes the queries (self-preference risk)")
        query_set = f"{slug(corp.id)}-{slug(_model_id(gen_client))}-{_sha(sorted(gen.params.items()))}"
        qs, n_generated = _cached_queries(out / "queries" / f"{query_set}.json", gen, corp.docs)
        texts, arm = {q.qid: q.text for q in qs}, "synthetic"
    elif queries == "task":
        if not corp.queries:
            raise ValueError(f"{corp.name} has no queries of its own")
        qs, n_generated, texts, arm = None, None, corp.queries, "task"
        query_set = f"{slug(corp.id)}-task"
    else:
        qs, n_generated = own_queries(queries), None
        texts, arm = {q.qid: q.text for q in qs}, "own"
        query_set = f"{slug(corp.id)}-own-{_sha(*(f'{k}:{v}' for k, v in texts.items()))}"

    import mteb

    revisions = {m: mteb.get_model_meta(m).revision for m in models}  # mteb's pins: part of every cache identity
    gym_task = task_mod.build(corp, qs)
    ranked: dict[str, list[Ranked]] = {}
    for m in models:
        folder = out / "predictions" / f"{slug(m)}@{revisions[m]}" / query_set
        ranked[m] = retrieval.top_k(retrieval.predict(m, gym_task, folder, batch_size=batch_size), corp, texts, top_k)

    jd = Judge(judge, instruction=description, workers=workers)
    verdicts: list[Verdict] = []
    for i, (a, b) in enumerate(itertools.combinations(models, 2), 1):
        logger.info("pair %d/%d: %s vs %s", i, len(models) * (len(models) - 1) // 2, a, b)
        key = verdict_key(jd, top_k, query_set, a, revisions[a], b, revisions[b])
        verdicts.extend(judge_pair_cached(out / "verdicts", jd, a, b, ranked[a], ranked[b], key))

    ratings = rate(verdicts, seed=seed)
    experiment = {
        "corpus_id": corp.id,
        "query_set": query_set,
        "arm": arm,
        "judge_model": _model_id(judge),
        "generator_model": _model_id(gen_client) if arm == "synthetic" else None,
        "task_description": description,
        "task_description_source": description_source,
        "judge_system": jd.system,
        "n_queries_generated": n_generated,
        "n_queries": len(texts),
        "top_k": top_k,
        "seed": seed,
        **({f"gen_{k}": v for k, v in gen.params.items()} if arm == "synthetic" else {}),
        "models": models,
        "model_revisions": revisions,
    }
    experiment["config_hash"] = results.config_hash(experiment)
    path = results.record_path(out, corp.name, experiment)
    if path.exists():  # same configuration, same verdicts: the record stands, agreement included
        return Result.from_disk(path)
    result = Result(results.build_record(corp, experiment, ratings, verdicts, time.time() - started, revisions), path)
    result.to_disk()
    return result
