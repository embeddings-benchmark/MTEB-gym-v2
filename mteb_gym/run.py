"""
One call runs the pipeline and caches every stage on disk:

    corpus -> queries -> mteb retrieval per model -> pairwise judging -> Bradley-Terry -> record

    <output_folder>/queries/      generated queries, one file per (corpus, generator, parameters)
    <output_folder>/predictions/  <model>@<revision>/<query set>/  mteb's prediction file
    <output_folder>/verdicts/     one file per model pair
    <output_folder>/records/      one record per run: ratings, config, diagnostics
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
from .llm import llm_settings
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
    """The sentence the generator and the judge are given. By default the one mteb gives the
    embedding models for this task: `TaskMetadata.prompt` if the task has one, else nothing, which
    judges plain relevance as mteb's own fallback does. `task_description` replaces it."""
    if task_description is not None:
        return task_description, "config:task_description"
    prompt = task_prompt(getattr(corpus.metadata, "prompt", None))
    return (prompt, "mteb:task_prompt") if prompt else (None, None)


def _cached_queries(path: Path, gen: QueryGenerator, docs: dict[str, str]) -> tuple[list[Query], int | None]:
    if path.exists():
        data = json.loads(path.read_text())
        if data["queries"]:
            return [Query(**q) for q in data["queries"]], data.get("n_generated")
        logger.warning("empty query cache at %s (crash artifact); regenerating", path)
    queries = gen.run(docs)
    if not queries:
        raise RuntimeError(
            "query generation kept 0 queries: the generator's answers were not parseable JSON, "
            "or the filter rejected all of them (see the log)"
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
    """Identity of a pair's verdicts: judge, resolved prompt, what it is shown (top_k documents of
    doc_chars each), query set, both models at their revisions."""
    return _sha(
        _model_id(judge.client), judge.system, top_k, judge.doc_chars, query_set, f"{a}@{rev_a}", f"{b}@{rev_b}"
    )


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
    doc_chars: int = 2000,
    seed: int = 0,
    filter_queries: bool = True,
    output_folder: str | Path = "results",
    batch_size: int = 32,
    workers: int = 8,
) -> Result:
    """Rank `models` on `corpus` with an LLM judge.

    Args:
        corpus: An mteb retrieval task name, a folder of .txt/.md files, or a .jsonl with id and text.
        models: mteb model ids. They run through mteb itself.
        judge: An LLM client (mteb_gym.LLM or MockLLM) that compares the retrieved lists.
        generator: An LLM client that writes the queries. Defaults to the judge.
        queries: "synthetic" (generated), "original" (the dataset's own queries), or your own as a
            .jsonl with id and text, a .txt with one query per line, or a list of strings.
        task_description: One sentence on what counts as a good result, given to generator and judge.
            Defaults to the task's mteb prompt.
        n_queries: Queries to generate.
        top_k: Documents judged per query.
        doc_chars: Characters of each document shown to the judge; longer documents are cut and marked.
        seed: Seed for document sampling and the bootstrap.
        filter_queries: LLM quality filter and deduplication of generated queries.
        output_folder: Where queries, predictions, verdicts and the record are written.
        batch_size: Encoding batch size.
        workers: Concurrent LLM calls.

    Returns:
        The run's Result: leaderboard, record and path.
    """
    started = time.time()
    out, models = Path(output_folder), list(models)
    if not models:
        raise ValueError("no models given")
    gen_client = generator if generator is not None else judge

    corp = corpus_mod.load(corpus)
    description, description_source = resolve_description(task_description, corp)
    logger.info("criterion: %s", description or "plain relevance")
    gen = QueryGenerator(
        gen_client, task_description=description, n_queries=n_queries, seed=seed, filter=filter_queries, workers=workers
    )

    if queries == "synthetic":
        if generator is None:
            logger.warning("no generator given: the judge also writes the queries (self-preference risk)")
        query_set = f"{slug(corp.id)}-{slug(_model_id(gen_client))}-{_sha(sorted(gen.params.items()))}"
        qs, n_generated = _cached_queries(out / "queries" / f"{query_set}.json", gen, corp.docs)
        texts, arm = {q.qid: q.text for q in qs}, "synthetic"
        generator_settings = gen.settings or llm_settings(gen_client)  # cached queries: only the client is known
    elif queries == "original":
        if not corp.queries:
            raise ValueError(f"{corp.name} has no queries of its own")
        qs, n_generated, texts, arm = None, None, corp.queries, "original"
        generator_settings = None
        query_set = f"{slug(corp.id)}-original"
    else:
        qs, n_generated = own_queries(queries), None
        texts, arm = {q.qid: q.text for q in qs}, "own"
        generator_settings = None
        query_set = f"{slug(corp.id)}-own-{_sha(*(f'{k}:{v}' for k, v in texts.items()))}"

    import mteb

    revisions = {m: mteb.get_model_meta(m).revision for m in models}  # mteb's pins: part of every cache identity
    gym_task = task_mod.build(corp, qs)
    ranked: dict[str, list[Ranked]] = {}
    for m in models:
        folder = out / "predictions" / f"{slug(m)}@{revisions[m]}" / query_set
        ranked[m] = retrieval.top_k(retrieval.predict(m, gym_task, folder, batch_size=batch_size), corp, texts, top_k)

    jd = Judge(judge, instruction=description, workers=workers, doc_chars=doc_chars)
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
        "doc_chars": doc_chars,
        "seed": seed,
        **({f"gen_{k}": v for k, v in gen.params.items()} if arm == "synthetic" else {}),
        "models": models,
        "model_revisions": revisions,
    }
    experiment["config_hash"] = results.config_hash(experiment)
    path = results.record_path(out, corp.name, experiment)
    if path.exists():  # same configuration, same verdicts: the record stands, agreement included
        return Result.from_disk(path)
    llms = {"judge": llm_settings(judge), "generator": generator_settings}
    result = Result(
        results.build_record(corp, experiment, ratings, verdicts, time.time() - started, revisions, llms), path
    )
    result.to_disk()
    return result
