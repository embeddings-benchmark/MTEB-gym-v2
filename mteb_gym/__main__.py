"""python -m mteb_gym --corpus NFCorpus --models mteb/baseline-bm25s BAAI/bge-base-en-v1.5 --judge Qwen/Qwen3-8B --judge-url http://localhost:8000/v1"""

from __future__ import annotations

import argparse
import logging

from . import llm, run


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="mteb-gym", description="Rank embedding models on a corpus with an LLM judge.")
    ap.add_argument("--corpus", required=True, help="mteb retrieval task name, or a directory / .jsonl of documents")
    ap.add_argument("--models", nargs="+", required=True, help="mteb model ids, e.g. mteb/baseline-bm25s")
    ap.add_argument(
        "--judge", required=True, help="judge model id on an OpenAI-compatible endpoint ('mock' for a dry run)"
    )
    ap.add_argument("--judge-url", default=None, help="base url for the judge endpoint")
    ap.add_argument("--generator", default=None, help="query generator model id (default: the judge)")
    ap.add_argument("--generator-url", default=None)
    ap.add_argument(
        "--queries", default="synthetic", help="'synthetic', 'task' (the benchmark's own), or a path to your queries"
    )
    ap.add_argument(
        "--task-description",
        default=None,
        help="one sentence on what counts as a good result (default: the task's mteb prompt)",
    )
    ap.add_argument("--n-queries", type=int, default=100)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-filter", action="store_true")
    ap.add_argument("--out", default="results")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    result = run(
        args.corpus,
        args.models,
        llm(args.judge, base_url=args.judge_url),
        llm(args.generator, base_url=args.generator_url) if args.generator else None,
        queries=args.queries,
        task_description=args.task_description,
        n_queries=args.n_queries,
        top_k=args.top_k,
        seed=args.seed,
        filter=not args.no_filter,
        out=args.out,
    )
    print(result.leaderboard)
    bias = result.record["diagnostics"]["a_first_rate"]
    if bias is not None:
        print(f"\nposition bias (a_first_rate): {bias:.2f}  (0.50 = unbiased)")
    print(f"record -> {result.path}")


if __name__ == "__main__":
    main()
