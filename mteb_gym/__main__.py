"""python -m mteb_gym --corpus NFCorpus --models mteb/baseline-bm25s BAAI/bge-base-en-v1.5 --judge Qwen/Qwen3-8B --judge-url http://localhost:8000/v1"""

from __future__ import annotations

import argparse
import logging

from . import llm, run


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="mteb-gym", description="Rank embedding models on a corpus with an LLM judge.")
    ap.add_argument("--corpus", required=True, help="mteb retrieval task name, or a directory / .jsonl of documents")
    ap.add_argument("--models", nargs="+", required=True, help="mteb model ids, e.g. mteb/baseline-bm25s")
    ap.add_argument("--judge", required=True, help="judge model id on an OpenAI-compatible endpoint ('mock' for a dry run)")
    ap.add_argument("--judge-url", default=None, help="base url for the judge endpoint")
    ap.add_argument("--generator", default=None, help="query generator model id (default: the judge)")
    ap.add_argument("--generator-url", default=None)
    ap.add_argument("--n-queries", type=int, default=100)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/run")
    ap.add_argument("--arm", choices=["synthetic", "human"], default="synthetic")
    ap.add_argument("--no-intent", action="store_true",
                    help="generic relevance for generation and judging instead of the task's own criterion")
    ap.add_argument("--no-filter", action="store_true")
    ap.add_argument("--corpus-cap", type=int, default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    result = run(
        args.corpus, args.models, llm(args.judge, base_url=args.judge_url),
        llm(args.generator, base_url=args.generator_url) if args.generator else None,
        n_queries=args.n_queries, top_k=args.top_k, seed=args.seed, out=args.out, arm=args.arm,
        intent=None if args.no_intent else "auto", filter=not args.no_filter,
        corpus_cap=args.corpus_cap,
    )
    print(result.leaderboard)
    if result.judge.a_first_rate is not None:
        print(f"\nposition bias (a_first_rate): {result.judge.a_first_rate:.2f}  (0.50 = unbiased)")
    print(f"results -> {result.record_path}")


if __name__ == "__main__":
    main()
