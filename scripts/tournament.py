#!/usr/bin/env python3
"""
N-model round-robin tournament. Defaults to a sub-1B model set comparable to
Rohan's, plus a BM25 anchor.

    # local Qwen3-4B judge (recommended on the GPU box):
    PYTHONPATH=. python3 scripts/tournament.py \
        --judge qwen3 --base-url http://localhost:8000/v1 \
        --n-queries 300 --output results/qwen3_synth

    # Claude judge:
    export ANTHROPIC_API_KEY=sk-ant-...
    PYTHONPATH=. python3 scripts/tournament.py --n-queries 300

    # dry run:
    PYTHONPATH=. python3 scripts/tournament.py --mock
"""
import argparse
import logging
import os

from gym import Gym, GymConfig
from gym.clients import AnthropicClient, MockClient, OpenAICompatClient

# Rohan's original 7 (the open sub-1B set from the arena) + the bm25 anchor.
DEFAULT_MODELS = [
    "bm25",
    "sentence-transformers/all-MiniLM-L6-v2",
    "intfloat/multilingual-e5-small",
    "jinaai/jina-embeddings-v2-base-en",
    "intfloat/multilingual-e5-large-instruct",
    "BAAI/bge-large-en-v1.5",
    "nomic-ai/nomic-embed-text-v1.5",
    "mixedbread-ai/mxbai-embed-large-v1",
]


def _api_key(args):
    if args.api_key_env:
        key = os.environ.get(args.api_key_env)
        if not key:
            raise SystemExit(f"--api-key-env {args.api_key_env} is set but empty")
        return key
    return "EMPTY"


def build_judge(args):
    if args.mock:
        return MockClient()
    if args.judge == "qwen3":
        return OpenAICompatClient(model=args.model or "Qwen/Qwen3-4B-Instruct-2507",
                                  base_url=args.base_url, api_key=_api_key(args))
    return AnthropicClient(model=args.model or "claude-haiku-4-5-20251001")


def build_generator(args):
    """Separate query-generator client. Keep it family-disjoint from the judge
    (and the competitors) or the self-preference warning will fire."""
    if not args.gen_model:
        return None          # falls back to the judge client, with a warning
    return OpenAICompatClient(model=args.gen_model,
                              base_url=args.gen_base_url or args.base_url,
                              api_key=_api_key(args))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--judge", choices=["claude", "qwen3"], default="claude")
    ap.add_argument("--model", default=None, help="override judge model id")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--task", default="NFCorpus")
    ap.add_argument("--n-queries", type=int, default=300)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--method", choices=["bradley_terry", "elo"], default="bradley_terry")
    ap.add_argument("--no-filter", action="store_true")
    ap.add_argument("--workers", type=int, default=8,
                    help="concurrent judge/filter calls (vLLM throughput needs this)")
    ap.add_argument("--gen-workers", type=int, default=16,
                    help="concurrent query-generation calls; order-preserving, "
                         "so the generated query set is identical at any count")
    ap.add_argument("--seed", type=int, default=0,
                    help="query-generation seed (the regeneration sweep varies this)")
    ap.add_argument("--api-key-env", default=None,
                    help="env var holding the API key for openai-compatible judges")
    ap.add_argument("--gen-model", default=None,
                    help="separate query-generator model id (keep family-disjoint "
                         "from the judge and the competitors)")
    ap.add_argument("--gen-base-url", default=None,
                    help="base url for the generator endpoint (defaults to --base-url)")
    ap.add_argument("--output", default="results/tournament")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    cfg = GymConfig(task_name=args.task, n_queries=args.n_queries, top_k=args.top_k,
                    method=args.method, filter_queries=not args.no_filter,
                    judge_workers=args.workers, gen_workers=args.gen_workers,
                    seed=args.seed, output_dir=args.output)
    gym = Gym(cfg, judge_client=build_judge(args), gen_client=build_generator(args))
    gym.run(args.models)
    print(gym.leaderboard_str)
    if gym.judge.a_first_rate is not None:
        print(f"\nposition bias (a_first_rate): {gym.judge.a_first_rate:.2f}")
    if gym.judge.parse_failure_rate:
        print(f"judge parse failures: {gym.judge.parse_failure_rate:.1%} "
              "(these score as ties; investigate if above a few percent)")
    print(f"\nresults -> {cfg.output_dir}/leaderboard.json")
    print("next: python3 scripts/validate.py --gym "
          f"{cfg.output_dir}/leaderboard.json --truth mteb")


if __name__ == "__main__":
    main()
