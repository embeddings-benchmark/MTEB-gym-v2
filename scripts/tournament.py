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

from gym import Gym, GymConfig
from gym.clients import AnthropicClient, MockClient, OpenAICompatClient

DEFAULT_MODELS = [
    "bm25",
    "sentence-transformers/all-MiniLM-L6-v2",
    "intfloat/multilingual-e5-small",
    "intfloat/multilingual-e5-large-instruct",
    "BAAI/bge-large-en-v1.5",
    "nomic-ai/nomic-embed-text-v1.5",
    "mixedbread-ai/mxbai-embed-large-v1",
]


def build_judge(args):
    if args.mock:
        return MockClient()
    if args.judge == "qwen3":
        return OpenAICompatClient(model=args.model or "Qwen/Qwen3-4B-Instruct-2507",
                                  base_url=args.base_url, api_key="EMPTY")
    return AnthropicClient(model=args.model or "claude-haiku-4-5-20251001")


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
    ap.add_argument("--output", default="results/tournament")
    args = ap.parse_args()

    cfg = GymConfig(task_name=args.task, n_queries=args.n_queries, top_k=args.top_k,
                    method=args.method, filter_queries=not args.no_filter,
                    output_dir=args.output)
    gym = Gym(cfg, judge_client=build_judge(args))
    gym.run(args.models)
    print(gym.leaderboard_str)
    if gym.judge.a_first_rate is not None:
        print(f"\nposition bias (a_first_rate): {gym.judge.a_first_rate:.2f}")
    print(f"\nresults -> {cfg.output_dir}/leaderboard.json")
    print("next: python3 scripts/validate.py --gym "
          f"{cfg.output_dir}/leaderboard.json --truth mteb")


if __name__ == "__main__":
    main()
