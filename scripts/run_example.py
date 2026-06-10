#!/usr/bin/env python3
"""
Quickstart. Runs a 3-model tournament on NFCorpus.

Mock judge (free, no key, proves the plumbing):
    PYTHONPATH=. python3 scripts/run_example.py --mock

Real judge (Claude Haiku — cheap):
    export ANTHROPIC_API_KEY=sk-ant-...
    PYTHONPATH=. python3 scripts/run_example.py

Qwen3-4B via a local vLLM server (the team's recommended free judge):
    PYTHONPATH=. python3 scripts/run_example.py --judge qwen3 --base-url http://localhost:8000/v1
"""
import argparse

from gym import Gym, GymConfig
from gym.clients import AnthropicClient, MockClient, OpenAICompatClient


def build_judge(args):
    if args.mock:
        return MockClient()
    if args.judge == "qwen3":
        return OpenAICompatClient(model=args.model, base_url=args.base_url, api_key="EMPTY")
    return AnthropicClient(model=args.model or "claude-haiku-4-5-20251001")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--judge", choices=["claude", "qwen3"], default="claude")
    ap.add_argument("--model", default=None)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--task", default="NFCorpus")
    ap.add_argument("--n-queries", type=int, default=60)
    ap.add_argument("--no-filter", action="store_true")
    ap.add_argument("--output", default="results/example")
    args = ap.parse_args()

    cfg = GymConfig(task_name=args.task, n_queries=args.n_queries,
                    filter_queries=not args.no_filter, output_dir=args.output)
    judge = build_judge(args)
    gym = Gym(cfg, judge_client=judge)

    models = [
        "bm25",                                       # lexical baseline
        "sentence-transformers/all-MiniLM-L6-v2",
        "intfloat/multilingual-e5-small",             # now encoded WITH query:/passage:
    ]
    ratings = gym.run(models)
    print(gym.leaderboard_str)
    if gym.judge.a_first_rate is not None:
        print(f"\nposition bias (a_first_rate): {gym.judge.a_first_rate:.2f}  (0.50 = unbiased)")
    print(f"results -> {cfg.output_dir}/leaderboard.json")


if __name__ == "__main__":
    main()
