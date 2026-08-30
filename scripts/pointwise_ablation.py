"""
Pointwise vs Pairwise ablation.

Loads already-generated queries + retrieved results from a completed tournament
run, re-judges each model's results independently (1-5 score), then compares
the pointwise ranking against the pairwise Bradley-Terry ranking.

If both produce the same model ordering, pointwise is a valid O(n) substitute
for pairwise at O(n^2) cost.

Usage:
    python scripts/pointwise_ablation.py \
        --tournament-dir results/tournament_7model/ \
        --task NFCorpus \
        --judge claude \
        --model claude-haiku-4-5-20251001 \
        --output results/pointwise_ablation/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from gym.clients import AnthropicClient, MockClient, OpenAICompatClient
from gym.config import GymConfig
from gym.encoders import Encoder, cache_key
from gym.judge import PointwiseJudge, PointwiseRanker
from gym.query_generator import Query
from gym.retrieval_harness import RetrievalHarness, Retrieved
from gym.validate import correlate


def load_queries(tournament_dir: Path, task_name: str) -> list[Query]:
    path = tournament_dir / f"queries_{task_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"No cached queries at {path}. Run the tournament first.")
    data = json.loads(path.read_text())
    return [Query(**q) for q in data]


def load_or_encode(model: str, corpus: dict[str, str],
                   queries: list[Query], cache_dir: Path, top_k: int) -> list[Retrieved]:
    """Reuse cached corpus embeddings from the tournament run if available."""
    harness = RetrievalHarness(cache_dir, top_k=top_k)
    enc = Encoder(model)
    return harness.retrieve(enc, corpus, queries)


def load_corpus(task_name: str) -> dict[str, str]:
    import mteb
    task = mteb.get_tasks(tasks=[task_name])[0]
    task.load_data()
    legacy = getattr(task, "corpus", None)
    if legacy:
        corpus = legacy.get("test") or legacy[next(iter(legacy))]
        out: dict[str, str] = {}
        for did, doc in corpus.items():
            if isinstance(doc, dict):
                out[did] = (doc.get("title", "") + " " + doc.get("text", "")).strip()
            else:
                out[did] = str(doc).strip()
        return out
    subsets = task.dataset
    subset = subsets.get("default") or subsets[next(iter(subsets))]
    split_data = subset.get("test") or subset[next(iter(subset))]
    corpus_ds = split_data["corpus"]
    return {
        row["id"]: ((row.get("title") or "") + " " + (row.get("text") or "")).strip()
        for row in corpus_ds
    }


def make_client(judge: str, model_override: str | None, base_url: str | None, mock: bool):
    if mock:
        return MockClient()
    if judge == "claude":
        return AnthropicClient(model=model_override or "claude-haiku-4-5-20251001")
    if judge == "qwen3":
        return OpenAICompatClient(
            model=model_override or "Qwen/Qwen3-4B-Instruct-2507",
            base_url=base_url or "http://localhost:8000/v1",
            api_key="EMPTY",
        )
    raise ValueError(f"Unknown judge: {judge}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tournament-dir", required=True,
                        help="Path to completed tournament output dir (has queries_*.json)")
    parser.add_argument("--models", nargs="+",
                        help="Models to score (default: same 7 as the main tournament)")
    parser.add_argument("--task", default="NFCorpus")
    parser.add_argument("--judge", default="claude", choices=["claude", "qwen3"])
    parser.add_argument("--model", default=None, help="Override judge model id")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", default="results/pointwise_ablation/")
    args = parser.parse_args()

    tournament_dir = Path(args.tournament_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = tournament_dir / "cache"   # reuse tournament embeddings

    models = args.models or [
        "sentence-transformers/all-MiniLM-L6-v2",
        "intfloat/multilingual-e5-small",
        "BAAI/bge-large-en-v1.5",
        "intfloat/multilingual-e5-large-instruct",
        "nomic-ai/nomic-embed-text-v1.5",
        "mixedbread-ai/mxbai-embed-large-v1",
        "jinaai/jina-embeddings-v2-base-en",
    ]

    print(f"Loading queries from {tournament_dir}...")
    queries = load_queries(tournament_dir, args.task)
    print(f"  {len(queries)} queries loaded")

    print(f"Loading corpus: {args.task}...")
    corpus = load_corpus(args.task)
    print(f"  {len(corpus)} docs")

    client = make_client(args.judge, args.model, args.base_url, args.mock)
    judge = PointwiseJudge(client, note=f"{args.judge}/{args.model or 'default'}")
    ranker = PointwiseRanker()

    all_scores = []
    for model in models:
        print(f"\nScoring {model}...")
        retrieved = load_or_encode(model, corpus, queries, cache_dir, args.top_k)
        scores = judge.score_all(retrieved, model)
        ranker.add(scores)
        all_scores.extend([asdict(s) for s in scores])
        mean = sum(s.score for s in scores) / len(scores)
        print(f"  mean score: {mean:.3f} over {len(scores)} queries")

    # Save raw scores
    (output_dir / "pointwise_scores.json").write_text(
        json.dumps(all_scores, indent=2))

    # Leaderboard
    leaderboard = ranker.rank()
    (output_dir / "pointwise_leaderboard.json").write_text(
        json.dumps(leaderboard, indent=2))

    print("\n" + ranker.format_leaderboard())

    # Compare against pairwise if available
    pairwise_path = tournament_dir / "leaderboard.json"
    if pairwise_path.exists():
        print("\n--- Correlation: pointwise vs pairwise Bradley-Terry ---")
        pairwise_ratings = {
            m["name"]: m["rating"]
            for m in json.loads(pairwise_path.read_text())["models"]
        }
        pointwise_ratings = {r["model"]: r["mean_score"] for r in leaderboard}
        result = correlate(pointwise_ratings, pairwise_ratings)
        if "error" in result:
            print(f"  {result['error']}")
        else:
            print(f"  Spearman ρ = {result['spearman_rho']:.3f}  "
                  f"(p = {result['spearman_p']:.4f})")
            print(f"  Kendall  τ = {result['kendall_tau']:.3f}")
            print(f"  Pointwise ranking: {result['gym_ranking']}")
            print(f"  Pairwise  ranking: {result['truth_ranking']}")
        (output_dir / "correlation_pointwise_vs_pairwise.json").write_text(
            json.dumps(result, indent=2))

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
