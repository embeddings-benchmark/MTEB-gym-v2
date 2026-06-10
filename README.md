# MTEB Gym 

Offline, LLM-judged arena for embedding models. Generate synthetic queries from any corpus, have two models retrieve, let an LLM judge which result set is better, and rank models by ELO — no human labels, no live Arena traffic. AlpacaEval, but for retrieval.

```
corpus → synthetic queries (LLM) → filter (heuristic + LLM + dedup)
                                          │
                              ┌───────────┴───────────┐
                           model A retrieve        model B retrieve   (+ BM25 anchor)
                              └───────────┬───────────┘
                                  LLM judge (A/B + B/A, fractional)
                                          │
                              Bradley-Terry rating + bootstrap CI
                                          │
                           validate: Spearman/Kendall vs MTEB/Arena
```

Based on the [MTEB Gym discussion](https://github.com/embeddings-benchmark/mteb/discussions/3068) by Muennighoff, KennethEnevoldsen, and orionw.

## Quickstart

```bash
pip install -e ".[full]"
```

```python
from sentence_transformers import SentenceTransformer
from gym import MTEBGym
from gym.clients import AnthropicClient

gym = MTEBGym(
    model_a=SentenceTransformer("Alibaba-NLP/gte-Qwen2-7B-instruct"),
    model_b=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2"),
    model_a_name="gte-Qwen2-7B",
    model_b_name="MiniLM-L6",
    llm_client=AnthropicClient(model="claude-sonnet-4-6"),
    output_dir="results/",
)

gym.run_mteb_task("NFCorpus", n_queries=200)
gym.leaderboard()
```

Output:

```
================================================================
  MTEB GYM LEADERBOARD
================================================================
Rank  Model                                    ELO      CI ±     W     L     T   Win%
----------------------------------------------------------------------
1     MiniLM-L6-v2                             1004    ±  47    25    13    61   54.0%
2     multilingual-e5-small                     996    ±  46    13    25    61   46.0%
```

## What's new in v0.2

A ground-up rewrite based on issues the team surfaced on Slack.

- **Model-aware encoding.** Asymmetric models (e5, bge, nomic, mxbai, gte-Qwen) are now encoded with their required query/document prefixes. Encoding e5 without `query:` / `passage:` silently wrecks retrieval — the likely reason v0.1 ranked MiniLM above e5-small while MTEB ranks them the other way.
- **Query filtering.** Cheap heuristics → LLM 1–5 quality score → near-duplicate removal. Bad synthetic queries are the main thing that drags correlation down.
- **Fractional judging** instead of collapsing every order-disagreement to a tie. A split decision scores 0.5 and still feeds the rating — v0.1 was throwing away most of the signal with its ~85% tie rate.
- **Position-bias diagnostic** (`a_first_rate`): how often the judge picks the first-shown system. 0.50 = unbiased; the team saw ~0.67 in practice. Flipping cancels it in the score; the metric lets you watch it per judge.
- **Bradley-Terry MLE** ratings with bootstrap CIs (what Chatbot Arena uses), with ELO kept as a fallback.
- **BM25 baseline** as a lexical anchor every dense model should beat.
- **Clean caching** — no longer breaks on `/` in model names, no more divide-by-zero: embeddings are L2-normalised once, retrieval is a single dot product.

## Architecture

```
gym/
├── config.py            GymConfig — one object threaded through the run
├── encoders.py          prefix registry + L2-normalised encoding
├── clients.py           Mock / Anthropic / OpenAI-compat (Qwen3) / HF / Ensemble
├── query_generator.py   synthetic generation + 3-stage filter
├── retrieval_harness.py cached corpus encode + top-k retrieval
├── baselines.py         BM25
├── judge.py             pairwise judge, fractional scoring, bias diagnostic
├── scoring.py           Bradley-Terry + Elo, bootstrap CIs
├── gym.py               orchestrator with per-stage caching
└── validate.py          Spearman / Kendall vs ground truth
scripts/                 run_example · tournament · validate
tests/                   mock-based end-to-end smoke test
```

## Query generation strategy

The query generator shows the LLM `k` random corpus docs and asks it to write a query that is *related but not directly answered* by those docs. This forces retrieval rather than shallow surface matching. Queries then go through a 3-stage filter before being used.

Key design choices (per the MTEB Gym discussion):
- On-the-fly queries (like LMSys) rather than fixed sets → captures diverse difficulty
- Shared query sets across model pairs for fair comparison
- Filter bad queries before judging — the main lever for improving correlation

## Judge design

- Pairwise comparison: `(query, top-k from A, top-k from B)` → winner
- Run each pair twice with A/B swapped; fractional scoring keeps the signal from split decisions
- Position-bias diagnostic tracks judge A-preference rate
- Prompt calibrated for retrieval nuance: relevance, completeness, precision

## Multi-model tournament

```bash
python scripts/tournament.py \
    --models "BAAI/bge-large-en-v1.5" "sentence-transformers/e5-large-v2" "Alibaba-NLP/gte-large-en-v1.5" \
    --task NFCorpus \
    --n-queries 200 \
    --output results/tournament/
```

## Validate against MTEB

```bash
python scripts/validate.py \
    --gym results/tournament/leaderboard.json --truth mteb
```

Reports Spearman ρ + Kendall τ with bootstrap CI. Target: match Rohan's ρ ≈ 0.86 from real Arena queries.

## Custom corpus

```python
corpus = {
    "doc_0": "Attention is all you need...",
    "doc_1": "BERT: Pre-training of Deep Bidirectional Transformers...",
}
gym.run(corpus=corpus, n_queries=100, corpus_name="ml_papers")
```

Works on any corpus — medical, legal, customer service, internal docs.

## Design notes

- No ground-truth labels required — the LLM judge replaces nDCG
- Bradley-Terry MLE ratings with 95% bootstrap CIs
- Caching of corpus embeddings: re-run with new models without re-encoding
- Resume-safe: intermediate pairs and verdicts saved to JSONL/JSON
- Run tests with `PYTHONPATH=. python3 tests/test_pipeline.py` — no GPU, no key

## Citation

```bibtex
@misc{mteb-gym-2025,
  title  = {MTEB Gym: LLM-as-Judge Offline Arena for Embedding Models},
  note   = {Based on MTEB Gym discussion: github.com/embeddings-benchmark/mteb/discussions/3068},
  year   = {2025}
}
```

Building on:

```bibtex
@article{muennighoff2022mteb,
  title  = {MTEB: Massive Text Embedding Benchmark},
  author = {Muennighoff, Niklas and Tazi, Nouamane and Magne, Loïc and Reimers, Nils},
  year   = {2022}
}
```
