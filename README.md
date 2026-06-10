# MTEB Gym

Offline, LLM-judged arena for embedding models. Generate synthetic queries from
any corpus, have two models retrieve, let an LLM judge which result set is
better, and rank models by Bradley-Terry rating — no human labels, no live
Arena traffic. AlpacaEval, but for retrieval.

```
corpus ─▶ generate queries ─▶ filter (heuristic + LLM + dedup)
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

## What's new in v0.2

This is a ground-up rewrite that fixes the issues the team surfaced on Slack.

- **Model-aware encoding (the big one).** Asymmetric models (e5, bge, nomic,
  mxbai, gte-Qwen) are now encoded with their required query/document prefixes.
  Encoding e5 without `query:` / `passage:` silently wrecks its retrieval — the
  likely reason the old gym ranked MiniLM above e5-small while MTEB ranks them
  the other way. Symmetric models (MiniLM, mpnet, jina) correctly get no prefix.
- **Query filtering** (Rohan's key lever): cheap heuristics → LLM 1–5 quality
  score → near-duplicate removal. Bad synthetic queries are the main thing that
  drags correlation down, so we generate an overshoot and keep only the best.
- **Fractional judging** instead of collapsing every order-disagreement to a
  tie. A split decision scores 0.5 and still feeds the rating, so we keep the
  signal that the old ~85%-tie scheme threw away.
- **Position-bias diagnostic** (`a_first_rate`): how often the judge picks the
  first-shown system. 0.50 = unbiased; the team saw ~0.67. Flipping cancels it
  in the score; the metric lets you watch it per judge.
- **Bradley-Terry MLE** ratings with bootstrap CIs (what Chatbot Arena uses),
  with online Elo kept as a fallback.
- **BM25 baseline** as a lexical anchor every dense model should beat.
- **Proper Qwen3-4B support** via an OpenAI-compatible client (point it at a
  local vLLM server — free, fast, reproducible), plus Claude and an ensemble.
- **Clean caching** that no longer breaks on `/` in model names, and **no more
  divide-by-zero**: embeddings are L2-normalised once, retrieval is a single dot
  product.

## Install

```bash
pip install -e ".[full]"      # models + all judge backends
pip install -e ".[retrieval,claude]"   # or pick what you need
```

## Run

```bash
# dry run, no key
PYTHONPATH=. python3 scripts/run_example.py --mock

# real run with Claude Haiku (cheap)
export ANTHROPIC_API_KEY=sk-ant-...
PYTHONPATH=. python3 scripts/run_example.py

# full tournament with a local Qwen3-4B judge
vllm serve Qwen/Qwen3-4B-Instruct-2507 --port 8000        # on the GPU box
PYTHONPATH=. python3 scripts/tournament.py \
    --judge qwen3 --base-url http://localhost:8000/v1 \
    --n-queries 300 --output results/qwen3_synth

# how well does it agree with MTEB?
PYTHONPATH=. python3 scripts/validate.py \
    --gym results/qwen3_synth/leaderboard.json --truth mteb
```

## The experiment this is built for

Does ranking from **synthetic** queries correlate with ground truth as well as
Rohan's **real** Arena queries (ρ ≈ 0.86)? Run the tournament on a shared model
set, then `validate.py` reports Spearman ρ + Kendall τ with a bootstrap CI on
the correlation — so you can also say whether two ρ values are actually
distinguishable given how few models there are.

## Layout

```
gym/
  config.py            GymConfig — one object threaded through the run
  encoders.py          prefix registry + L2-normalised encoding
  clients.py           Mock / Anthropic / OpenAI-compat (Qwen3) / HF / Ensemble
  query_generator.py   synthetic generation + 3-stage filter
  retrieval_harness.py cached corpus encode + top-k dot-product retrieval
  baselines.py         BM25
  judge.py             pairwise judge, fractional scoring, bias diagnostic
  scoring.py           Bradley-Terry + Elo, bootstrap CIs
  gym.py               orchestrator with per-stage caching
  validate.py          Spearman / Kendall vs ground truth
scripts/               run_example · tournament · validate
tests/                 mock-based end-to-end smoke test
```

Run the tests with `PYTHONPATH=. python3 tests/test_pipeline.py` — no GPU, no
key, no network.
