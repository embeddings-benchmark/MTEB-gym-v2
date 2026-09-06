# MTEB Gym

Label-free, LLM-judged model selection for embedding models.

MTEB Gym takes an unlabeled corpus, generates synthetic queries, retrieves documents with candidate embedding models, compares their results with an LLM judge, and aggregates the pairwise outcomes into a Bradley–Terry ranking.

```text
corpus → synthetic queries → retrieval → pairwise LLM judging → model ranking
```

The goal is simple: **given a new corpus with no relevance labels, which embedding model should you use?**

MTEB Gym originated from the [MTEB Gym discussion](https://github.com/embeddings-benchmark/mteb/discussions/3068).

## Installation

```bash
pip install "mteb-gym @ git+https://github.com/embeddings-benchmark/MTEB-gym-v2"
```

Add extras for a real judge: `"mteb-gym[openai] @ git+..."` for any OpenAI-compatible endpoint, `[claude]` for the Anthropic API, `[colbert]` for late-interaction models.

## Quickstart

**Dry run: no API key, no GPU, about a minute on CPU.** The mock judge answers deterministically, so this checks that everything is installed and wired up; it does not rank models.

```python
import mteb_gym as gym

result = gym.run(
    corpus="NanoNFCorpusRetrieval",
    models=["mteb/baseline-bm25s", "sentence-transformers/all-MiniLM-L6-v2"],
    judge=gym.llm("mock"),
    n_queries=20,
    out="results/demo",
)
print(result.leaderboard)
```

**Real run.** Any MTEB retrieval task or a directory / .jsonl of your own documents, MTEB model ids, an LLM judge on any OpenAI-compatible endpoint, and a generator from a different model family.

```python
result = gym.run(
    corpus="NFCorpus",
    models=["mteb/baseline-bm25s", "BAAI/bge-base-en-v1.5", "intfloat/e5-base-v2"],
    judge=gym.llm("Qwen/Qwen3-8B", base_url="http://localhost:8000/v1"),
    generator=gym.AnthropicClient("claude-sonnet-4-5"),
    n_queries=100,
    out="results/nfcorpus",
)
print(result.leaderboard)
print(result.agreement())  # MTEB tasks only: agreement of the ranking with official scores
```

Or from the shell:

```bash
mteb-gym --corpus NFCorpus --models mteb/baseline-bm25s BAAI/bge-base-en-v1.5 --judge Qwen/Qwen3-8B --judge-url http://localhost:8000/v1
```

Models run through mteb itself (`mteb.evaluate` on a task the gym builds from the corpus and its queries), so prompts, revisions, similarity functions and sparse / late-interaction paths are exactly those of an official MTEB run.

## How it works

1. **Generate queries**  
   Sample corpus documents and use a generator LLM to produce natural-language queries.

2. **Filter queries**  
   Remove degenerate, low-quality, and near-duplicate generations.

3. **Retrieve**  
   Every candidate model retrieves documents for the same query set using its registered MTEB encoding behavior.

4. **Judge pairwise**  
   An LLM compares two models' retrieved lists. Each comparison is run in both A/B and B/A order to reduce position bias, and split decisions contribute fractionally rather than being discarded.

5. **Rank models**  
   Pairwise outcomes are aggregated with Bradley–Terry to produce the leaderboard. Uncertainty is estimated by resampling queries.

6. **Record the run**  
   Queries, mteb's retrieval predictions, judge verdicts, model revisions, the resolved configuration, and the Git revision are persisted. Every stage is cached by content, so a rerun repeats only what changed and a reported ranking can be reproduced offline without new LLM calls.

## Validation against MTEB

When human-labeled MTEB results exist, they can be used **after** a Gym run to measure how closely the label-free ranking matches the official ranking.

The official labels are never exposed to the Gym pipeline itself.

```python
from mteb_gym import rank_agreement

agreement = rank_agreement("results/")  # official scores from the MTEB results repository
agreement = rank_agreement("results/", evaluate_missing=True)  # also run mteb on the real task for models without one
```

Official scores come from the MTEB results repository through mteb's result cache; models without one are skipped unless `evaluate_missing=True`, in which case mteb evaluates them on the real task and stores the result in that cache. The output records per model whether its anchor was official or self-run, and reports Spearman, Kendall, top-10 Spearman and AP correlation with bootstrap intervals.

For a genuinely unlabeled corpus, validation is optional — the Gym leaderboard itself is the output.

## Architecture

```text
mteb_gym/
├── corpus.py     load an MTEB task corpus or a local one
├── queries.py    synthetic query generation and filtering
├── task.py       the corpus + queries as an mteb retrieval task
├── retrieval.py  mteb.evaluate per model; read mteb's predictions
├── judge.py      pairwise LLM judging, both presentation orders
├── rank.py       Bradley-Terry with bootstrap CIs
├── record.py     the result record
├── validate.py   agreement with official MTEB scores
├── llm.py        judge / generator clients
└── run.py        the pipeline, every stage cached on disk
```

## Development

```bash
make install     # uv sync with the dev tools
make test        # pytest, including two end-to-end runs (a local corpus and a Nano MTEB task)
make lint        # ruff format + check
```

Tests use the mock LLM: no API key, no GPU. The end-to-end runs download a small MTEB task and MiniLM once. CI runs the suite on Python 3.10 and 3.13.

## Citation

Paper and citation information will be added with the MTEB Gym release.

MTEB Gym builds on MTEB:

```bibtex
@article{muennighoff2022mteb,
  title  = {MTEB: Massive Text Embedding Benchmark},
  author = {Muennighoff, Niklas and Tazi, Nouamane and Magne, Loïc and Reimers, Nils},
  year   = {2022}
}
```
