# MTEB Gym

Label-free, LLM-judged model selection for embedding models.

MTEB Gym takes an unlabeled corpus, generates synthetic queries, retrieves documents with candidate embedding models, compares their results with an LLM judge, and aggregates the pairwise outcomes into a Bradley–Terry ranking.

```text
corpus → synthetic queries → retrieval → pairwise LLM judging → model ranking
```

**Given a new corpus with no relevance labels, which embedding model should you use?**

MTEB Gym originated from the [MTEB Gym discussion](https://github.com/embeddings-benchmark/mteb/discussions/3068).

## Installation

```bash
pip install "mteb-gym @ git+https://github.com/embeddings-benchmark/MTEB-gym-v2"
```

Add `[colbert]` for late-interaction models.

## Quickstart

**Dry run: no API key, no GPU, about a minute on CPU.** The mock judge answers deterministically, so this checks that everything is installed and wired up; it does not rank models.

```python
import mteb_gym as gym

result = gym.run(
    corpus="NanoNFCorpusRetrieval",
    models=["mteb/baseline-bm25s", "sentence-transformers/all-MiniLM-L6-v2"],
    judge=gym.MockLLM(),
    n_queries=20,
    output_folder="results/demo",
)
print(result.leaderboard)
```

**Real run.** Any MTEB retrieval task or a directory / .jsonl of your own documents, MTEB model ids, and an LLM judge and a generator from different model families. LLMs are addressed by model id and an OpenAI-compatible endpoint, whether a local vLLM server or a provider API (Anthropic and Gemini expose compatible endpoints). If you already have queries, pass them instead of generating: `queries="queries.jsonl"` (or a list). `task_description` is one sentence on what counts as a good result, given to the generator and the judge; for an MTEB task it defaults to the task's own prompt.

```python
result = gym.run(
    corpus="NFCorpus",
    models=["mteb/baseline-bm25s", "BAAI/bge-base-en-v1.5", "intfloat/e5-base-v2"],
    judge=gym.LLM("Qwen/Qwen3-8B", base_url="http://localhost:8000/v1"),
    generator=gym.LLM("MiniMaxAI/MiniMax-M2", base_url="http://localhost:8001/v1"),
    n_queries=100,
    output_folder="results/nfcorpus",
)
print(result.leaderboard)
result.to_dataframe()  # one row per model
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
   Queries, mteb's retrieval predictions, judge verdicts, model revisions, the resolved configuration, and the Git revision are persisted. Every stage is cached by identity (dataset and model revisions, judge and prompt), so a rerun repeats only what changed and a reported ranking can be reproduced offline without new LLM calls.

## Validation against MTEB

For an MTEB task, the ranking can be compared with the official scores after a run; the labels never enter the pipeline. `queries="original"` runs the queries the dataset came with instead of synthetic ones, which isolates the judge.

```python
result.agreement()  # one run
gym.load_results("results/").agreement()  # every run under a directory
gym.load_results("results/").to_dataframe()  # all runs in one frame
```

Official scores come from the MTEB results repository through mteb's result cache. Models without one are skipped unless `agreement(evaluate_missing=True)`, which runs mteb on the real task and stores the result in that cache. The output records whether each anchor was official or self-run, with Spearman, Kendall, top-10 Spearman and AP correlation and bootstrap intervals.

## Architecture

```text
mteb_gym/
├── corpus.py     load an MTEB task corpus or a local one
├── queries.py    synthetic query generation and filtering
├── task.py       the corpus + queries as an mteb retrieval task
├── retrieval.py  mteb.evaluate per model; read mteb's predictions
├── judge.py      pairwise LLM judging, both presentation orders
├── rank.py       Bradley-Terry with bootstrap CIs
├── results.py    records: Result, Results, load_results
├── agreement.py  agreement with official MTEB scores
├── llm.py        LLM (any OpenAI-compatible endpoint) and MockLLM
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
