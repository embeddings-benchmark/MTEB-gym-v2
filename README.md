# MTEB Gym

Label-free, LLM-judged model selection for embedding models.

Give MTEB Gym a corpus, an MTEB task or your own documents. It generates queries for that corpus, or takes queries you supply. Every candidate model retrieves for the same queries, an LLM judge compares the retrieved lists pairwise, and Bradley–Terry turns the comparisons into a ranking. No relevance labels are needed at any step.

```text
corpus → queries (generated, or yours) → retrieval → pairwise LLM judging → model ranking
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

## Usage

`gym.run` needs three things: a corpus, the models to rank, and an LLM judge.

**Corpus.** An MTEB retrieval task name such as `"NFCorpus"`, or a path to your own documents: a directory of `.txt` / `.md` files, or a `.jsonl` with `id` and `text` fields.

**Models.** MTEB model ids, such as `"BAAI/bge-base-en-v1.5"` or `"mteb/baseline-bm25s"`. They run through mteb itself, so prompts, revisions, similarity functions and sparse / late-interaction paths are exactly those of an official MTEB run.

**LLMs.** `gym.LLM(model, base_url=...)` addresses any OpenAI-compatible endpoint: a local vLLM or Ollama server, OpenAI, Together, OpenRouter, or the compatible endpoints of Anthropic and Gemini. `api_key` defaults to `OPENAI_API_KEY`. Use a judge and a generator from different model families.

**Queries.** By default the generator writes `n_queries` queries from the corpus. If you already have queries, pass them instead: `queries="queries.jsonl"` (`id` and `text` fields), a `.txt` with one query per line, or a list of strings. For an MTEB task, `queries="original"` runs the queries the dataset came with.

**Task description.** One sentence on what counts as a good result, given to the generator and the judge, for example `"Given a claim, find documents that refute it"`. For an MTEB task it defaults to the task's own prompt.

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
```

Or from the shell:

```bash
mteb-gym --corpus NFCorpus --models mteb/baseline-bm25s BAAI/bge-base-en-v1.5 --judge Qwen/Qwen3-8B --judge-url http://localhost:8000/v1
```

**Output.** Everything lands under `output_folder`:

```text
results/nfcorpus/
├── records/NFCorpus__Qwen3-8B__MiniMax-M2__q100-s0-<hash>.json   # the run: ratings, config, diagnostics
├── queries/                                                        # generated queries with quality scores
├── predictions/                                                    # mteb's retrieval output per model
└── verdicts/                                                       # judge verdicts per model pair, with reasoning
```

A rerun of the same configuration reuses all of it and makes no LLM calls. Read a run back with `gym.Result.from_disk(path)` (`.leaderboard`, `.to_dataframe()`), or every run under a directory with `gym.load_results("results/")`.

**Cost.** Judging makes 2 calls per query per model pair (both presentation orders): 100 queries and 10 models is 9,000 judge calls, plus about 1.6 generator calls per kept query. A rerun with one model added judges only the new pairs.

**Other arguments.** `top_k` (documents judged per query, 10), `seed`, `filter_queries` (LLM quality filter and deduplication, on by default), `batch_size` for encoding, `workers` for concurrent LLM calls.

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
