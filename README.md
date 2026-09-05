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
pip install -e ".[full]"
```

## Quickstart

```python
import mteb_gym as gym

result = gym.run(
    corpus="NFCorpus",                     # any MTEB retrieval task, or a directory / .jsonl of your own documents
    models=["mteb/baseline-bm25s", "BAAI/bge-base-en-v1.5", "intfloat/e5-base-v2"],   # MTEB model ids
    judge=gym.llm("Qwen/Qwen3-8B", base_url="http://localhost:8000/v1"),   # any OpenAI-compatible endpoint
    generator=gym.AnthropicClient("claude-sonnet-4-5"),                    # a different model family from the judge
    n_queries=100,
    out="results/nfcorpus",
)
print(result.leaderboard)
print(result.agreement())   # only for an MTEB task: how well the ranking agrees with official scores
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
   Queries, retrievals, judge verdicts, model revisions, corpus controls, configuration hashes, Git revision, and result metadata are persisted. This lets reported rankings and downstream statistics be reproduced offline without repeating LLM calls.

Corpus sampling and size controls are part of the run identity, so incompatible runs cannot silently share cached artifacts.

A completely fresh LLM call is not guaranteed to be byte-identical because inference may be nondeterministic; the recorded run artifacts are the reproducibility boundary for a reported result.

## Validation against MTEB

When human-labeled MTEB results exist, they can be used **after** a Gym run to measure how closely the label-free ranking matches the official ranking.

The official labels are never exposed to the Gym pipeline itself.

```python
from mteb_gym import rank_agreement

agreement = rank_agreement("results/")
print(agreement)
```

`rank_agreement()` uses the MTEB result cache when a score is available and follows the normal MTEB evaluation path when a required model/task result is missing.

Agreement statistics include Spearman and Kendall rank correlation.

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

## Testing

Run offline tests:

```bash
python tests/test_pipeline.py
```

They run on a mock LLM and cover query generation determinism, judging, Bradley-Terry, the verdict cache and resume, the result record, and rank agreement. The end-to-end test runs when mteb and a cached MiniLM are available.

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
