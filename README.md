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

Embedding models are loaded through MTEB so registered prompts, revisions, and encoding behavior are handled automatically.

```python
import mteb

from gym import Gym, GymConfig

models = [
    mteb.get_model("Alibaba-NLP/gte-Qwen2-7B-instruct"),
    mteb.get_model("sentence-transformers/all-MiniLM-L6-v2"),
]

config = GymConfig(
    models=models,
    judge=judge_client,
    generator=generator_client,
    output_dir="results/",
)

gym = Gym(config)
result = gym.run()
```

Model names do not need to be supplied separately for registered MTEB models; identifying metadata comes from the model itself.

LLM backends use a common client interface, with OpenAI-compatible endpoints preferred where available. For evaluation runs, the generator and judge should come from different model families.

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
from gym import rank_agreement

agreement = rank_agreement("results/")
print(agreement)
```

`rank_agreement()` uses the MTEB result cache when a score is available and follows the normal MTEB evaluation path when a required model/task result is missing.

Agreement statistics include Spearman and Kendall rank correlation.

For a genuinely unlabeled corpus, validation is optional — the Gym leaderboard itself is the output.

## Architecture

```text
gym/
├── config.py             experiment configuration
├── encoders.py           model-aware encoding
├── clients.py            LLM clients
├── query_generator.py    query generation + filtering
├── retrieval_harness.py  retrieval + caching
├── baselines.py          retrieval baselines
├── judge.py              pairwise judging
├── scoring.py            Bradley–Terry scoring
├── results.py            standardized result artifacts
├── repro.py              reproducibility metadata
├── gym.py                experiment orchestration
└── validate.py           rank agreement
```

## Testing

Run the offline smoke suite with:

```bash
PYTHONPATH=. python3 tests/test_pipeline.py
```

The tests cover the end-to-end pipeline, reproducibility helpers, standardized result artifacts, corpus controls, caching/resume behavior, the unified Gym API, and MTEB rank-agreement fallback behavior.

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
