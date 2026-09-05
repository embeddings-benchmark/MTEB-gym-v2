"""Central configuration for a gym run. One object threaded through the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GymConfig:
    # --- corpus / task ---
    task_name: str = "NFCorpus"
    corpus_split: str = "test"
    arm: str = "synthetic"
    models: list[str] = field(default_factory=list)   # MTEB model ids; "bm25" and "colbert" are aliases (baselines.ALIASES)
    judge: str | None = None
    generator: str | None = None
    judge_base_url: str | None = None
    generator_base_url: str | None = None
    # corpus_cap: seeded subsample size; inject_qrels_docs: qrels split whose judged docs stay in it
    corpus_cap: int | None = None
    inject_qrels_docs: str | None = None
    # Local corpus: a directory of .txt/.md files or a .jsonl with id/text; task_name is then a label.
    corpus_path: str | None = None
    # Judge criterion: "auto" = the task's own mteb prompt (generic when it has
    # none or corpus_path is set); None = generic; other text = used verbatim.
    judge_instruction: str | None = "auto"

    # --- query generation ---
    n_queries: int = 100          # target number of *kept* queries after filtering
    docs_per_query: int = 3       # docs shown to the generator per query
    gen_overshoot: float = 1.6    # generate this multiple of n_queries before filtering
    min_query_chars: int = 15
    max_query_chars: int = 240
    gen_workers: int = 16         # concurrent generation calls (order-preserving)

    # --- query filtering ---
    filter_queries: bool = True
    filter_min_score: int = 3     # LLM quality score 1-5; keep >= this
    dedup_threshold: float = 0.92 # cosine-dup cutoff on a cheap encoder

    # --- retrieval ---
    top_k: int = 10
    encode_batch_size: int = 32   # per-model encode batch; lower for very long documents

    # --- judging ---
    flip_positions: bool = True   # run A/B and B/A to measure & cancel position bias
    judge_workers: int = 8        # concurrent judge/filter calls

    # --- scoring ---
    bootstrap_samples: int = 1000
    elo_base: float = 1000.0
    elo_scale: float = 400.0

    # --- io ---
    output_dir: Path = Path("results/run")
    seed: int = 0

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.cache_dir = self.output_dir / "cache"

    def ensure_dirs(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
