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
    models: list[str] = field(default_factory=list)
    judge: str | None = None
    generator: str | None = None
    judge_base_url: str | None = None
    generator_base_url: str | None = None
    # Explicit experiment-defining corpus controls. None preserves the legacy
    # environment-variable behavior until Gym resolves them at runtime.
    corpus_cap: int | None = None
    inject_qrels_docs: str | None = None
    # Unlabeled local corpus: a directory of .txt/.md files or a .jsonl with
    # id/text fields. When set, task_name is just the record label and
    # judge_instruction supplies the relevance criterion (no registry entry).
    corpus_path: str | None = None
    judge_instruction: str | None = None
    # Default: the task's own mteb prompt is the judge instruction (the same
    # text instruction-tuned embedders saw when the anchors were produced); a
    # task without one falls back to the generic prompt, like mteb falls back
    # to unprefixed encoding. Wrong/missing prompts are fixed upstream in mteb.
    # False = force the generic prompt (control arm / reproducing old runs).
    judge_instruction_from_task: bool = True

    # --- query generation ---
    n_queries: int = 100          # target number of *kept* queries after filtering
    docs_per_query: int = 3       # docs shown to the generator per query
    gen_overshoot: float = 1.6    # generate this multiple of n_queries before filtering
    min_query_chars: int = 15
    max_query_chars: int = 240
    gen_workers: int = 16         # concurrent generation calls; results are
                                  # order-preserving, so worker count never
                                  # changes the generated query set

    # --- query filtering (the biggest correlation lever) ---
    filter_queries: bool = True
    filter_min_score: int = 3     # LLM quality score 1-5; keep >= this
    dedup_threshold: float = 0.92 # cosine-dup cutoff on a cheap encoder

    # --- retrieval ---
    top_k: int = 10
    use_mteb_models: bool = True  # encode with mteb ModelMeta prompts; falls back to encoders._REGISTRY

    # --- judging ---
    flip_positions: bool = True   # run A/B and B/A to measure & cancel position bias
    judge_batch_note: str = ""    # free-form tag stored in verdicts
    judge_workers: int = 8        # concurrent judge/filter calls; clients are
                                  # thread-safe and vLLM throughput needs batching

    # --- scoring ---
    method: str = "bradley_terry" # "bradley_terry" | "elo"
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
