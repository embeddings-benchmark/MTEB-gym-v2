"""Encode ONE model's corpus embeddings for a task and cache to disk, then exit.

One model per process so process-exit guarantees full GPU release — defeats the
mteb-model accumulation that OOMs a single long-lived tournament process on
long-document tasks (Touché) and giant corpora (adapted from tejas's helper).
Mirrors the tournament's corpus load + cache path exactly, so the .npy is a
cache hit for the subsequent run."""
import sys
import types
from pathlib import Path

from gym.config import GymConfig
from gym.gym import Gym
from gym.retrieval_harness import RetrievalHarness
from gym.encoders import make_encoder

model = sys.argv[1]
task = sys.argv[2]
out = sys.argv[3]

cfg = GymConfig(task_name=task, output_dir=Path(out), top_k=10)
corpus = Gym.load_corpus(types.SimpleNamespace(cfg=cfg))
h = RetrievalHarness(cfg.cache_dir, top_k=10)
enc = make_encoder(model, task_name=task)
embs = h.encode_corpus(enc, corpus)
print(f"OK {model}: corpus_docs={len(corpus)} embs={getattr(embs, 'shape', None)}", flush=True)
