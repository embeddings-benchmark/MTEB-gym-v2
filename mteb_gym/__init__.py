"""MTEB Gym: label-free, LLM-judged model selection for embedding models."""

from .llm import llm
from .run import Result, run
from .validate import correlate, rank_agreement

__all__ = ["run", "llm", "Result", "rank_agreement", "correlate"]
