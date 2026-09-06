"""MTEB Gym: label-free, LLM-judged model selection for embedding models."""

from .llm import llm
from .results import Result, Results, load_results
from .run import run

__all__ = ["run", "llm", "Result", "Results", "load_results"]
