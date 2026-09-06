"""MTEB Gym: label-free, LLM-judged model selection for embedding models."""

from .llm import LLM, MockLLM
from .results import Result, Results, load_results
from .run import run

__all__ = ["run", "LLM", "MockLLM", "Result", "Results", "load_results"]
