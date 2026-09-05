"""MTEB Gym: label-free, LLM-judged model selection for embedding models."""

from .llm import AnthropicClient, llm
from .run import Result, run
from .validate import correlate, rank_agreement

__all__ = ["run", "llm", "AnthropicClient", "Result", "rank_agreement", "correlate"]
