"""MTEB Gym — offline, LLM-judged arena for embedding models."""

from .config import GymConfig
from .gym import Gym
from .judge import Judge, Verdict, task_prompt
from .query_generator import Query, QueryGenerator
from .scoring import ModelRating, format_leaderboard, rate
from .validate import correlate, rank_agreement, report

__all__ = [
    "GymConfig", "Gym", "task_prompt", "Judge", "Verdict", "Query", "QueryGenerator",
    "ModelRating", "format_leaderboard", "rate", "correlate",
    "rank_agreement", "report",
]
__version__ = "0.2.0"
