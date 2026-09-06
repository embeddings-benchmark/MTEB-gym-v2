"""LLM clients. A client implements chat(messages, temperature=0.0) -> str."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time


def llm(model: str, base_url: str | None = None, api_key: str | None = None, **kwargs):
    """Client for `model` on an OpenAI-compatible /chat/completions endpoint: vLLM,
    Ollama, OpenAI, Together, OpenRouter, and the compatible endpoints of Anthropic
    and Gemini. "mock" is the deterministic test client (no network)."""
    if model == "mock":
        return MockClient(**kwargs)
    return OpenAICompatClient(model, base_url=base_url, api_key=api_key, **kwargs)


class MockClient:
    """Deterministic stand-in for tests and dry runs: same input, same output."""

    model = "mock"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def _hash(self, text: str) -> int:
        return int(hashlib.sha256(f"{self.seed}:{text}".encode()).hexdigest()[:8], 16)

    def chat(self, messages: list[dict], temperature: float = 0.0) -> str:
        prompt = " ".join(m.get("content", "") for m in messages)
        h = self._hash(prompt)
        if "rate the quality" in prompt.lower():
            return json.dumps({"score": (h % 5) + 1, "reason": "mock"})
        if "system a" in prompt.lower():
            return json.dumps({"winner": ["A", "B", "tie"][h % 3], "confidence": "low", "reasoning": "mock"})
        words = re.findall(r"[A-Za-z]{4,}", prompt.split("[1] ", 1)[-1])[:6]  # from the first shown document
        return json.dumps({"query": "what is known about " + (" ".join(words) or f"topic {h % 1000}")})


class OpenAICompatClient:
    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 512,
        max_retries: int = 4,
        extra_body: dict | None = None,
        timeout: float = 120.0,
    ):
        from openai import OpenAI

        # timeout: a hung call would otherwise stall a pool worker forever
        self.client = OpenAI(
            base_url=base_url, api_key=api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"), timeout=timeout
        )
        self.model = model
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.extra_body = extra_body  # server knobs, e.g. {"chat_template_kwargs": {"enable_thinking": False}}

    def chat(self, messages: list[dict], temperature: float = 0.0) -> str:
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=self.max_tokens,
                    extra_body=self.extra_body,
                )
                return resp.choices[0].message.content or ""
            except Exception:  # noqa: BLE001 - transient API errors
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2**attempt)
        return ""
