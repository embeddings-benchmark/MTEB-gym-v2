"""LLM clients: LLM for any OpenAI-compatible /chat/completions endpoint (vLLM, Ollama,
OpenAI, Together, OpenRouter, and the compatible endpoints of Anthropic and Gemini),
MockLLM for tests and dry runs. A client implements chat(messages, temperature=0.0) -> str."""

from __future__ import annotations

import hashlib
import json
import os
import re


class MockLLM:
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


class LLM:
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

        # The SDK retries connection errors, timeouts, 429 and 5xx with backoff. Anything else
        # (bad key, unknown model) raises at once. timeout: a hung call would otherwise stall a worker forever.
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"),
            timeout=timeout,
            max_retries=max_retries,
        )
        self.model = model
        self.max_tokens = max_tokens
        self.extra_body = extra_body  # server knobs, e.g. {"chat_template_kwargs": {"enable_thinking": False}}
        self._rejected: set[str] = set()  # sampling parameters this model refused

    def chat(self, messages: list[dict], temperature: float = 0.0) -> str:
        params = {"temperature": temperature, "max_tokens": self.max_tokens}
        while True:
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    extra_body=self.extra_body,
                    **{k: v for k, v in params.items() if k not in self._rejected},
                )
                return resp.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001
                # Some models refuse a temperature or an output cap (reasoning models take neither) and
                # answer 400 naming the parameter. Drop it, remember, and run that model at its defaults.
                rejected = [k for k in params if k in str(e) and k not in self._rejected]
                if getattr(e, "status_code", None) != 400 or not rejected:
                    raise
                self._rejected.update(rejected)
