"""LLM clients. Each implements chat(messages, temperature=0.0) -> str."""

from __future__ import annotations

import hashlib
import json
import os
import time


def llm(model: str, base_url: str | None = None, api_key: str | None = None, **kwargs):
    """Client for `model`: "mock" (tests, no network), "claude-*" via the Anthropic
    API, anything else via an OpenAI-compatible endpoint (vLLM, OpenAI, Together)."""
    if model == "mock":
        return MockClient(**kwargs)
    if model.lower().startswith("claude"):
        return AnthropicClient(model, api_key=api_key, **kwargs)
    return OpenAICompatClient(model, base_url=base_url, api_key=api_key, **kwargs)


class MockClient:
    """Deterministic pseudo-LLM: same input, same output. Answers generation,
    filtering and judging prompts with plausible JSON by sniffing the prompt."""

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
        return json.dumps({"query": f"what is the relationship between factor {h % 1000} "
                                    "and clinical outcomes in affected patients"})


class AnthropicClient:
    def __init__(self, model: str, api_key: str | None = None, max_tokens: int = 512, max_retries: int = 4):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model
        self.max_tokens = max_tokens
        self.max_retries = max_retries

    def chat(self, messages: list[dict], temperature: float = 0.0) -> str:
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        convo = [m for m in messages if m["role"] != "system"]
        for attempt in range(self.max_retries):
            try:
                resp = self.client.messages.create(model=self.model, max_tokens=self.max_tokens,
                                                   temperature=temperature, system=system or None,
                                                   messages=convo)
                return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            except Exception:  # noqa: BLE001 - retry transient errors
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
        return ""


class OpenAICompatClient:
    """Any /chat/completions endpoint. For a local vLLM server:
    llm("<served-model-id>", base_url="http://localhost:8000/v1")."""

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None,
                 max_tokens: int = 512, max_retries: int = 4, extra_body: dict | None = None,
                 timeout: float = 120.0):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"),
                             timeout=timeout)   # a hung call would otherwise stall a pool worker forever
        self.model = model
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.extra_body = extra_body   # vLLM knobs, e.g. {"chat_template_kwargs": {"enable_thinking": False}}

    def chat(self, messages: list[dict], temperature: float = 0.0) -> str:
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(model=self.model, messages=messages,
                                                           temperature=temperature, max_tokens=self.max_tokens,
                                                           extra_body=self.extra_body)
                return resp.choices[0].message.content or ""
            except Exception:  # noqa: BLE001
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
        return ""
