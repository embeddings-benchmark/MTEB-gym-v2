"""
LLM clients. Every client implements one method:

    chat(messages: list[dict], temperature: float = 0.0) -> str

so the generator and judge never care which model is behind it:

  MockClient            deterministic, no network -- tests and dry runs
  AnthropicClient       Claude (api.anthropic.com)
  OpenAICompatClient    anything speaking the OpenAI /chat/completions schema:
                        OpenAI, Together, Fireworks, or a local vLLM/TGI server
"""

from __future__ import annotations

import hashlib
import json
import os
import time


# --------------------------------------------------------------------------- #
# Mock                                                                        #
# --------------------------------------------------------------------------- #
class MockClient:
    """
    Deterministic pseudo-LLM. Same input -> same output, so test runs are
    reproducible and free. Returns plausible JSON for both generation and
    judging prompts by sniffing the prompt text.
    """

    def __init__(self, seed: int = 0):
        self.seed = seed

    def _hash(self, text: str) -> int:
        h = hashlib.sha256(f"{self.seed}:{text}".encode()).hexdigest()
        return int(h[:8], 16)

    def chat(self, messages: list[dict], temperature: float = 0.0) -> str:
        prompt = " ".join(m.get("content", "") for m in messages)
        h = self._hash(prompt)
        if "rate the quality" in prompt.lower() or "quality score" in prompt.lower():
            return json.dumps({"score": (h % 5) + 1, "reason": "mock"})
        if "system a" in prompt.lower() or "result set" in prompt.lower():
            winner = ["A", "B", "tie"][h % 3]
            return json.dumps({"winner": winner, "confidence": "low", "reasoning": "mock"})
        # generation
        return json.dumps({"query": f"what is the relationship between factor {h % 1000} "
                                    f"and clinical outcomes in affected patients"})


# --------------------------------------------------------------------------- #
# Anthropic                                                                   #
# --------------------------------------------------------------------------- #
class AnthropicClient:
    """Claude judge/generator via the Anthropic API."""

    def __init__(self, model: str,
                 max_tokens: int = 512, api_key: str | None = None,
                 max_retries: int = 4):
        import anthropic
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model
        self.max_tokens = max_tokens
        self.max_retries = max_retries

    def chat(self, messages: list[dict], temperature: float = 0.0) -> str:
        # Pull out a system message if present (Anthropic takes it separately).
        system = ""
        convo = []
        for m in messages:
            if m["role"] == "system":
                system += m["content"] + "\n"
            else:
                convo.append({"role": m["role"], "content": m["content"]})
        for attempt in range(self.max_retries):
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=temperature,
                    system=system.strip() or None,
                    messages=convo,
                )
                return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            except Exception as e:  # noqa: BLE001 - retry transient errors
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
        return ""


# --------------------------------------------------------------------------- #
# OpenAI-compatible (OpenAI, Together, vLLM, Ollama, ...)                      #
# --------------------------------------------------------------------------- #
class OpenAICompatClient:
    """
    Talks to any OpenAI /chat/completions endpoint. This is the recommended way
    to run a local judge: serve it with vLLM and point base_url at it.

        # on the GPU box:
        vllm serve <served-model-id> --port 8000

        # in code:
        judge = OpenAICompatClient(
            model="<served-model-id>",
            base_url="http://localhost:8000/v1",
            api_key="EMPTY",
        )
    """

    def __init__(self, model: str, base_url: str | None = None,
                 api_key: str | None = None, max_tokens: int = 512,
                 max_retries: int = 4, extra_body: dict | None = None,
                 timeout: float = 120.0):
        from openai import OpenAI
        # a hung HTTP call would otherwise stall a pool worker forever
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"),
            timeout=timeout,
        )
        self.model = model
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        # vLLM knobs, e.g. {"chat_template_kwargs": {"enable_thinking": False}}
        self.extra_body = extra_body

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
            except Exception:  # noqa: BLE001
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)
        return ""


