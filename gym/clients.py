"""
LLM clients. Every client implements one method:

    chat(messages: list[dict], temperature: float = 0.0) -> str

So the generator and judge never care which model is behind it. Clients here:

  MockClient            deterministic, no network — for tests and dry runs
  AnthropicClient       Claude (api.anthropic.com)
  OpenAICompatClient    anything speaking the OpenAI /chat/completions schema:
                        OpenAI, Together, Fireworks, DeepInfra, or a *local*
                        vLLM / TGI / Ollama server hosting Qwen3-4B
  HFQwen3Client         Qwen3 via huggingface_hub InferenceClient (rate-limited)
  EnsembleClient        majority/median vote over several judges (Adnan's idea)

For the team setup, the recommended judge is Qwen3-4B served locally with vLLM
(free, fast, reproducible) via OpenAICompatClient pointed at localhost.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter


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
    """
    Claude judge/generator. Default model is Haiku — by far the cheapest option
    and plenty strong for pairwise relevance judgements. Swap to Sonnet/Opus via
    `model=` if you want a stronger judge for the validation table.
    """

    def __init__(self, model: str = "claude-haiku-4-5-20251001",
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
    to run Qwen3-4B for the team: serve it with vLLM and point base_url at it.

        # on the GPU box:
        vllm serve Qwen/Qwen3-4B-Instruct-2507 --port 8000

        # in code:
        judge = OpenAICompatClient(
            model="Qwen/Qwen3-4B-Instruct-2507",
            base_url="http://localhost:8000/v1",
            api_key="EMPTY",
        )
    """

    def __init__(self, model: str, base_url: str | None = None,
                 api_key: str | None = None, max_tokens: int = 512,
                 max_retries: int = 4, extra_body: dict | None = None):
        from openai import OpenAI
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"),
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


# --------------------------------------------------------------------------- #
# HF InferenceClient Qwen3 (convenient but rate-limited on the free tier)     #
# --------------------------------------------------------------------------- #
class HFQwen3Client:
    """Qwen3 via huggingface_hub. Handy for a quick try; rate-limited when free."""

    def __init__(self, model: str = "Qwen/Qwen3-4B-Instruct-2507",
                 token: str | None = None, max_tokens: int = 512):
        from huggingface_hub import InferenceClient
        self.client = InferenceClient(model=model, token=token or os.environ.get("HF_TOKEN"))
        self.model = model
        self.max_tokens = max_tokens

    def chat(self, messages: list[dict], temperature: float = 0.0) -> str:
        resp = self.client.chat_completion(
            messages=messages, max_tokens=self.max_tokens,
            temperature=max(temperature, 1e-3),
        )
        return resp.choices[0].message.content or ""


# --------------------------------------------------------------------------- #
# Ensemble (Adnan's suggestion)                                               #
# --------------------------------------------------------------------------- #
class EnsembleClient:
    """
    Runs several judges and combines their JSON verdicts. We don't try to parse
    here — we just hand back the *individual* responses joined, and let the judge
    layer aggregate (it already knows the verdict schema). Use `vote()` for a
    clean majority over already-parsed winners.
    """

    def __init__(self, clients: list):
        if not clients:
            raise ValueError("EnsembleClient needs at least one client")
        self.clients = clients

    def chat(self, messages: list[dict], temperature: float = 0.0) -> str:
        # Return a JSON array of member responses; judge layer handles it.
        return json.dumps([c.chat(messages, temperature) for c in self.clients])

    @staticmethod
    def vote(winners: list[str]) -> tuple[str, float]:
        """Majority winner + agreement fraction. Ties broken to 'tie'."""
        counts = Counter(winners)
        top, n = counts.most_common(1)[0]
        agreement = n / len(winners)
        # If the top two are tied in count, the panel disagrees -> tie.
        most = counts.most_common(2)
        if len(most) > 1 and most[0][1] == most[1][1]:
            return "tie", agreement
        return top, agreement
