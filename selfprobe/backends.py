"""One OpenAI-compatible client, several providers, plus a local transformers fallback."""

from __future__ import annotations

import asyncio
import os
import random
from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    api_key_env: str
    supports_completions: bool  # /v1/completions, needed for the base-model logprob protocol
    supports_chat_logprobs: bool = True


PROVIDERS: dict[str, Provider] = {
    # Ollama returns top_logprobs on /v1/chat/completions but not on /v1/completions, so the
    # base-model arm needs vLLM.
    "ollama": Provider("ollama", os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
                       "OLLAMA_API_KEY", False),
    "vllm": Provider("vllm", os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
                     "VLLM_API_KEY", True),
    "openrouter": Provider("openrouter", "https://openrouter.ai/api/v1",
                           "OPENROUTER_API_KEY", False),
    "gemini": Provider("gemini", "https://generativelanguage.googleapis.com/v1beta/openai",
                       "GEMINI_API_KEY", False, supports_chat_logprobs=False),
    "groq": Provider("groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY", False),
    "cerebras": Provider("cerebras", "https://api.cerebras.ai/v1", "CEREBRAS_API_KEY", False),
}

RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class BackendError(RuntimeError):
    pass


class OpenAICompatBackend:
    def __init__(self, provider: str, model: str, *, client: httpx.AsyncClient,
                 max_retries: int = 5, timeout: float = 120.0) -> None:
        if provider not in PROVIDERS:
            raise KeyError(f"unknown provider {provider!r}; known: {sorted(PROVIDERS)}")
        self.provider = PROVIDERS[provider]
        self.model = model
        self._client = client
        self._max_retries = max_retries
        self._timeout = timeout

    @property
    def _headers(self) -> dict[str, str]:
        key = os.environ.get(self.provider.api_key_env, "") or "dummy"
        return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    async def _post(self, path: str, payload: dict) -> dict:
        url = f"{self.provider.base_url.rstrip('/')}{path}"
        last: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                r = await self._client.post(url, json=payload, headers=self._headers,
                                            timeout=self._timeout)
                if r.status_code in RETRYABLE_STATUS:
                    raise BackendError(f"HTTP {r.status_code}: {r.text[:300]}")
                if r.status_code >= 400:
                    raise BackendError(f"HTTP {r.status_code}: {r.text[:500]}")
                return r.json()
            except (httpx.TransportError, httpx.TimeoutException, BackendError) as exc:
                last = exc
                if isinstance(exc, BackendError) and "HTTP 4" in str(exc):
                    status = str(exc).split()[1].rstrip(":")
                    if status.isdigit() and int(status) not in RETRYABLE_STATUS:
                        raise
                # full jitter: free tiers rate-limit hard and synchronised retries make it worse
                await asyncio.sleep(random.uniform(0, min(2 ** attempt, 30)))
        raise BackendError(f"exhausted {self._max_retries} retries: {last}")

    async def chat(self, messages: list[dict], *, temperature: float = 1.0,
                   max_tokens: int = 256, seed: int | None = None) -> str:
        payload: dict = {"model": self.model, "messages": messages,
                         "temperature": temperature, "max_tokens": max_tokens}
        if seed is not None:
            payload["seed"] = seed
        data = await self._post("/chat/completions", payload)
        try:
            choice = data["choices"][0]["message"]
            # reasoning models may leave content empty when the budget went to thinking
            return (choice.get("content") or choice.get("reasoning_content") or "").strip()
        except (KeyError, IndexError) as exc:
            raise BackendError(f"unexpected chat response: {str(data)[:300]}") from exc

    async def chat_top_logprobs(self, messages: list[dict], top_k: int = 20) -> dict[str, float]:
        """Top-k logprobs of the first generated token. The prompt must be written so that
        token is the answer letter itself."""
        if not self.provider.supports_chat_logprobs:
            raise BackendError(f"provider {self.provider.name} returns no chat top_logprobs")
        data = await self._post("/chat/completions", {
            "model": self.model, "messages": messages, "temperature": 0.0,
            "max_tokens": 1, "logprobs": True, "top_logprobs": top_k,
        })
        try:
            top = data["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        except (KeyError, IndexError, TypeError) as exc:
            raise BackendError(f"no chat top_logprobs: {str(data)[:300]}") from exc
        out: dict[str, float] = {}
        for entry in top:
            key = str(entry["token"]).strip()
            if key not in out or entry["logprob"] > out[key]:
                out[key] = float(entry["logprob"])
        return out

    async def next_token_logprobs(self, prompt: str, top_k: int = 20) -> dict[str, float]:
        """Top-k logprobs at the next position. vLLM /v1/completions only."""
        if not self.provider.supports_completions:
            raise BackendError(f"provider {self.provider.name} has no /completions endpoint")
        data = await self._post("/completions", {
            "model": self.model, "prompt": prompt, "max_tokens": 1,
            "temperature": 0.0, "logprobs": top_k,
        })
        try:
            top = data["choices"][0]["logprobs"]["top_logprobs"][0]
        except (KeyError, IndexError, TypeError) as exc:
            raise BackendError(f"no top_logprobs: {str(data)[:300]}") from exc
        return {str(k): float(v) for k, v in top.items()}


_LOCAL_CACHE: dict[str, tuple] = {}


def _load_local(model: str):
    if model not in _LOCAL_CACHE:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.set_grad_enabled(False)
        tok = AutoTokenizer.from_pretrained(model)
        mdl = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float32).eval()
        _LOCAL_CACHE[model] = (tok, mdl)
    return _LOCAL_CACHE[model]


class LocalTransformersBackend:
    """Same interface, executed in-process. Slow, but needs no key, GPU or server."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.provider = Provider("local", "in-process", "", True)

    def _blocking_chat(self, messages, temperature, max_tokens, seed):
        import torch

        tok, mdl = _load_local(self.model)
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt")
        if seed is not None:
            torch.manual_seed(seed)
        out = mdl.generate(**ids, max_new_tokens=max_tokens, do_sample=temperature > 0,
                           temperature=temperature if temperature > 0 else None,
                           top_p=1.0, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    async def chat(self, messages, *, temperature=1.0, max_tokens=256, seed=None) -> str:
        return await asyncio.to_thread(self._blocking_chat, messages, temperature,
                                       max_tokens, seed)

    def _blocking_topk(self, ids_text, top_k: int, is_chat: bool):
        import torch

        tok, mdl = _load_local(self.model)
        text = (tok.apply_chat_template(ids_text, tokenize=False, add_generation_prompt=True)
                if is_chat else ids_text)
        ids = tok(text, return_tensors="pt")
        lp = torch.log_softmax(mdl(**ids).logits[0, -1, :].float(), dim=-1)
        top = torch.topk(lp, top_k)
        out: dict[str, float] = {}
        for i, v in zip(top.indices, top.values):
            key = tok.decode(i).strip()
            if key not in out or float(v) > out[key]:
                out[key] = float(v)
        return out

    async def chat_top_logprobs(self, messages, top_k: int = 20) -> dict[str, float]:
        return await asyncio.to_thread(self._blocking_topk, messages, top_k, True)

    async def next_token_logprobs(self, prompt: str, top_k: int = 20) -> dict[str, float]:
        return await asyncio.to_thread(self._blocking_topk, prompt, top_k, False)


def make_backend(provider: str, model: str, *, client: httpx.AsyncClient):
    if provider == "local":
        return LocalTransformersBackend(model)
    return OpenAICompatBackend(provider, model, client=client)
