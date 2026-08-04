"""
LLMClient — a thin, provider-agnostic seam over the OpenAI-*compatible* chat-completions API.

"OpenAI-compatible" means the wire format, not the vendor: the same client talks to OpenAI,
OpenRouter, Together, a local vLLM / Ollama / LM Studio server, or any proxy — selected purely by
``base_url`` + ``model`` + which API-key env var to read. Nothing here hardcodes a vendor or model.

Design rules (doc-08 / doc-09):
  - LAZY: no network, no key access, no ``openai`` import at module import time — fail on first
    ``complete`` call instead, so CI/tests run without a key or the package installed.
  - The concrete client is dumb (one ``complete`` call); reproducibility/caching is a separate
    wrapper (``CachingReplayClient``), kept orthogonal so any client can be made replayable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class LLMResponse:
    """A normalized response. ``raw`` keeps the full provider payload for the RunRecord."""

    text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)  # [{id, name, arguments(str)}]
    model: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
    cost_usd: float | None = None
    cached: bool = False


@runtime_checkable
class LLMClient(Protocol):
    """Anything that can turn chat messages (+ optional tools) into an ``LLMResponse``."""

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        seed: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> LLMResponse: ...


class OpenAICompatClient:
    """Concrete client for any OpenAI-compatible endpoint.

    Args:
        base_url: e.g. ``https://api.openai.com/v1`` | ``https://openrouter.ai/api/v1`` |
            ``http://localhost:11434/v1`` (Ollama) | ``http://localhost:8000/v1`` (vLLM) |
            ``https://generativelanguage.googleapis.com/v1beta/openai/`` (Gemini).
            ``None`` uses the ``openai`` SDK default.
        api_key_env: environment variable holding the key (default ``OPENAI_API_KEY``).
        default_model: model id used when ``complete(model=...)`` is not given.
        drop_params: wire-level parameters this endpoint does NOT accept, stripped before the call.
            "OpenAI-compatible" is a family, not a standard: Gemini's compat layer hard-rejects
            ``seed`` with a 400. Dropping is deliberately wire-only — the caller's ``seed`` still
            enters the cache key, so a recorded tape stays keyed by the *logical* request and
            replays byte-for-byte. (Cost: the provider no longer honours the seed, so live sampling
            diversity across probe candidates must come from the prompt, not the seed field.)
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        default_model: str | None = None,
        timeout: float = 120.0,
        drop_params: frozenset[str] | set[str] | None = None,
    ) -> None:
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.default_model = default_model
        self.timeout = timeout
        self.drop_params = frozenset(drop_params or ())
        self._client: Any = None  # the openai.OpenAI instance, created lazily

    def _ensure(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI  # imported lazily on first use
            except ImportError as e:  # pragma: no cover - environment-dependent
                raise RuntimeError(
                    "The 'openai' package is required for OpenAICompatClient "
                    "(`poetry add openai`)."
                ) from e
            key = os.environ.get(self.api_key_env)
            if not key:
                raise RuntimeError(
                    f"LLM API key env var '{self.api_key_env}' is not set. "
                    "Set it, or use replay mode (no live calls)."
                )
            self._client = OpenAI(api_key=key, base_url=self.base_url, timeout=self.timeout)
        return self._client

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        seed: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> LLMResponse:
        client = self._ensure()
        chosen = model or self.default_model
        if not chosen:
            raise ValueError("No model specified (pass model=... or set default_model).")
        kwargs: dict[str, Any] = {"model": chosen, "messages": messages, "temperature": temperature}
        if seed is not None:
            kwargs["seed"] = seed
        if tools:
            kwargs["tools"] = tools
        if response_format:
            kwargs["response_format"] = response_format
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if extra:
            # Provider-specific top-level params (e.g. gpt-oss ``reasoning_effort="low"`` to stop it
            # over-thinking and returning empty content). Verified as a direct kwarg on the AIRI vLLM.
            kwargs.update(extra)
        for name in self.drop_params:
            kwargs.pop(name, None)

        resp = client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        tool_calls: list[dict[str, Any]] = []
        for tc in (getattr(msg, "tool_calls", None) or []):
            tool_calls.append(
                {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
            )
        usage = resp.usage.model_dump() if getattr(resp, "usage", None) is not None else {}
        raw = resp.model_dump() if hasattr(resp, "model_dump") else {}
        return LLMResponse(
            text=msg.content or "",
            tool_calls=tool_calls,
            model=getattr(resp, "model", chosen),
            usage=usage,
            raw=raw,
        )
