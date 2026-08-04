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
import re
import threading
import time
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
    #: The content-addressed cache key this response was stored/served under. Set by
    #: ``CachingReplayClient`` and recorded by the regents, so a reported run's tape can be exported
    #: exactly (rather than reconstructed from an I/O log that does not keep every request knob),
    #: and so a replay miss names the entry it wanted.
    cache_key: str = ""


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
        max_retries: int = 6,
        min_interval: float = 0.0,
    ) -> None:
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.default_model = default_model
        self.timeout = timeout
        self.drop_params = frozenset(drop_params or ())
        # Free and shared endpoints rate-limit aggressively (Gemini's free tier is 15 requests per
        # minute per model). A multi-seed experiment is thousands of calls, so a 429 partway through
        # must cost a pause, not the run. ``min_interval`` paces proactively; ``max_retries`` with
        # server-suggested backoff recovers from the bursts that slip through.
        self.max_retries = max_retries
        self.min_interval = min_interval
        self._last_call_at = 0.0
        self._lock = threading.Lock()
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

    _RETRY_AFTER_RE = re.compile(r"retry(?:Delay|-after)['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)", re.I)

    @classmethod
    def _is_retryable(cls, exc: Exception) -> bool:
        """True for rate limits and transient server errors, by duck-typing rather than by isinstance.

        The concrete exception classes live in the lazily-imported ``openai`` package, and importing
        them here to catch them would reintroduce the import-time dependency this seam exists to
        avoid — CI runs with no key and, on a minimal install, no SDK.
        """
        status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        if status in (408, 409, 429, 500, 502, 503, 504) or str(status) in ("429", "503"):
            return True
        name = type(exc).__name__
        return any(k in name for k in ("RateLimit", "APIConnection", "APITimeout", "InternalServer"))

    @classmethod
    def _suggested_delay(cls, exc: Exception) -> float | None:
        """The server's own retry hint, if it published one (Gemini returns ``retryDelay: '6s'``)."""
        m = cls._RETRY_AFTER_RE.search(str(exc))
        return float(m.group(1)) if m else None

    def _pace(self) -> None:
        """Hold the configured minimum gap between outbound calls."""
        if self.min_interval <= 0:
            return
        with self._lock:
            wait = self.min_interval - (time.monotonic() - self._last_call_at)
            if wait > 0:
                time.sleep(wait)
            self._last_call_at = time.monotonic()

    def _call_with_retry(self, client: Any, kwargs: dict[str, Any]) -> Any:
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._pace()
            try:
                return client.chat.completions.create(**kwargs)
            except Exception as e:  # noqa: BLE001 - re-raised below unless retryable
                if attempt >= self.max_retries or not self._is_retryable(e):
                    raise
                last = e
                # Server hint if there is one, else exponential backoff. The offset keeps several
                # arms started at the same moment from re-colliding on every retry; it is derived
                # from the pid rather than drawn, both because the global RNG is banned under
                # govsim/core (it would break clone/rollout reproducibility) and because a
                # per-process constant decorrelates parallel runs more reliably than a shared draw.
                delay = self._suggested_delay(e)
                if delay is None:
                    delay = min(60.0, 2.0 * (2 ** attempt))
                time.sleep(delay + (os.getpid() % 100) / 100.0)
        raise last if last else RuntimeError("unreachable")

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

        resp = self._call_with_retry(client, kwargs)
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
