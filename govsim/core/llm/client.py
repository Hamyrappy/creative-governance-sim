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


def response_is_collapsed(text: str, tool_calls: Any, usage: dict[str, Any] | None) -> bool:
    """Reasoning collapse: the call cost tokens and produced no completion.

    The signature is ``completion_tokens == 0`` with a positive ``total_tokens``. The whole budget
    went into the reasoning channel and nothing was emitted as an answer.

    ``text`` is deliberately NOT required to be empty, and getting this wrong made an earlier
    version of this predicate almost inert — it fired on 1 cache entry instead of 1003. The provider
    leaks the reasoning into the ``content`` field while still reporting ``completion_tokens = 0``,
    so a collapsed call arrives with tens of thousands of characters of ``<thought>`` in ``text``,
    ending mid-loop. Judging by "is the text empty" therefore misses every real case.

    Validated against the recorded tape, where the separation is essentially perfect:

        completion_tokens == 0   ->     0 produced an action,  1003 did not
        completion_tokens  > 0   ->  6389 produced an action,    44 did not

    A genuine ``tool_calls`` payload still counts as having answered, and is kept as a guard: it
    costs nothing and protects against a provider that reports usage differently.

    Shared by :class:`OpenAICompatClient` (which must catch it live) and
    :class:`~govsim.core.llm.cache.CachingReplayClient` (which must not silently re-serve one), so
    the two cannot drift on what counts as a collapse.
    """
    # The field must be PRESENT. Defaulting a missing ``completion_tokens`` to 0 would classify
    # every response from a provider that reports only ``total_tokens`` as a collapse, and the
    # mitigation would then re-ask on every single call. Absent evidence, do not diagnose.
    if not usage or usage.get("completion_tokens") is None:
        return False
    try:
        completion = int(usage["completion_tokens"])
        total = int(usage.get("total_tokens") or 0)
    except (TypeError, ValueError):
        return False
    if completion != 0 or total <= 0:
        return False
    return not tool_calls


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
        # A reasoning model given a large token budget spends it: measured ~99 s per call at
        # max_tokens=4000, and longer for the tail that needs more. The old 120 s default sat just
        # above the median, so the slow tail died with APITimeoutError — which looks like a flaky
        # network and is actually a budget mismatch.
        timeout: float = 600.0,
        drop_params: frozenset[str] | set[str] | None = None,
        max_retries: int = 6,
        min_interval: float = 0.0,
        # How many times to re-ask when the model burns its whole budget on reasoning and returns
        # nothing. 0 disables the mitigation entirely, which is what an experiment measuring the
        # RAW collapse rate wants.
        deliberation_retries: int = 2,
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
        self.deliberation_retries = deliberation_retries
        #: Decisions that collapsed into pure deliberation. Read per arm and REPORTED — the rate is
        #: a finding about the channel, not a defect to be quietly repaired.
        self.deliberation_collapses = 0
        #: Re-ask calls actually spent. Separate from the collapse count so a budget-matching claim
        #: can be checked: a mitigation that silently doubles one arm's call budget is a confound of
        #: its own.
        self.deliberation_retries_used = 0
        #: Collapses the re-asks failed to recover. These decisions really are no-ops and must be
        #: subtracted from an arm's effective treatment, not counted as governed.
        self.deliberation_unrecovered = 0
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
        # A 400 is normally a malformed request and must NOT be retried — retrying one is how a
        # sweep burns hours on a bug. The single exception is the provider's geo-routing
        # precondition, which we observed appear and clear within seconds on an otherwise identical
        # request. It is matched on the message rather than the status so no other 400 is caught.
        if str(status) == "400" and "location is not supported" in str(exc).lower():
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

    #: Appended verbatim when a call collapses into deliberation and returns no answer. Deliberately
    #: content-free about *what* to decide — it must not push the model toward any particular policy,
    #: or the mitigation becomes a treatment.
    _ANSWER_NOW = (
        "You have already reasoned about this at length. Do not deliberate further. "
        "Emit your action now, in the required form, and nothing else."
    )

    @staticmethod
    def _is_reasoning_collapse(resp: Any) -> bool:
        """Did the model spend its whole budget thinking and return no answer at all?

        The signature is ``completion_tokens == 0`` with a large ``total_tokens``: everything went
        into the reasoning channel and nothing reached the completion. MEASURED on
        ``gemma-4-31b-it`` at ``max_tokens=12000``: 31 of 400 decisions in the outcome-feedback arm,
        each ending inside a verbatim repetition loop ("Wait, let's try `0.9 * (I / 0.06)` and
        `0.5`." repeated until the budget ran out).

        This is NOT the truncation failure this project hit before, and the distinction matters
        because the remedies are opposite. Truncation at ``max_tokens=1500`` was a budget problem
        and raising the cap fixed it. This happens AT a 12000-token cap, and raising it further only
        buys more loop. It is a degenerate decoding state; the only way out is to re-ask.
        """
        try:
            usage = resp.usage
            as_dict = {
                "completion_tokens": getattr(usage, "completion_tokens", 0),
                "total_tokens": getattr(usage, "total_tokens", 0),
            }
            msg = resp.choices[0].message
        except Exception:  # noqa: BLE001 - a provider without usage cannot be diagnosed
            return False
        # One shared predicate with the cache layer, so the two cannot drift on what a collapse is.
        return response_is_collapsed(msg.content or "", getattr(msg, "tool_calls", None), as_dict)

    def _retry_after_collapse(self, client: Any, kwargs: dict[str, Any],
                              messages: list[dict[str, Any]]) -> Any:
        """Re-ask, telling the model to stop deliberating, with a hard cap on attempts.

        Why this is a mitigation rather than a thumb on the scale: collapse is *correlated with the
        treatment* — a harness channel lengthens the prompt and invites longer deliberation, so the
        arms carrying more information collapse more often (measured: 7.8% for outcome feedback,
        0.2% for no harness). Left alone, the decision silently becomes a no-op, the previously
        installed law stays in force, and the arm is no longer the treatment its label claims. The
        ablation would then be measuring which channel makes the model loop.

        The nudge says only "stop and answer"; it never suggests what to answer, so it cannot move
        the policy toward any particular choice. The retry COUNT is recorded and reported per arm,
        because the collapse rate is itself a finding about the channel and must not be silently
        repaired away.
        """
        # ONE increment per collapsed decision, not per retry attempt: the reported quantity is the
        # share of DECISIONS that collapsed, so counting attempts would inflate an arm's rate purely
        # because its collapses were harder to recover from.
        self.deliberation_collapses += 1
        resp = None
        for _ in range(self.deliberation_retries):
            retry_kwargs = dict(kwargs)
            retry_kwargs["messages"] = list(messages) + [
                {"role": "user", "content": self._ANSWER_NOW}
            ]
            # The smaller budget is part of the fix, not an economy: the failure is an unbounded
            # reasoning loop, and a tight ceiling forces the decode out of it.
            retry_kwargs["max_tokens"] = min(int(kwargs.get("max_tokens") or 1024), 1024)
            self.deliberation_retries_used += 1
            resp = self._call_with_retry(client, retry_kwargs)
            if not self._is_reasoning_collapse(resp):
                return resp
        self.deliberation_unrecovered += 1
        return resp

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
        if self.deliberation_retries and self._is_reasoning_collapse(resp):
            resp = self._retry_after_collapse(client, kwargs, messages)
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
