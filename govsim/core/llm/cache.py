"""
CachingReplayClient — an on-disk cache + replay tape wrapping any ``LLMClient``.

This is the single cheapest load-bearing fix for *both* reproducibility and cost (doc-08 §3.2):
seeding numpy/``random`` is meaningless while the decisive actor is a live, drifting LLM. With a
replay tape, an LLM-in-the-loop experiment becomes reproducible (replay from disk, never call) and
ablation re-runs become free (paired, cache-served, so the only varying source is the world seed).

Modes:
  - ``live``   : always call the inner client and persist the result.
  - ``cache``  : serve from disk on a hit; otherwise call + persist. (default)
  - ``replay`` : serve from disk only; raise on a miss (never calls the network).

The cache key includes the messages, model, temperature, seed, tools, response_format AND a
``SCHEMA_VERSION`` — so editing a prompt or a tool schema invalidates stale replays rather than
silently serving a mismatched response.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from govsim.core.llm.client import LLMResponse, response_is_collapsed

SCHEMA_VERSION = "2"  # bumped: cache key now includes max_tokens + provider extra (e.g. reasoning_effort)


class CachingReplayClient:
    def __init__(self, inner: Any, cache_dir: str | Path, mode: str = "cache",
                 repair_collapses: bool = True) -> None:
        if mode not in ("live", "cache", "replay"):
            raise ValueError("mode must be one of: live | cache | replay")
        self.inner = inner
        self.dir = Path(cache_dir)
        self.mode = mode
        #: Treat a cached reasoning collapse as a MISS in ``cache`` mode, so the inner client's
        #: mitigation can run. Set False to study the raw recorded rate.
        self.repair_collapses = repair_collapses
        #: How many cache hits were discarded as collapses. Reported, not hidden: it is the number
        #: of decisions a re-run actually repaired.
        self.collapsed_hits = 0
        self.dir.mkdir(parents=True, exist_ok=True)

    def _key(
        self,
        messages: list[dict[str, Any]],
        model: str | None,
        temperature: float,
        seed: int | None,
        tools: list[dict[str, Any]] | None,
        response_format: dict[str, Any] | None,
        max_tokens: int | None,
        extra: dict[str, Any] | None,
    ) -> str:
        blob = json.dumps(
            {
                "messages": messages,
                "model": model,
                "temperature": temperature,
                "seed": seed,
                "tools": tools,
                "response_format": response_format,
                "max_tokens": max_tokens,
                "extra": extra,
                "schema_version": SCHEMA_VERSION,
            },
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

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
        key = self._key(messages, model, temperature, seed, tools, response_format, max_tokens, extra)
        path = self._path(key)

        if self.mode != "live" and path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            data.pop("cache_key", None)  # derived, never trusted from disk
            data["cached"] = True
            hit = LLMResponse(**data, cache_key=key)
            # A cached REASONING COLLAPSE is re-issued rather than served, because the mitigation
            # that repairs it lives in the inner client and a cache hit never reaches it. Without
            # this, re-running a contaminated arm to repair it would replay the identical collapses
            # from disk and change nothing — the arm would look repaired and be exactly as broken.
            #
            # NOT done in replay mode: a recorded tape must reproduce byte-for-byte, including its
            # failures, or "replayable without an API key" stops being true.
            if (self.mode == "cache" and self.repair_collapses
                    and response_is_collapsed(hit.text, hit.tool_calls, hit.usage)):
                self.collapsed_hits += 1
            else:
                return hit

        if self.mode == "replay":
            raise KeyError(
                f"replay miss: no cached LLM response for key {key[:12]}… "
                "(run once in 'cache'/'live' mode to record the tape)."
            )

        resp = self.inner.complete(
            messages,
            model=model,
            temperature=temperature,
            seed=seed,
            tools=tools,
            response_format=response_format,
            max_tokens=max_tokens,
            extra=extra,
        )
        stored = asdict(resp)
        stored.pop("cached", None)     # never persist the cached flag …
        stored.pop("cache_key", None)  # … nor the key, which is the filename
        path.write_text(json.dumps(stored, ensure_ascii=False, indent=2), encoding="utf-8")
        return replace(resp, cache_key=key)
