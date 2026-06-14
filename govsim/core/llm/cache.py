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
from dataclasses import asdict
from pathlib import Path
from typing import Any

from govsim.core.llm.client import LLMResponse

SCHEMA_VERSION = "2"  # bumped: cache key now includes max_tokens + provider extra (e.g. reasoning_effort)


class CachingReplayClient:
    def __init__(self, inner: Any, cache_dir: str | Path, mode: str = "cache") -> None:
        if mode not in ("live", "cache", "replay"):
            raise ValueError("mode must be one of: live | cache | replay")
        self.inner = inner
        self.dir = Path(cache_dir)
        self.mode = mode
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
            data["cached"] = True
            return LLMResponse(**data)

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
        stored.pop("cached", None)  # never persist the cached flag
        path.write_text(json.dumps(stored, ensure_ascii=False, indent=2), encoding="utf-8")
        return resp
