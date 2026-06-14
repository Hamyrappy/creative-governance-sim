"""
govsim.core.llm — the LLM seam.

An OpenAI-*compatible* client (``OpenAICompatClient``): any endpoint / provider / local model,
configured by ``base_url`` + ``model`` (never hardcoded to OpenAI-the-company), wrapped by a
``CachingReplayClient`` so an LLM-in-the-loop experiment is reproducible (replay from tape) and
cheap to re-run (cache hits). Replaces the deleted Gemini/LangChain stack.
"""

from govsim.core.llm.client import LLMClient, LLMResponse, OpenAICompatClient
from govsim.core.llm.cache import CachingReplayClient, SCHEMA_VERSION

__all__ = [
    "LLMClient",
    "LLMResponse",
    "OpenAICompatClient",
    "CachingReplayClient",
    "SCHEMA_VERSION",
]
