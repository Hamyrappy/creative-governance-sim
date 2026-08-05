"""Detecting and recovering from reasoning collapse — the model that thinks until it forgets to act.

MEASURED on ``gemma-4-31b-it`` at ``max_tokens=12000``: 31 of 400 decisions in the outcome-feedback
arm returned ``completion_tokens=0`` with ``total_tokens≈12862``, each ending inside a verbatim
repetition loop ("Wait, let's try `0.9 * (I / 0.06)` and `0.5`." repeated until the budget ran out).
The no-harness arm collapsed on 1 of 400.

That correlation is the whole reason this exists. A collapsed decision is a silent no-op — the
previously installed law stays in force — so the arms carrying more information quietly become
"stickier policy" than the arms carrying less, and the ablation measures which channel makes the
model loop. It must be detected, mitigated identically across arms, and COUNTED.

The distinction from plain truncation is load-bearing and the remedies are opposite: truncation at
``max_tokens=1500`` was a budget problem that a larger cap fixed; this happens AT a 12000-token cap
and a larger cap only buys more loop.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from govsim.core.llm.client import (
    LLMResponse,
    OpenAICompatClient,
    response_is_collapsed,
)


def _resp(*, completion: int, total: int, content: str = "", tool_calls=None):
    """A minimal duck-typed stand-in for an OpenAI chat-completion response."""
    return SimpleNamespace(
        usage=SimpleNamespace(completion_tokens=completion, total_tokens=total),
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))],
    )


# --- what collapse is, and what it is not -------------------------------------------------------

def test_the_measured_collapse_signature_is_detected():
    """completion_tokens=0 with a large total and nothing in the message."""
    assert OpenAICompatClient._is_reasoning_collapse(_resp(completion=0, total=12862))


def test_a_normal_answer_is_not_collapse():
    assert not OpenAICompatClient._is_reasoning_collapse(
        _resp(completion=57, total=922, content="set_lockdown: 0.9 if I > 0.02 else 0.0")
    )


def test_a_tool_call_with_no_text_is_not_collapse():
    """The tool-call path legitimately returns empty content. Flagging it would re-ask on every
    well-formed decision and double the call budget of the best-behaved arms."""
    assert not OpenAICompatClient._is_reasoning_collapse(
        _resp(completion=0, total=900, tool_calls=[{"id": "1", "name": "set_lockdown"}])
    )


def test_text_without_usage_accounting_is_not_collapse():
    """Some providers omit usage. Absent evidence, do not re-ask — an unnecessary retry costs a call
    and, worse, appends a nudge that was never warranted."""
    assert not OpenAICompatClient._is_reasoning_collapse(
        _resp(completion=0, total=0, content="set_lockdown: 0.5")
    )


def test_leaked_reasoning_in_the_content_field_IS_still_collapse():
    """The case that actually occurs, and the one an earlier predicate missed.

    The provider puts the whole ``<thought>`` into ``content`` while still reporting
    ``completion_tokens = 0``. Requiring empty text made the detector fire on 1 cache entry instead
    of 1003 — validated against the tape, where completion_tokens==0 predicted "produced no action"
    on 1003 of 1003 calls.
    """
    thought = "<thought>" + "Wait, let's try `0.9 * (I / 0.06)` and `0.5`.\n" * 400
    assert OpenAICompatClient._is_reasoning_collapse(
        _resp(completion=0, total=12862, content=thought)
    )


def test_a_provider_with_no_usage_object_is_handled():
    assert not OpenAICompatClient._is_reasoning_collapse(
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="x", tool_calls=None))])
    )


# --- the mitigation -----------------------------------------------------------------------------

class _FakeClient:
    """Returns collapses until ``succeed_after`` calls, then a real answer. Records every request."""

    def __init__(self, succeed_after: int) -> None:
        self.succeed_after = succeed_after
        self.requests: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(kwargs)
        if len(self.requests) > self.succeed_after:
            return _resp(completion=40, total=900, content="set_lockdown: 0.4")
        return _resp(completion=0, total=12000)


def _client(retries: int = 2) -> OpenAICompatClient:
    c = OpenAICompatClient(default_model="m", deliberation_retries=retries, max_retries=0)
    return c


def test_a_collapse_is_re_asked_and_recovers():
    c = _client()
    fake = _FakeClient(succeed_after=1)
    msgs = [{"role": "user", "content": "govern"}]
    resp = c._retry_after_collapse(fake, {"model": "m", "messages": msgs, "max_tokens": 12000}, msgs)
    assert resp.choices[0].message.content == "set_lockdown: 0.4"
    assert c.deliberation_collapses == 1


def test_the_nudge_is_appended_and_says_nothing_about_what_to_decide():
    """If the retry hinted at a policy it would be a treatment, not a mitigation — and it would be
    applied more often to the arms carrying more information."""
    c = _client()
    fake = _FakeClient(succeed_after=1)
    msgs = [{"role": "user", "content": "govern"}]
    c._retry_after_collapse(fake, {"model": "m", "messages": msgs, "max_tokens": 12000}, msgs)
    sent = fake.requests[0]["messages"]
    assert sent[:-1] == msgs, "the original prompt must be preserved verbatim"
    nudge = sent[-1]["content"]
    for leading in ("lockdown", "vacc", "0.", "increase", "decrease", "raise", "lower"):
        assert leading not in nudge.lower(), nudge


def test_the_retry_budget_is_tightened_to_break_the_loop():
    """The failure is an unbounded reasoning loop; re-asking with the same huge ceiling invites the
    same loop again."""
    c = _client()
    fake = _FakeClient(succeed_after=1)
    msgs = [{"role": "user", "content": "govern"}]
    c._retry_after_collapse(fake, {"model": "m", "messages": msgs, "max_tokens": 12000}, msgs)
    assert fake.requests[0]["max_tokens"] <= 1024


def test_retries_are_bounded_and_the_count_is_recorded():
    """A model that always collapses must not spin forever, and the rate must survive as a number:
    it is a finding about the channel, not a defect to repair silently."""
    c = _client(retries=2)
    fake = _FakeClient(succeed_after=99)
    msgs = [{"role": "user", "content": "govern"}]
    resp = c._retry_after_collapse(fake, {"model": "m", "messages": msgs, "max_tokens": 12000}, msgs)
    assert len(fake.requests) == 2
    assert c.deliberation_collapses == 1, "one collapsed DECISION, however many re-asks it took"
    assert c.deliberation_retries_used == 2
    assert c.deliberation_unrecovered == 1
    assert OpenAICompatClient._is_reasoning_collapse(resp), "an unrecovered collapse is returned as-is"


def test_the_mitigation_can_be_switched_off_entirely():
    """An experiment measuring the RAW collapse rate must be able to see it unrepaired."""
    assert OpenAICompatClient(default_model="m", deliberation_retries=0).deliberation_retries == 0


# --- the cache must not silently re-serve a collapse ---------------------------------------------

def test_a_cached_collapse_is_re_issued_in_cache_mode(tmp_path):
    """Otherwise re-running a contaminated arm repairs nothing.

    The mitigation lives in the inner client, and a cache hit never reaches it. Replaying the
    identical collapses from disk would leave the arm exactly as broken while looking repaired —
    which is worse than not re-running at all, because the re-run is the evidence of repair.
    """
    from govsim.core.llm.cache import CachingReplayClient

    class _Inner:
        def __init__(self):
            self.calls = 0

        def complete(self, messages, **kw):
            self.calls += 1
            return LLMResponse(text="set_lockdown: 0.3", usage={"completion_tokens": 12,
                                                                "total_tokens": 400})

    inner = _Inner()
    c = CachingReplayClient(inner, tmp_path, mode="cache")
    msgs = [{"role": "user", "content": "govern"}]
    key = c._key(msgs, None, 0.0, None, None, None, None, None)
    # Seed the cache with a recorded collapse, exactly as the live sweep wrote one.
    (tmp_path / f"{key}.json").write_text(json.dumps({
        "text": "", "tool_calls": [], "model": "m",
        "usage": {"completion_tokens": 0, "total_tokens": 12862}, "raw": {}, "cost_usd": None,
    }), encoding="utf-8")

    got = c.complete(msgs)
    assert got.text == "set_lockdown: 0.3", "the collapse should have been re-issued, not served"
    assert inner.calls == 1
    assert c.collapsed_hits == 1


def test_replay_mode_still_reproduces_a_recorded_collapse_byte_for_byte(tmp_path):
    """A tape must replay exactly, failures included, or 'replayable without an API key' is false."""
    from govsim.core.llm.cache import CachingReplayClient

    class _Boom:
        def complete(self, *a, **k):
            raise AssertionError("replay mode must never call the network")

    c = CachingReplayClient(_Boom(), tmp_path, mode="replay")
    msgs = [{"role": "user", "content": "govern"}]
    key = c._key(msgs, None, 0.0, None, None, None, None, None)
    (tmp_path / f"{key}.json").write_text(json.dumps({
        "text": "", "tool_calls": [], "model": "m",
        "usage": {"completion_tokens": 0, "total_tokens": 12862}, "raw": {}, "cost_usd": None,
    }), encoding="utf-8")

    got = c.complete(msgs)
    assert got.text == "" and got.cached
    assert response_is_collapsed(got.text, got.tool_calls, got.usage)


def test_repair_can_be_disabled_to_measure_the_raw_recorded_rate(tmp_path):
    from govsim.core.llm.cache import CachingReplayClient

    class _Boom:
        def complete(self, *a, **k):
            raise AssertionError("must not be called when repair is off")

    c = CachingReplayClient(_Boom(), tmp_path, mode="cache", repair_collapses=False)
    msgs = [{"role": "user", "content": "govern"}]
    key = c._key(msgs, None, 0.0, None, None, None, None, None)
    (tmp_path / f"{key}.json").write_text(json.dumps({
        "text": "", "tool_calls": [], "model": "m",
        "usage": {"completion_tokens": 0, "total_tokens": 12862}, "raw": {}, "cost_usd": None,
    }), encoding="utf-8")
    assert c.complete(msgs).text == ""
    assert c.collapsed_hits == 0


def test_a_normal_cached_response_is_still_served_without_a_call(tmp_path):
    """The repair must not turn every cache hit into a live call — that would defeat the tape."""
    from govsim.core.llm.cache import CachingReplayClient

    class _Boom:
        def complete(self, *a, **k):
            raise AssertionError("a healthy cache hit must not call the network")

    c = CachingReplayClient(_Boom(), tmp_path, mode="cache")
    msgs = [{"role": "user", "content": "govern"}]
    key = c._key(msgs, None, 0.0, None, None, None, None, None)
    (tmp_path / f"{key}.json").write_text(json.dumps({
        "text": "set_lockdown: 0.7", "tool_calls": [], "model": "m",
        "usage": {"completion_tokens": 9, "total_tokens": 500}, "raw": {}, "cost_usd": None,
    }), encoding="utf-8")
    assert c.complete(msgs).text == "set_lockdown: 0.7"
    assert c.collapsed_hits == 0


def test_a_provider_reporting_only_total_tokens_is_never_diagnosed():
    """Defaulting a MISSING completion_tokens to 0 would classify every response from such a
    provider as collapsed, and the mitigation would re-ask on literally every call."""
    assert not response_is_collapsed("resp", [], {"total_tokens": 1})
    assert not response_is_collapsed("resp", [], {"total_tokens": 1, "completion_tokens": None})
    # Present and genuinely zero is still a collapse.
    assert response_is_collapsed("<thought>...", [], {"total_tokens": 900, "completion_tokens": 0})
