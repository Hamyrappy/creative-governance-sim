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

from types import SimpleNamespace

from govsim.core.llm.client import OpenAICompatClient


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
