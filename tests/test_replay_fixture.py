"""
A committed replay-tape fixture: an LLM-in-the-loop experiment that runs **key-free in CI** by
serving every model call from a recorded tape under ``tests/tapes/`` (no network, no API key). This
proves the reproducibility guarantee end-to-end in the test suite, not just by hand.

The tape was recorded once against a live OpenAI-compatible endpoint (gpt-oss, tool-calling). To
regenerate it after an intentional prompt/dynamics change, run ``_record()`` with the endpoint env set
(see the function), then re-pin ``_EXPECTED_SCORE`` from a replay run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from govsim.core import EveryN, Experiment, Hypothesis, Runner
from govsim.core.llm import CachingReplayClient
from govsim.domains.scalar import CubicSystem, Lever, ScalarLeverInterface, StabilizationLoss
from govsim.regents import LLMRegent

_TAPE_DIR = Path(__file__).parent / "tapes" / "cubic_llm"
# Re-recorded 2026-08-04 against the Gemini OpenAI-compat endpoint; the AIRI endpoint the
# original tape came from now 403s. The tape's job is to prove that a RECORDED run replays
# key-free, which any tool-calling model can demonstrate.
_MODEL = "gemma-4-31b-it"
_MAX_TOKENS = 4000
_EXTRA = None
_EXPECTED_SCORE = -0.24510704520462223  # pinned from the recorded tape; see _record()


def _build(client) -> Experiment:
    def factory(seed: int) -> CubicSystem:
        s = CubicSystem({"param_A": 0.95, "param_B": 0.5, "sigma_epsilon": 0.05, "target_x": 0.0})
        s.reset(seed)
        return s

    return Experiment(
        name="replay_fixture",
        system_factory=factory,
        action_interface=ScalarLeverInterface([Lever("set_control_input", (-2.0, 2.0), "current_u")]),
        regents={"regent:0": LLMRegent(llm=client, model=_MODEL, temperature=0.0, seed=0,
                                       max_tokens=_MAX_TOKENS, extra=_EXTRA)},
        objectives={"regent:0": StabilizationLoss(lam=0.1)},
        schedule=EveryN(20),
        seeds=[0],
        horizon=60,
        hypothesis=Hypothesis(id="H-replay", claim="a committed tape reproduces an LLM run",
                              baseline="self", primary_metric="mse"),
    )


class _NoNetwork:
    def complete(self, *a, **k):  # pragma: no cover - asserts replay never hits the network
        raise AssertionError("replay mode must not call the network")


@pytest.mark.skipif(not _TAPE_DIR.exists(), reason="replay tape fixture not recorded")
def test_llm_regent_replays_from_committed_tape():
    client = CachingReplayClient(_NoNetwork(), _TAPE_DIR, mode="replay")
    rec = Runner().run(_build(client))[0]
    assert len(rec.llm_io) >= 1
    assert all(call["cached"] for call in rec.llm_io)               # every call served from the tape
    assert abs(rec.score["regent:0"] - _EXPECTED_SCORE) < 1e-9      # byte-for-byte reproducible, key-free


def _record() -> None:  # pragma: no cover - maintenance helper (needs a live endpoint + key)
    """Record the tape live, then print the score to pin as ``_EXPECTED_SCORE``. Run with e.g.:
        OPENAI_BASE_URL=… OPENAI_API_KEY_ENV=AIRI_KEY  (key in env)
        python -c "from tests.test_replay_fixture import _record; _record()"
    """
    import os

    from dotenv import load_dotenv

    from govsim.core.llm import OpenAICompatClient

    load_dotenv()
    inner = OpenAICompatClient(
        base_url=os.environ["OPENAI_BASE_URL"],
        api_key_env=os.environ.get("OPENAI_API_KEY_ENV", "OPENAI_API_KEY"),
        drop_params=frozenset(
            x.strip() for x in os.environ.get("GOVSIM_LLM_DROP_PARAMS", "").split(",") if x.strip()))
    client = CachingReplayClient(inner, _TAPE_DIR, mode="cache")
    rec = Runner().run(_build(client))[0]
    print("RECORDED score:", repr(rec.score["regent:0"]))
