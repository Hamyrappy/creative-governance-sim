"""
Tests for the OPRORegent baseline (trace-less optimization-by-prompting). Key-free: a fake client
returns a control law whose gain is a deterministic function of the request seed, cycling through a
fixed list. One gain (1.9) cancels the cubic pole (x_{k+1}=0.95x+0.5u ⇒ 0.95-0.5·1.9=0), so OPRO's
rollout scoring must make it the incumbent best the regent commits.
"""

from __future__ import annotations

import json

from govsim.core import EveryN, Experiment, Hypothesis, Runner, RolloutContext
from govsim.core.llm import LLMResponse
from govsim.domains.scalar import CubicSystem, Lever, ScalarLeverInterface, StabilizationLoss
from govsim.regents import OPRORegent


class _OPROFakeClient:
    GAINS = [0.3, 1.9, 0.8, 1.2]

    def complete(self, messages, *, model=None, temperature=0.0, seed=None, tools=None,
                 response_format=None, max_tokens=None, extra=None):
        k = self.GAINS[(seed or 0) % len(self.GAINS)]
        return LLMResponse(
            tool_calls=[{"id": "1", "name": "set_control_input",
                         "arguments": json.dumps({"expr": f"-{k} * current_x"})}],
            model=model or "fake",
        )


def _cubic():
    s = CubicSystem({"param_A": 0.95, "param_B": 0.5, "sigma_epsilon": 0.0})
    s.reset(0)
    s.current_x = 1.0
    return s


def _iface():
    return ScalarLeverInterface([Lever("set_control_input", (-3.0, 3.0), "current_u")])


def test_opro_archive_converges_to_best_gain():
    sys, iface = _cubic(), _iface()
    ctx = RolloutContext(sys, iface, StabilizationLoss(lam=0.0))
    r = OPRORegent("set_control_input", _OPROFakeClient(), "fake", probe_horizon=40, probe_seeds=(0,))
    scratch = {"_rollout": ctx}
    space = iface.action_space(sys, "regent:0")

    emitted = None
    for _ in range(6):
        emitted = r.decide(sys.observe(), space, scratch)[0].payload["expr"]

    assert emitted == "-1.9 * current_x"  # the pole-cancelling gain is committed as the incumbent best
    archive = scratch["_opro_archive"]
    assert len(archive) == 6  # one (law, score) appended per decision
    assert archive[-1][0] == "-1.9 * current_x"  # archive sorted ascending ⇒ best last


def test_opro_records_llm_io_and_is_trace_less():
    sys, iface = _cubic(), _iface()
    ctx = RolloutContext(sys, iface, StabilizationLoss(lam=0.0))
    r = OPRORegent("set_control_input", _OPROFakeClient(), "fake", probe_seeds=(0,))
    scratch = {"_rollout": ctx}
    r.decide(sys.observe(), iface.action_space(sys, "regent:0"), scratch)
    assert scratch["_llm_calls"][0]["regent"] == "opro"
    # trace-less: the regent never reads scratch["trace"] (no error channel) — only its own archive
    assert "_opro_archive" in scratch


def test_opro_degrades_without_rollout_context():
    """No scratch['_rollout'] (non-rollable system) ⇒ emit the LLM's raw proposal, empty archive."""
    sys, iface = _cubic(), _iface()
    r = OPRORegent("set_control_input", _OPROFakeClient(), "fake")
    scratch: dict = {}
    reqs = r.decide(sys.observe(), iface.action_space(sys, "regent:0"), scratch)
    assert reqs and reqs[0].payload["expr"] == "-0.3 * current_x"  # seed 0 ⇒ first gain
    assert scratch["_opro_archive"] == []


def test_opro_end_to_end_through_runner_stabilizes():
    def factory(seed: int) -> CubicSystem:
        s = CubicSystem({"param_A": 0.95, "param_B": 0.5, "sigma_epsilon": 0.05})
        s.reset(seed)
        s.current_x = 1.0
        return s

    exp = Experiment(
        name="opro_smoke",
        system_factory=factory,
        action_interface=ScalarLeverInterface([Lever("set_control_input", (-3.0, 3.0), "current_u")]),
        regents={"regent:0": OPRORegent("set_control_input", _OPROFakeClient(), "fake",
                                        probe_horizon=30, probe_seeds=(0, 1))},
        objectives={"regent:0": StabilizationLoss(lam=0.05)},
        schedule=EveryN(15),
        seeds=[0],
        horizon=150,
        hypothesis=Hypothesis(id="H-opro", claim="opro runs", baseline="static", primary_metric="mse"),
    )
    rec = Runner().run(exp)[0]
    assert abs(rec.metrics_series[-1]["current_x"]) < 0.3   # OPRO found a stabilizing law
    assert any(call["regent"] == "opro" for call in rec.llm_io)
