"""
Tests for the regents: PID/LQR baselines (deterministic) and the LLMRegent (key-free via a fake
client + the replay tape). No network, no API key.
"""

from __future__ import annotations

from govsim.core import EveryN, Experiment, Hypothesis, Runner
from govsim.core.action import ActionSpace, VerbSpec
from govsim.core.llm import CachingReplayClient, LLMResponse
from govsim.core.system import Observation
from govsim.domains.scalar import CubicSystem, Lever, ScalarLeverInterface, StabilizationLoss
from govsim.regents import LLMRegent, LQRRegent, PIDRegent, parse_action_requests
from govsim.regents.llm_regent import default_prompt_assembler


def _space():
    return ActionSpace(
        verbs=[VerbSpec(name="set_control_input", value_range=(-2.0, 2.0))],
        context_vars=["current_x", "previous_x", "target_x", "current_u", "step"],
    )


# --- PID / LQR baselines --------------------------------------------------------------------

def test_pid_regent_emits_pd_expression():
    r = PIDRegent(verb="set_control_input", kp=0.9, kd=0.3)
    req = r.decide(Observation(vars={"current_x": 1.0, "previous_x": 0.5, "target_x": 0.0}), _space(), {})[0]
    assert req.verb == "set_control_input"
    assert "current_x - target_x" in req.payload["expr"]
    assert "current_x - previous_x" in req.payload["expr"]


def test_lqr_gain_matches_analytic_scalar():
    # A=B=Q=R=1 ⇒ P solves P^2-P-1=0 ⇒ K = B P A/(R+B^2 P) = φ/(1+φ) ≈ 0.618.
    r = LQRRegent(verb="set_control_input", A=1.0, B=1.0, Q=1.0, R=1.0)
    assert abs(r.gain - 0.6180339887) < 1e-6


def test_lqr_regent_stabilizes_linear_plant():
    iface = ScalarLeverInterface([Lever("set_control_input", (-5.0, 5.0), "current_u")])
    sys = CubicSystem({"param_A": 0.95, "param_B": 0.5, "sigma_epsilon": 0.0})
    sys.reset(0)
    sys.current_x = 1.0
    r = LQRRegent(verb="set_control_input", A=0.95, B=0.5, Q=1.0, R=0.1)
    iface.apply(r.decide(sys.observe("regent:0"), iface.action_space(sys, "regent:0"), {}), sys)
    for _ in range(50):
        sys.step()
    assert abs(sys.current_x) < 0.05  # driven toward 0


# --- LLMRegent parsing ----------------------------------------------------------------------

class _ToolClient:
    def complete(self, messages, *, model=None, temperature=0.0, seed=None, tools=None, response_format=None):
        return LLMResponse(tool_calls=[{"id": "1", "name": "set_control_input", "arguments": '{"expr": "-0.8 * current_x"}'}], model=model or "fake")


class _JSONClient:
    def __init__(self, text):
        self.text = text

    def complete(self, messages, *, model=None, temperature=0.0, seed=None, tools=None, response_format=None):
        return LLMResponse(text=self.text, model=model or "fake")


def test_parse_tool_call():
    resp = LLMResponse(tool_calls=[{"id": "1", "name": "set_control_input", "arguments": '{"expr": "-1.0*current_x"}'}])
    reqs = parse_action_requests(resp, _space(), "regent:0")
    assert len(reqs) == 1 and reqs[0].payload == {"expr": "-1.0*current_x"}


def test_parse_fenced_json_fallback():
    resp = LLMResponse(text='Here:\n```json\n{"verb": "set_control_input", "expr": "-0.5*current_x"}\n```')
    reqs = parse_action_requests(resp, _space(), "regent:0")
    assert reqs[0].payload == {"expr": "-0.5*current_x"}


def test_parse_legacy_value_expression_single_verb():
    resp = LLMResponse(text='{"policy_type_id": "set_control_input", "value_expression": "-0.7*current_x"}')
    reqs = parse_action_requests(resp, _space(), "regent:0")
    assert reqs[0].payload == {"expr": "-0.7*current_x"}


def test_llm_regent_records_io_into_scratch():
    r = LLMRegent(llm=_ToolClient(), model="fake")
    scratch: dict = {}
    reqs = r.decide(Observation(vars={"current_x": 1.0}), _space(), scratch)
    assert reqs[0].payload["expr"] == "-0.8 * current_x"
    assert len(scratch["_llm_calls"]) == 1 and scratch["_llm_calls"][0]["model"] == "fake"


def test_default_prompt_assembler_includes_trace_and_obs():
    msgs = default_prompt_assembler(Observation(vars={"current_x": 1.0}, t=5), _space(), {"trace": "rejected: bad name"})
    assert msgs[0]["role"] == "system" and "set_control_input" in msgs[0]["content"]
    assert "step 5" in msgs[1]["content"] and "rejected: bad name" in msgs[1]["content"]


# --- LLMRegent end-to-end through Runner with the replay tape (key-free) ---------------------

def test_llm_regent_end_to_end_with_replay(tmp_path):
    client = CachingReplayClient(_ToolClient(), tmp_path, mode="cache")

    def factory(seed: int) -> CubicSystem:
        s = CubicSystem({"param_A": 0.95, "param_B": 0.5, "sigma_epsilon": 0.1})
        s.reset(seed)
        return s

    exp = Experiment(
        name="cubic_llm_smoke",
        system_factory=factory,
        action_interface=ScalarLeverInterface([Lever("set_control_input", (-2.0, 2.0), "current_u")]),
        regents={"regent:0": LLMRegent(llm=client, model="fake")},
        objectives={"regent:0": StabilizationLoss(lam=0.1)},
        schedule=EveryN(20),
        seeds=[0],
        horizon=60,
        hypothesis=Hypothesis(id="H-llm-smoke", claim="llm regent runs end-to-end", baseline="scripted", primary_metric="mse"),
    )
    rec = Runner().run(exp)[0]
    assert len(rec.llm_io) >= 1  # the regent's calls were recorded into the RunRecord
    assert rec.llm_io[0]["regent_id"] == "regent:0"
