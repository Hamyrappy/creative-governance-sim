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


def test_opro_realized_mode_credits_deployed_law():
    sys, iface = _cubic(), _iface()
    r = OPRORegent("set_control_input", _OPROFakeClient(), "fake", scoring="realized")
    space = iface.action_space(sys, "regent:0")
    scratch: dict = {}

    # decision 1: nothing deployed yet ⇒ archive empty, propose (seed 0 ⇒ gain 0.3), deploy it
    out1 = r.decide(sys.observe(), space, scratch)
    assert out1[0].payload["expr"] == "-0.3 * current_x"
    assert scratch["_opro_pending"] == "-0.3 * current_x"
    assert scratch["_opro_archive"] == []

    # the Runner supplies the realized score of the deployed law before the next decision
    scratch["_last_realized_score"] = -0.5
    out2 = r.decide(sys.observe(), space, scratch)
    assert scratch["_opro_archive"] == [("-0.3 * current_x", -0.5)]  # the deployed law was credited
    assert out2[0].payload["expr"] == "-1.9 * current_x"            # explores a new proposal (seed 1)
    assert scratch["_opro_pending"] == "-1.9 * current_x"


def test_opro_realized_mode_runs_through_runner():
    def factory(seed: int) -> CubicSystem:
        s = CubicSystem({"param_A": 0.95, "param_B": 0.5, "sigma_epsilon": 0.05})
        s.reset(seed)
        return s

    exp = Experiment(
        name="opro_realized_smoke",
        system_factory=factory,
        action_interface=ScalarLeverInterface([Lever("set_control_input", (-3.0, 3.0), "current_u")]),
        regents={"regent:0": OPRORegent("set_control_input", _OPROFakeClient(), "fake", scoring="realized")},
        objectives={"regent:0": StabilizationLoss(lam=0.05)},
        schedule=EveryN(15),
        seeds=[0],
        horizon=120,
        hypothesis=Hypothesis(id="H-opro-realized", claim="realized opro runs", baseline="static", primary_metric="mse"),
    )
    rec = Runner().run(exp)[0]
    # the realized-feedback channel let OPRO credit and explore across several decisions
    assert sum(1 for c in rec.llm_io if c.get("regent") == "opro") >= 2


def test_cubic_nonlinear_lqr_experiment_is_key_free_and_runs():
    from govsim.experiments import get

    exp = get("cubic_nonlinear_lqr")
    exp.seeds = [0]
    rec = Runner().run(exp)[0]
    assert rec.system_id == "CubicSystem" and rec.regent_specs["regent:0"]["type"] == "LQRRegent"


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


def test_incumbent_moves_when_a_later_law_scores_better():
    """OPRO must be able to change its mind. It could not, and the cause was elsewhere.

    An audit found the incumbent frozen at the first proposal across ascending, descending and
    shuffled proposal orders. The cause was not in this file: ``EpidemicLoss.evaluate`` was reading
    an undifferenced running total, so realized scores fell monotonically with time and the earliest
    law always held the archive's top slot. With a real score the exploit branch tracks the best law
    as intended — but the failure was invisible from inside OPRO, so it is pinned here.
    """
    import json

    from govsim.core.action import ActionSpace, VerbSpec
    from govsim.core.llm.client import LLMResponse
    from govsim.core.system import Observation

    class _Stub:
        def __init__(self, exprs):
            self.exprs, self.i = list(exprs), 0

        def complete(self, messages, **kw):
            expr = self.exprs[min(self.i, len(self.exprs) - 1)]
            self.i += 1
            return LLMResponse(text="", tool_calls=[
                {"id": "1", "name": "set_lockdown", "arguments": json.dumps({"expr": expr})}])

    space = ActionSpace(verbs=[VerbSpec(name="set_lockdown", value_range=(0.0, 0.9))],
                        context_vars=["I"])
    regent = OPRORegent("set_lockdown", _Stub([f"0.{i}" for i in range(1, 9)]), "stub",
                        scoring="realized")
    scratch: dict = {}
    deployed = []
    for k in range(8):
        reqs = regent.decide(Observation(vars={"I": 0.1}, scope="regent:0", t=k * 10), space, scratch)
        deployed.append(reqs[0].payload["expr"] if reqs else None)
        scratch["_last_realized_score"] = -10.0 + k  # each successive law is genuinely better

    assert len(set(deployed)) > 1, "the incumbent never changed despite strictly improving scores"
    archive = scratch["_opro_archive"]
    assert archive[-1][1] == max(s for _, s in archive), "archive's last entry must be its best"
