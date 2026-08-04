"""
Regression tests for the review-fix batch (see the audit summary). Each test pins the corrected
behaviour of one confirmed finding so it cannot silently regress. All key-free.
"""

from __future__ import annotations

import json

import pytest

from govsim.analysis import collapse_summary, compare, metric_by_seed
from govsim.core import EveryN, Experiment, Harness, Hypothesis, Runner, RolloutContext
from govsim.core.action import ActionRequest
from govsim.core.harness import HarnessComponent, Outcome
from govsim.core.llm import LLMResponse
from govsim.core.regent import ScriptedRegent, StaticRegent
from govsim.core.sandbox import compile_expr, eval_safe
from govsim.core.system import Observation
from govsim.domains.scalar import (
    CubicSystem, Lever, ScalarLeverInterface, SIRSystem, StabilizationLoss,
)
from govsim.harness import EpisodicMemory
from govsim.regents import OPRORegent


# --- #8 SIR population conservation ---------------------------------------------------------

def test_sir_conserves_population_each_step():
    s = SIRSystem({"beta0": 0.35, "gamma": 0.10, "noise_sigma": 0.0})
    s.reset(0)
    s.lockdown, s.vacc = 0.0, 0.3
    total0 = s.S + s.I + s.R
    for _ in range(30):  # S stays > 0 here, so the max(0,S) clamp never fires
        s.step()
        assert abs((s.S + s.I + s.R) - total0) < 1e-12  # S+I+R invariant every step


# --- #4 / #26 compare() significance floor --------------------------------------------------

def test_compare_single_seed_is_never_significant():
    res = compare({0: 1.0}, {0: 2.0}, lower_is_better=True)  # a zero-width degenerate CI
    assert res["n"] == 1
    assert not res["excludes_zero"] and not res["a_better_than_b"]
    assert res["underpowered"]


def test_compare_no_shared_seeds_flags_empty_pairing():
    res = compare({0: 1.0}, {5: 2.0}, lower_is_better=True)
    assert res["n"] == 0 and res["no_shared_seeds"] and not res["a_better_than_b"]


# --- #6 / #11 / #30 metric_by_seed fails loud on an unknown metric --------------------------

def _scripted_records():
    def factory(seed: int) -> CubicSystem:
        s = CubicSystem({"param_A": 0.95, "param_B": 0.5, "sigma_epsilon": 0.05})
        s.reset(seed)
        return s

    exp = Experiment(
        name="mbys_probe",
        system_factory=factory,
        action_interface=ScalarLeverInterface([Lever("set_control_input", (-2.0, 2.0), "current_u")]),
        regents={"regent:0": ScriptedRegent(verb="set_control_input", expr="-0.9 * current_x")},
        objectives={"regent:0": StabilizationLoss(lam=0.1)},
        schedule=EveryN(10),
        seeds=[0, 1],
        horizon=40,
        hypothesis=Hypothesis(id="H", claim="c", baseline="b", primary_metric="mse"),
    )
    return Runner().run(exp)


def test_metric_by_seed_raises_on_unknown_metric():
    recs = _scripted_records()
    assert set(metric_by_seed(recs, "mse")) == {0, 1}  # known metric still works
    with pytest.raises(KeyError):
        metric_by_seed(recs, "totally_not_a_metric")


# --- #27 collapse_summary keeps diverged (non-finite) runs as the worst case -----------------

class _FakeRec:
    def __init__(self, seed, score, terminated=None):
        self.seed = seed
        self.score = {"regent:0": score}
        self.terminated_at_step = terminated


def test_collapse_summary_treats_nonfinite_score_as_worst():
    recs = [_FakeRec(0, -0.5), _FakeRec(1, float("nan"), terminated=10)]
    cs = collapse_summary(recs)
    assert cs["n_nonfinite_score"] == 1
    assert cs["worst_score"] == float("-inf")  # a NaN run is not silently dropped from the worst-case


# --- #1 / #7 / #9 / #21 post-shock windowed metric ------------------------------------------

def test_stabilization_loss_post_shock_window_isolates_post_shock():
    # pre-shock (steps 0..3) off-target, post-shock (steps 4..7) on-target
    rows = [{"step": float(i), "current_x": (2.0 if i < 4 else 0.0), "current_u": 0.0, "target_x": 0.0}
            for i in range(8)]
    obj = StabilizationLoss(lam=0.0, post_shock_step=4)
    comps = obj.components(rows)
    assert comps["mse"] == 2.0        # whole-horizon: (4·4 + 4·0)/8
    assert comps["post_mse"] == 0.0   # post-shock only: all on target


def test_post_mse_is_worst_when_run_ends_before_shock():
    # a run that terminated at step 3, shock at 100 ⇒ no post-shock rows ⇒ post_mse must NOT be a
    # flattering 0.0 but worst-case inf (the collapse detector tracks it separately)
    rows = [{"step": float(i), "current_x": 5.0, "current_u": 0.0, "target_x": 0.0} for i in range(4)]
    comps = StabilizationLoss(lam=0.0, post_shock_step=100).components(rows)
    assert comps["post_mse"] == float("inf")


def test_h1_arms_are_scored_on_post_shock_metric():
    from govsim.experiments import get

    for name in ("cubic_nonlinear", "cubic_nonlinear_lqr", "cubic_nonlinear_opro",
                 "cubic_nonlinear_llm", "cubic_nonlinear_llm_obfuscated", "coupled_regime_shift"):
        exp = get(name)
        assert exp.hypothesis.primary_metric == "post_mse", name
    exp = get("cubic_nonlinear_lqr")
    exp.seeds = [0]
    rec = Runner().run(exp)[0]
    assert "post_mse" in rec.components["regent:0"]


# --- #43 StabilizationLoss fails loud on a mis-wired state key --------------------------------

def test_stabilization_loss_raises_on_missing_state_key():
    rows = [{"wrong_key": 1.0, "current_u": 0.0}]
    with pytest.raises(KeyError):
        StabilizationLoss().components(rows)


# --- #2 / #17 / #18 sandbox hardening --------------------------------------------------------

def test_sandbox_rejects_unicode_identifier_bypass():
    # fullwidth 'random.random()' — NFKC-normalizes to the real module at compile time
    assert not compile_expr("ｒａｎｄｏｍ.ｒａｎｄｏｍ()", ["current_x"]).ok


def test_sandbox_rejects_exponentiation_bomb():
    assert not compile_expr("9 ** 9 ** 9", ["current_x"]).ok       # chained power
    assert not compile_expr("current_x ** 64", ["current_x"]).ok   # huge literal exponent


def test_sandbox_still_allows_ordinary_powers():
    r = compile_expr("current_x ** 3", ["current_x"])
    assert r.ok and eval_safe(r.compiled, {"current_x": 2.0}) == 8.0


def test_sandbox_removes_unbounded_allocators():
    assert not compile_expr("np.arange(1000000000)", ["current_x"]).ok
    assert not compile_expr("np.linspace(0, 1, 1000000000)", ["current_x"]).ok


# --- #3 / #14 / #24 OPRO realized mode exploits its incumbent best ---------------------------

class _OPROFake:
    def complete(self, messages, *, model=None, temperature=0.0, seed=None, tools=None,
                 response_format=None, max_tokens=None, extra=None):
        return LLMResponse(
            tool_calls=[{"id": "1", "name": "set_control_input",
                         "arguments": json.dumps({"expr": "-1.9 * current_x"})}],
            model=model or "fake",
        )


def test_opro_realized_mode_exploits_incumbent_best():
    iface = ScalarLeverInterface([Lever("set_control_input", (-3.0, 3.0), "current_u")])
    sys = CubicSystem({"param_A": 0.95, "param_B": 0.5, "sigma_epsilon": 0.0})
    sys.reset(0)
    r = OPRORegent("set_control_input", _OPROFake(), "fake", scoring="realized",
                   explore_warmup=0, explore_period=100)
    space = iface.action_space(sys, "regent:0")
    # archive already holds a distinct best law; this is an EXPLOIT decision (idx=1, no warmup/period hit)
    scratch = {"_opro_archive": [("-0.55 * current_x", -0.01)], "_opro_realized_idx": 1}
    out = r.decide(sys.observe(), space, scratch)
    # it re-deploys the archived best, NOT the fresh temp-0.8 proposal ("-1.9 * current_x")
    assert out[0].payload["expr"] == "-0.55 * current_x"
    assert scratch["_opro_pending"] == "-0.55 * current_x"


# --- #13 EpisodicMemory retrieves by state similarity, not recency ---------------------------

def test_episodic_memory_ignores_step_in_similarity():
    m = EpisodicMemory(k=1)
    m.episodes = [
        {"state": {"current_x": 0.05, "step": 0.0}, "actions": [{"verb": "v", "expr": "law_close"}], "score": 1.0},
        {"state": {"current_x": 9.0, "step": 100.0}, "actions": [{"verb": "v", "expr": "law_far"}], "score": 1.0},
    ]
    scratch: dict = {}
    m.on_observe(Observation(vars={"current_x": 0.0, "step": 101.0}), None, scratch)
    # nearest by STATE is the x≈0.05 episode; if 'step' leaked into the metric it would pick the recent one
    assert "law_close" in scratch["memory"] and "law_far" not in scratch["memory"]


# --- #15 an empty regent decision is surfaced as an error (feeds TraceFeedback) --------------

class _ErrCapture(HarnessComponent):
    name = "err_capture"

    def __init__(self):
        self.errors: list = []

    def on_outcome(self, view, requests, outcome: Outcome, scratch: dict) -> None:
        self.errors.append(outcome.error)


def test_runner_surfaces_empty_action_as_error():
    cap = _ErrCapture()

    def factory(seed: int) -> CubicSystem:
        s = CubicSystem({"sigma_epsilon": 0.0})
        s.reset(seed)
        return s

    exp = Experiment(
        name="empty_action_probe",
        system_factory=factory,
        action_interface=ScalarLeverInterface([Lever("set_control_input", (-2.0, 2.0), "current_u")]),
        regents={"regent:0": StaticRegent()},  # always returns [] — the LLM "no parseable reply" analogue
        objectives={"regent:0": StabilizationLoss(lam=0.1)},
        schedule=EveryN(10),
        seeds=[0],
        horizon=30,
        hypothesis=Hypothesis(id="H", claim="c", baseline="b", primary_metric="mse"),
        harness=Harness([cap]),
    )
    Runner().run(exp)
    assert any(e and "no action produced" in e for e in cap.errors)


# --- #10 RunRecord captures harness composition (H3 ablation provenance) ---------------------

def test_run_record_captures_harness_components():
    from govsim.harness import TraceFeedback

    def factory(seed: int) -> CubicSystem:
        s = CubicSystem({"sigma_epsilon": 0.0})
        s.reset(seed)
        return s

    exp = Experiment(
        name="harness_prov_probe",
        system_factory=factory,
        action_interface=ScalarLeverInterface([Lever("set_control_input", (-2.0, 2.0), "current_u")]),
        regents={"regent:0": ScriptedRegent(verb="set_control_input", expr="-0.9 * current_x")},
        objectives={"regent:0": StabilizationLoss(lam=0.1)},
        schedule=EveryN(10),
        seeds=[0],
        horizon=30,
        hypothesis=Hypothesis(id="H", claim="c", baseline="b", primary_metric="mse"),
        harness=Harness([TraceFeedback(), EpisodicMemory(k=2)]),
    )
    rec = Runner().run(exp)[0]
    names = [c["name"] for c in rec.harness_components]
    assert "trace_feedback" in names and "episodic_memory" in names
    assert all(c["enabled"] for c in rec.harness_components)
