"""
Tests for the scalar domain: the sandbox bridge, the lever interface (reject-with-feedback),
the cubic/SIR systems' determinism + per-system RNG, clone/rollout faithfulness, the
re-eval-cadence contract, and the objectives.
"""

from __future__ import annotations

import numpy as np

from govsim.core.action import ActionRequest
from govsim.core.sandbox import compile_expr, eval_safe
from govsim.domains.scalar import (
    CompanyProfit,
    CompanySystem,
    CubicSystem,
    EpidemicLoss,
    Lever,
    ScalarLeverInterface,
    SIRSystem,
    StabilizationLoss,
)


# --- sandbox bridge -------------------------------------------------------------------------

def test_compile_expr_accepts_whitelisted_names():
    res = compile_expr("-0.9 * current_x", ["current_x"])
    assert res.ok and res.compiled is not None
    assert eval_safe(res.compiled, {"current_x": 2.0}) == -1.8


def test_compile_expr_rejects_unknown_name_with_feedback():
    res = compile_expr("-0.9 * forbidden_var", ["current_x"])
    assert not res.ok
    assert "forbidden_var" in res.feedback


def test_compile_expr_rejects_statements():
    res = compile_expr("import os", ["current_x"])
    assert not res.ok and res.feedback


# --- ScalarLeverInterface -------------------------------------------------------------------

def _cubic_iface():
    return ScalarLeverInterface([Lever("set_control_input", (-2.0, 2.0), "current_u")])


def test_action_space_lists_verbs_and_context():
    sys = CubicSystem({"target_x": 0.0})
    space = _cubic_iface().action_space(sys, "regent:0")
    assert space.verb_names() == ["set_control_input"]
    assert "current_x" in space.context_vars and "target_x" in space.context_vars


def test_apply_rejects_bad_verb_and_bad_expr():
    sys = CubicSystem()
    iface = _cubic_iface()
    report = iface.apply(
        [
            ActionRequest("regent:0", "nonexistent", {"expr": "1.0"}),
            ActionRequest("regent:0", "set_control_input", {"expr": "current_x +"}),
        ],
        sys,
    )
    assert report.applied == []
    assert len(report.rejected) == 2
    assert "Unknown verb" in report.rejected[0][1]


def test_apply_installs_and_clips_lever_each_step():
    sys = CubicSystem({"param_A": 1.0, "param_B": 1.0, "sigma_epsilon": 0.0, "u_range": (-2.0, 2.0)})
    iface = _cubic_iface()
    # u = -100*current_x would blow past the range; it must be clipped to -2.0 each step.
    report = iface.apply([ActionRequest("regent:0", "set_control_input", {"expr": "-100.0 * current_x"})], sys)
    assert len(report.applied) == 1
    sys.current_x = 5.0
    sys.step()  # re-eval: u = clip(-500, [-2,2]) = -2.0
    assert sys.current_u == -2.0


# --- determinism + per-system RNG + clone faithfulness --------------------------------------

def test_cubic_determinism_same_seed_same_trajectory():
    a = CubicSystem({"sigma_epsilon": 0.1})
    b = CubicSystem({"sigma_epsilon": 0.1})
    a.reset(123)
    b.reset(123)
    xa = [(_a := a.step(), a.current_x)[1] for _ in range(20)]
    xb = [(_b := b.step(), b.current_x)[1] for _ in range(20)]
    assert xa == xb


def test_cubic_different_seed_different_trajectory():
    a = CubicSystem({"sigma_epsilon": 0.1})
    a.reset(1)
    b = CubicSystem({"sigma_epsilon": 0.1})
    b.reset(2)
    xa = [(a.step(), a.current_x)[1] for _ in range(20)]
    xb = [(b.step(), b.current_x)[1] for _ in range(20)]
    assert xa != xb


def test_clone_continues_same_stochastic_stream():
    sys = CubicSystem({"sigma_epsilon": 0.2})
    sys.reset(7)
    for _ in range(5):
        sys.step()
    twin = sys.clone()
    orig_tail = [(sys.step(), sys.current_x)[1] for _ in range(10)]
    twin_tail = [(twin.step(), twin.current_x)[1] for _ in range(10)]
    assert orig_tail == twin_tail  # the clone carries the Generator → faithful continuation


def test_clone_with_installed_lever_reevaluates_against_itself():
    # A clone must re-eval the installed lever against ITS OWN state, not the original's.
    sys = CubicSystem({"param_A": 1.0, "param_B": 1.0, "sigma_epsilon": 0.0})
    _cubic_iface().apply([ActionRequest("regent:0", "set_control_input", {"expr": "current_x"})], sys)
    sys.current_x = 1.0
    twin = sys.clone()
    twin.current_x = 9.0
    twin.step()  # u should be re-eval'd from twin.current_x (9.0), clipped to 2.0
    assert twin.current_u == 2.0
    sys.step()   # original unaffected: u from sys.current_x (1.0) = 1.0
    assert sys.current_u == 1.0


# --- objectives -----------------------------------------------------------------------------

def test_stabilization_loss_rewards_staying_on_target():
    on_target = [{"current_x": 0.0, "current_u": 0.0, "target_x": 0.0} for _ in range(10)]
    off_target = [{"current_x": 1.0, "current_u": 0.0, "target_x": 0.0} for _ in range(10)]
    obj = StabilizationLoss(lam=0.1)
    assert obj.evaluate(on_target) > obj.evaluate(off_target)
    comps = obj.components(off_target)
    assert comps["mse"] == 1.0 and comps["msu"] == 0.0


def test_epidemic_and_company_objectives():
    traj = [{"infected": 0.1, "cum_cost": float(i)} for i in range(5)]
    assert EpidemicLoss(lam=1.0).components(traj)["cum_cost"] == 4.0
    ctraj = [{"profit": 10.0, "cash": 10.0 * (i + 1)} for i in range(5)]
    assert CompanyProfit().evaluate(ctraj) == 10.0


# --- SIR runs on the SAME interface (generality smoke) --------------------------------------

def test_sir_runs_on_scalar_interface():
    sys = SIRSystem({"noise_sigma": 0.0})
    sys.reset(0)
    iface = ScalarLeverInterface([Lever("set_lockdown", (0.0, 0.9), "lockdown"), Lever("set_vaccination", (0.0, 0.5), "vacc")])
    report = iface.apply([ActionRequest("regent:0", "set_lockdown", {"expr": "0.7 if I > 0.05 else 0.1"})], sys)
    assert len(report.applied) == 1
    for _ in range(10):
        sys.step()
    assert 0.0 <= sys.lockdown <= 0.9
    assert sys.metrics()["infected"] >= 0.0


def test_company_runs_on_scalar_interface():
    sys = CompanySystem({"demand_sigma": 0.0})
    sys.reset(0)
    iface = ScalarLeverInterface([Lever("set_price", (0.0, 100.0), "price"), Lever("set_production", (0.0, 200.0), "production")])
    iface.apply(
        [
            ActionRequest("regent:0", "set_price", {"expr": "12.0"}),
            ActionRequest("regent:0", "set_production", {"expr": "max(0, last_demand)"}),
        ],
        sys,
    )
    for _ in range(5):
        sys.step()
    assert sys.price == 12.0
