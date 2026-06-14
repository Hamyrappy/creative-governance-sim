"""
Tests for the migrated ``CoupledSystem`` (multi-state linear plant with control inertia,
cross-coupling, parameter drift, periodic regime shocks). All key-free / deterministic.

Covers the contracts the rollout-based harnesses depend on: per-system Generator determinism,
clone faithfulness (a clone continues the SAME stochastic stream), the install-as-data eval
cadence (the lever re-evaluates each step), the control-inertia semantics, and that the regime
shock actually perturbs the trajectory.
"""

from __future__ import annotations

from govsim.core import EveryN, Experiment, Hypothesis, Runner
from govsim.domains.scalar import CoupledSystem, Lever, ScalarLeverInterface, StabilizationLoss
from govsim.core.regent import ScriptedRegent


def _make(**extra) -> CoupledSystem:
    params = {"param_A": 0.95, "param_B": 0.4, "sigma_epsilon": 0.1, "u_range": (-2.0, 2.0)}
    params.update(extra)
    s = CoupledSystem(params)
    s.reset(0)
    return s


def test_determinism_same_seed_same_trajectory():
    a, b = _make(), _make()
    xa = [(_step(a)) for _ in range(50)]
    xb = [(_step(b)) for _ in range(50)]
    assert xa == xb


def _step(sys: CoupledSystem) -> float:
    sys.step()
    return sys.current_x


def test_different_seed_diverges():
    a = _make()
    b = CoupledSystem({"param_A": 0.95, "param_B": 0.4, "sigma_epsilon": 0.1})
    b.reset(1)
    xa = [_step(a) for _ in range(30)]
    xb = [_step(b) for _ in range(30)]
    assert xa != xb


def test_clone_is_faithful_continuation():
    sys = _make()
    for _ in range(10):
        sys.step()
    clone = sys.clone()
    orig_tail = [_step(sys) for _ in range(20)]
    clone_tail = [_step(clone) for _ in range(20)]
    assert orig_tail == clone_tail  # the Generator state was carried, not re-seeded


def test_lever_installs_and_reevaluates_each_step():
    sys = _make(u_smoothing_rho=0.0)  # no inertia ⇒ u_eff == u_commanded ⇒ easy to read off
    iface = ScalarLeverInterface([Lever("set_control_input", (-2.0, 2.0), "u_commanded")])
    space = iface.action_space(sys, "regent:0")
    # install an expression that drives u toward -current_x
    report = iface.apply(
        [ScriptedRegent("set_control_input", "-1.0 * current_x").decide(sys.observe(), space, {})[0]], sys
    )
    assert report.applied and not report.rejected
    x_before = sys.current_x
    sys.step()
    # with rho_u=0, current_u == clipped(-1.0 * x_before)
    assert abs(sys.current_u - max(-2.0, min(2.0, -1.0 * x_before))) < 1e-9


def test_control_inertia_lags_commanded():
    sys = _make(u_smoothing_rho=0.8, sigma_epsilon=0.0, param_B_drift_sigma=0.0, param_C_drift_sigma=0.0)
    iface = ScalarLeverInterface([Lever("set_control_input", (-2.0, 2.0), "u_commanded")])
    space = iface.action_space(sys, "regent:0")
    iface.apply([ScriptedRegent("set_control_input", "2.0").decide(sys.observe(), space, {})[0]], sys)
    sys.step()
    # one step of u_eff = 0.8*0 + 0.2*2.0 = 0.4 (inertia ⇒ not the full 2.0 yet)
    assert abs(sys.current_u - 0.4) < 1e-9
    sys.step()
    assert abs(sys.current_u - (0.8 * 0.4 + 0.2 * 2.0)) < 1e-9  # 0.72, still climbing


def test_regime_shock_perturbs_trajectory():
    no_shock = _make(shock_period=0, sigma_epsilon=0.0, sigma_aux1=0.0, sigma_aux2=0.0,
                     param_B_drift_sigma=0.0, param_C_drift_sigma=0.0)
    shocked = _make(shock_period=10, shock_magnitude_aux1=1.0, shock_magnitude_aux2=-1.0,
                    sigma_epsilon=0.0, sigma_aux1=0.0, sigma_aux2=0.0,
                    param_B_drift_sigma=0.0, param_C_drift_sigma=0.0)
    a = [_step(no_shock) for _ in range(20)]
    b = [_step(shocked) for _ in range(20)]
    assert a[:9] == b[:9]          # identical until the first shock fires (step index 10)
    assert a[15] != b[15]          # the shock propagates into x_main via the coupling


def test_runs_end_to_end_through_runner():
    def factory(seed: int) -> CoupledSystem:
        s = CoupledSystem({"param_A": 0.95, "param_B": 0.4, "sigma_epsilon": 0.1, "shock_period": 60})
        s.reset(seed)
        return s

    exp = Experiment(
        name="coupled_smoke",
        system_factory=factory,
        action_interface=ScalarLeverInterface([Lever("set_control_input", (-2.0, 2.0), "u_commanded")]),
        regents={"regent:0": ScriptedRegent(verb="set_control_input", expr="-1.2 * current_x")},
        objectives={"regent:0": StabilizationLoss(lam=0.1)},
        schedule=EveryN(20),
        seeds=[0, 1],
        horizon=120,
        hypothesis=Hypothesis(id="H-coupled-smoke", claim="coupled runs", baseline="static", primary_metric="mse"),
    )
    recs = Runner().run(exp)
    assert len(recs) == 2
    assert all("current_x" in recs[0].metrics_series[0] for _ in [0])
    assert recs[0].components["regent:0"]["mse"] >= 0.0
