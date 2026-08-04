"""
Tests for the rollout primitive (``core/rollout.py``), the ``RolloutContext`` the Runner injects,
and the ``RolloutProbe`` harness component (rollout-dependent, gated on ``RollableSystem``).

All key-free: a deterministic "gain-sweep" fake regent emits a different proportional gain per
``scratch["_probe_seed_offset"]``, and the probe must select the gain that best stabilizes the plant.
"""

from __future__ import annotations

from govsim.core import (
    EveryN, Experiment, Harness, Hypothesis, Runner, RolloutContext, rollout,
)
from govsim.core.action import ActionRequest
from govsim.core.harness import HarnessComponent
from govsim.core.regent import Regent
from govsim.domains.scalar import CubicSystem, Lever, ScalarLeverInterface, StabilizationLoss
from govsim.harness import RolloutProbe


def _cubic(seed=0, **p):
    sys = CubicSystem({"param_A": 0.95, "param_B": 0.5, "sigma_epsilon": 0.0, **p})
    sys.reset(seed)
    sys.current_x = 1.0
    return sys


def _iface():
    return ScalarLeverInterface([Lever("set_control_input", (-3.0, 3.0), "current_u")])


# --- rollout() primitive --------------------------------------------------------------------

def test_rollout_does_not_mutate_the_live_system():
    sys = _cubic()
    iface, obj = _iface(), StabilizationLoss(lam=0.1)
    req = ActionRequest("regent:0", "set_control_input", {"expr": "-1.9 * current_x"})
    before = sys.current_x
    score, traj = rollout(sys, iface, obj, [req], horizon=30)
    assert sys.current_x == before  # only the clone advanced
    assert len(traj) == 30 and score <= 0.0  # post-step rows only (matches the Runner's slice)


def test_rollout_seed_resamples_independent_future_from_same_state():
    sys = _cubic(sigma_epsilon=0.2)
    iface, obj = _iface(), StabilizationLoss(lam=0.1)
    req = ActionRequest("regent:0", "set_control_input", {"expr": "-1.0 * current_x"})
    s0a = rollout(sys, iface, obj, [req], horizon=20, seed=7)[0]
    s0b = rollout(sys, iface, obj, [req], horizon=20, seed=7)[0]
    s1 = rollout(sys, iface, obj, [req], horizon=20, seed=8)[0]
    assert s0a == s0b   # same probe seed ⇒ identical future (paired)
    assert s0a != s1    # different probe seed ⇒ a different future


def test_rollout_context_scores_over_seeds():
    ctx = RolloutContext(_cubic(sigma_epsilon=0.1), _iface(), StabilizationLoss(lam=0.1))
    req = ActionRequest("regent:0", "set_control_input", {"expr": "-1.9 * current_x"})
    scores = ctx.score([req], horizon=20, seeds=[0, 1, 2])
    assert len(scores) == 3 and all(s <= 0.0 for s in scores)


# --- RolloutProbe ---------------------------------------------------------------------------

class _GainSweepRegent(Regent):
    """Emits u = -(0.4 + 0.5*offset)*current_x: gains 0.4/0.9/1.4/1.9 for offsets 0..3.
    The best stabilizer of x_{k+1}=0.95x+0.5u is k≈1.9 (pole 0.95-0.5*1.9=0)."""

    def decide(self, view, space, scratch):
        k = 0.4 + 0.5 * int(scratch.get("_probe_seed_offset", 0))
        return [ActionRequest(self.id, "set_control_input", {"expr": f"-{k:.4f} * current_x"})]


def test_rollout_probe_selects_the_best_gain():
    probe = RolloutProbe(n_candidates=4, horizon=40, seeds=(0,), lam=0.0)
    ctx = RolloutContext(_cubic(), _iface(), StabilizationLoss(lam=0.0))
    scratch = {"_rollout": ctx}
    base = lambda v, s, sc: _GainSweepRegent().decide(v, s, sc)
    chosen = probe.propose_hook(_GainSweepRegent(), ctx.system.observe(), _iface().action_space(ctx.system, "regent:0"), scratch, base)
    assert chosen[0].payload["expr"] == "-1.9000 * current_x"  # the strongest, pole-cancelling gain
    assert scratch["rollout_probe"]["n_candidates"] == 4


def test_rollout_probe_passes_through_without_context():
    """The precondition gate: no scratch['_rollout'] (non-rollable system) ⇒ pure pass-through."""
    probe = RolloutProbe()
    sentinel = [ActionRequest("regent:0", "set_control_input", {"expr": "0"})]
    out = probe.propose_hook(object(), None, None, {}, lambda v, s, sc: sentinel)
    assert out is sentinel


def test_rollout_probe_end_to_end_through_runner():
    def factory(seed: int) -> CubicSystem:
        s = CubicSystem({"param_A": 0.95, "param_B": 0.5, "sigma_epsilon": 0.05})
        s.reset(seed)
        s.current_x = 1.0
        return s

    exp = Experiment(
        name="probe_smoke",
        system_factory=factory,
        action_interface=ScalarLeverInterface([Lever("set_control_input", (-3.0, 3.0), "current_u")]),
        regents={"regent:0": _GainSweepRegent()},
        objectives={"regent:0": StabilizationLoss(lam=0.05)},
        schedule=EveryN(20),
        seeds=[0],
        horizon=120,
        hypothesis=Hypothesis(id="H-probe", claim="probe runs", baseline="no-probe", primary_metric="mse"),
        harness=Harness([RolloutProbe(n_candidates=4, horizon=30, seeds=(0, 1), lam=0.25)]),
    )
    rec = Runner().run(exp)[0]
    # the probe drove a strong gain ⇒ the plant is stabilized near 0 by the end
    assert abs(rec.metrics_series[-1]["current_x"]) < 0.3
    assert rec.components["regent:0"]["mse"] < 0.2
