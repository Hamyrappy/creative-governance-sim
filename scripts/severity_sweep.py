"""
Calibrate the H1 regime — turn the hand-dialed shock severity into an auditable design choice.

The problem this solves (``docs_gates/STATUS.md``, the ☐ AUTHOR knob): the size of the structural
shock decides whether H1 is even *answerable*. Too mild and a frozen pre-shock controller is still
near-optimal, so every arm ties and a null result says nothing about regents. Too harsh and every
arm diverges, so the comparison is between two collapses. Picking that knob by eye is a researcher
degree of freedom a reviewer will (rightly) attack.

So we measure it instead. For each candidate regime we run two *reference* controllers, both
key-free and deterministic:

  frozen  — LQR for the PRE-shock linearization, held fixed through the shock (never adapts)
  oracle  — clairvoyant: LQR for the POST-shock linearization plus exact cancellation of the
            nonlinearity (``OracleRegent``); the achievable end of the scale

Their gap **is** the headroom: how much post-shock loss adaptation could possibly recover.

    headroom = post_mse(frozen) / post_mse(oracle)

A regime with headroom ≈ 1 cannot falsify H1 no matter how good the regent is; a regime where the
oracle itself diverges is not a control problem. We report the whole surface and pick an operating
point inside the usable band, then pre-register it.

    uv run python scripts/severity_sweep.py                 # the default grid
    uv run python scripts/severity_sweep.py --seeds 20 --json out.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from govsim.core.experiment import Experiment, Hypothesis
from govsim.core.regent import ScriptedRegent
from govsim.core.runner import Runner
from govsim.core.schedule import EveryN
from govsim.domains.scalar import CubicSystem, Lever, ScalarLeverInterface, StabilizationLoss
from govsim.regents import LQRRegent, OracleRegent, PIDRegent

# The pre-shock plant every arm is built against (what a pre-shock-optimal designer would know).
A_PRE, B_PRE, G_PRE = 0.95, 0.5, 0.05
SHOCK_STEP = 100
HORIZON = 200
LAM = 0.1
U_RANGE = (-2.0, 2.0)


def make_factory(shock_params: dict[str, float], kick: float, sigma: float = 0.08):
    cfg = {
        "param_A": A_PRE, "param_B": B_PRE, "param_C": 0.0, "sigma_epsilon": sigma,
        "target_x": 0.0, "u_range": U_RANGE, "cubic_coeff": G_PRE, "state_exponent": 3,
        "shock_step": SHOCK_STEP,
        "shock_params": dict(shock_params),
        "shock_state_kick": kick,
    }

    def factory(seed: int) -> CubicSystem:
        sys_ = CubicSystem(cfg)
        sys_.reset(seed)
        return sys_

    return factory


def _experiment(name: str, factory, regent, seeds: list[int]) -> Experiment:
    return Experiment(
        name=name,
        system_factory=factory,
        action_interface=ScalarLeverInterface([Lever("set_control_input", U_RANGE, "current_u")]),
        regents={"regent:0": regent},
        objectives={"regent:0": StabilizationLoss(lam=LAM, post_shock_step=SHOCK_STEP)},
        schedule=EveryN(25),
        seeds=seeds,
        horizon=HORIZON,
        hypothesis=Hypothesis(
            id="H1-calibration", claim="severity calibration probe", baseline="frozen LQR vs oracle",
            primary_metric="post_mse",
        ),
    )


def arm_post_mse(name: str, factory, regent, seeds: list[int]) -> dict[str, float]:
    """Run one reference controller over the seeds; summarize its post-shock loss + collapses."""
    recs = Runner().run(_experiment(name, factory, regent, seeds))
    vals = [r.components["regent:0"]["post_mse"] for r in recs]
    finite = [v for v in vals if math.isfinite(v)]
    n_div = sum(1 for r in recs if r.terminated_at_step is not None) + (len(vals) - len(finite))
    return {
        "mean": statistics.mean(finite) if finite else float("inf"),
        "median": statistics.median(finite) if finite else float("inf"),
        "max": max(finite) if finite else float("inf"),
        "n_diverged": n_div,
    }


def evaluate_cell(label: str, shock_params: dict[str, float], kick: float, seeds: list[int]) -> dict:
    """Run all four reference controllers on one candidate regime and score its headroom."""
    factory = make_factory(shock_params, kick)
    a_post = float(shock_params.get("param_A", A_PRE))
    b_post = float(shock_params.get("param_B", B_PRE))
    g_post = float(shock_params.get("cubic_coeff", G_PRE))

    frozen = arm_post_mse("frozen", factory,
                          LQRRegent("set_control_input", A=A_PRE, B=B_PRE, Q=1.0, R=LAM), seeds)
    oracle = arm_post_mse("oracle", factory,
                          OracleRegent("set_control_input", A_post=a_post, B=b_post, Q=1.0, R=LAM,
                                       cubic_coeff_post=g_post), seeds)
    # A *tuned but non-adaptive* rival: the strongest thing a designer gets without seeing the shock.
    # If it already closes the gap, the regime rewards tuning, not adaptation, and H1 would be
    # measuring the wrong thing.
    pid = arm_post_mse("pid", factory, PIDRegent("set_control_input", kp=1.4, kd=0.3), seeds)
    nocontrol = arm_post_mse("none", factory,
                             ScriptedRegent(verb="set_control_input", expr="0.0"), seeds)

    headroom = (frozen["mean"] / oracle["mean"]) if oracle["mean"] > 0 else float("inf")
    row = {
        "label": label, "shock_params": dict(shock_params), "kick": kick,
        "frozen": frozen, "oracle": oracle, "pid": pid, "nocontrol": nocontrol,
        "headroom": headroom,
        "pid_closes": (pid["mean"] / oracle["mean"]) if oracle["mean"] > 0 else float("inf"),
        # "Usable" = a regime where the question is answerable: neither reference blows up, the
        # oracle stays well-conditioned, and there is real distance between never-adapting and
        # clairvoyant. Divergence is disqualifying in BOTH directions — an infinite ratio is not
        # headroom, it is an unbounded plant.
        "usable": bool(
            oracle["n_diverged"] == 0 and frozen["n_diverged"] == 0
            and math.isfinite(headroom) and headroom >= 1.5
            and math.isfinite(frozen["max"]) and frozen["max"] < 1e4
        ),
    }
    print(f"{label:<26} kick={kick:<4} | frozen={frozen['mean']:>10.4f} oracle={oracle['mean']:>8.4f} "
          f"pid={pid['mean']:>10.4f} | headroom={headroom:>9.2f}x pid/oracle={row['pid_closes']:>8.2f}x "
          f"div(f/o)={frozen['n_diverged']}/{oracle['n_diverged']} "
          f"{'USABLE' if row['usable'] else ''}")
    sys.stdout.flush()
    return row


def candidate_regimes() -> list[tuple[str, dict[str, float], float]]:
    """The shock families we are choosing between, as (label, shock_params, state kick).

    Family A (``dyn:``) shocks the STATE dynamics — the plant gets less stable and more nonlinear.
    Family B (``ctrl:``) shocks CONTROL EFFECTIVENESS — the lever's gain weakens, collapses, or
    reverses sign while the plant itself stays stable. Family B is the Lucas-critique regime: the
    instrument that used to work stops working (or works backwards), so a frozen rule is not merely
    stale but actively wrong, and the state stays bounded because ``A < 1`` and ``u`` is clipped.
    Family C (``both:``) combines a mild dynamics shift with a control-effectiveness shift.
    """
    out: list[tuple[str, dict[str, float], float]] = []
    for a in (1.03, 1.20, 1.35):
        out.append((f"dyn: A={a}", {"param_A": a, "cubic_coeff": 0.10}, 0.8))
    for b in (0.25, 0.12, 0.05, -0.12, -0.25, -0.50):
        out.append((f"ctrl: B={b}", {"param_B": b}, 0.8))
    for b in (0.12, -0.25):
        out.append((f"both: A=1.03,B={b}", {"param_A": 1.03, "param_B": b}, 0.8))
        out.append((f"both: A=1.03,B={b},g=.1", {"param_A": 1.03, "param_B": b, "cubic_coeff": 0.10}, 0.8))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=10, help="number of seeds per cell")
    ap.add_argument("--json", help="write the full surface to this path")
    args = ap.parse_args()
    seeds = list(range(args.seeds))

    print(f"# severity sweep: pre-shock (A={A_PRE}, B={B_PRE}, g={G_PRE}), shock@{SHOCK_STEP}, "
          f"horizon={HORIZON}, n_seeds={len(seeds)}")
    print("# headroom = post_mse(frozen)/post_mse(oracle) — the maximum adaptation could recover\n")
    rows = [evaluate_cell(label, sp, kick, seeds) for label, sp, kick in candidate_regimes()]

    usable = [r for r in rows if r["usable"]]
    print(f"\n{len(usable)}/{len(rows)} cells usable (no divergence in either reference, "
          f"headroom >= 1.5x, frozen stays bounded)")
    if usable:
        usable.sort(key=lambda r: -r["headroom"])
        print("Ranked by headroom:")
        for r in usable:
            print(f"  {r['label']:<26} headroom={r['headroom']:.2f}x  (frozen {r['frozen']['mean']:.4f} -> "
                  f"oracle {r['oracle']['mean']:.4f}; tuned PID sits at {r['pid_closes']:.2f}x oracle)")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
