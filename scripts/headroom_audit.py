"""
Cross-domain headroom audit — decide WHICH world the headline experiment should live in.

For every candidate domain we calibrate the same two references inside one shared policy family
(``govsim.analysis.calibration``):

    frozen  — optimal on the PRE-shock (stationary) world, then deployed unchanged through the break
    oracle  — optimal on the post-shock WINDOW of the actual shocked world (clairvoyant)

and report ``headroom = L(frozen)/L(oracle)`` on that window. Headroom is the whole budget an
adaptive regent competes for, so a domain with headroom ≈ 1 cannot host the experiment no matter
how good the regent is.

**Calibrate on the objective, never on one of its components.** A first pass here calibrated the
scalar arms on ``mse`` and the epidemic arm on ``total_infected`` — i.e. on the benefit side with
the cost side deleted. Every optimum then sat at a corner ("use the maximum lever"), and a corner
optimum is regime-invariant: no shock can move it, so every domain reported ≈1.0x headroom. That
was an artifact of the calibration metric, not a fact about the worlds.

Running this *before* committing to a flagship domain is the difference between "we found no
effect" and "we chose a world where no effect was findable".

    uv run python scripts/headroom_audit.py --seeds 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from govsim.analysis import PolicyFamily, calibrate, headroom
from govsim.core.schedule import EveryN
from govsim.domains.scalar import (
    CoupledSystem,
    CubicSystem,
    EpidemicLoss,
    Lever,
    ScalarLeverInterface,
    SIRSystem,
    StabilizationLoss,
)

SHOCK_STEP = 100


# --------------------------------------------------------------------------------------------
# Domain specs: each returns (pre_factory, post_factory, shocked_factory, iface, objective,
# family, schedule, horizon, metric). ``pre``/``post`` are *stationary* worlds used only to
# calibrate the two references; ``shocked`` is the real world both are then scored on.
# --------------------------------------------------------------------------------------------


def _mk(cls, cfg):
    def factory(seed: int):
        s = cls(cfg)
        s.reset(seed)
        return s
    return factory


def domain_cubic():
    base = {"param_A": 0.95, "param_B": 0.5, "param_C": 0.0, "sigma_epsilon": 0.08,
            "target_x": 0.0, "u_range": (-2.0, 2.0), "cubic_coeff": 0.05, "state_exponent": 3}
    shocked = dict(base, shock_step=SHOCK_STEP,
                   shock_params={"param_A": 1.03, "param_B": 0.12, "cubic_coeff": 0.10},
                   shock_state_kick=0.8)
    iface = ScalarLeverInterface([Lever("set_control_input", (-2.0, 2.0), "current_u")])
    family = PolicyFamily(
        verb="set_control_input",
        template="-({k} * current_x + {c} * current_x ** 3)",
        grid={"k": [0.5, 0.9, 1.3, 1.8, 2.5, 3.5, 6.0, 10.0],
              "c": [0.0, 0.1, 0.4, 1.0, 3.0]},
    )
    return dict(
        name="cubic (scalar plant)",
        pre=_mk(CubicSystem, base), shocked=_mk(CubicSystem, shocked), iface=iface,
        objective=StabilizationLoss(lam=0.1, post_shock_step=SHOCK_STEP),
        family=family, schedule=EveryN(25), horizon=200,
        calib_metric="loss", score_metric="post_loss",
    )


def domain_coupled():
    base = {"param_A": 0.95, "param_B": 0.4, "param_C": 0.0, "sigma_epsilon": 0.10,
            "target_x": 0.0, "u_range": (-2.0, 2.0), "u_smoothing_rho": 0.70,
            "shock_period": 0, "param_B_drift_sigma": 0.0}
    shocked = dict(base, shock_period=SHOCK_STEP, shock_magnitude_aux1=1.0,
                   shock_magnitude_aux2=-0.8, param_B_drift_sigma=0.01)
    iface = ScalarLeverInterface([Lever("set_control_input", (-2.0, 2.0), "u_commanded")])
    family = PolicyFamily(
        verb="set_control_input",
        template="-({k} * current_x + {d} * (current_x - previous_x))",
        grid={"k": [0.5, 1.0, 1.5, 2.2, 3.0, 4.5, 7.0],
              "d": [0.0, 0.5, 1.5, 3.0, 5.0]},
    )
    return dict(
        name="coupled (multi-state plant)",
        pre=_mk(CoupledSystem, base), shocked=_mk(CoupledSystem, shocked), iface=iface,
        objective=StabilizationLoss(lam=0.1, post_shock_step=SHOCK_STEP),
        family=family, schedule=EveryN(20), horizon=300,
        calib_metric="loss", score_metric="post_loss",
    )


def domain_sir_efficacy(lam: float = 0.02, efficacy: float = 0.25):
    """The instrument-failure regime: lockdown keeps costing what it always cost, and stops working.

    A threshold-on-prevalence rule is feedback, so it shrugs off a transmissibility shock. It cannot
    shrug off this one: prevalence rises, the rule locks down *harder*, and every unit of lockdown
    now buys a quarter of what it used to while costing the same. The optimal response — spend less
    on the broken instrument — is the opposite of what the rule does, which is exactly the wedge
    adaptation has to earn.
    """
    base = {"beta0": 0.35, "gamma": 0.10, "noise_sigma": 0.0, "shock_step": 10_000,
            "waning": 0.02, "import_rate": 0.0005, "lockdown_efficacy": 1.0}
    shocked = dict(base, shock_step=SHOCK_STEP, shock_factor=1.0,
                   shock_params={"lockdown_efficacy": efficacy})
    iface = ScalarLeverInterface([
        Lever("set_lockdown", (0.0, 0.9), "lockdown"),
        Lever("set_vaccination", (0.0, 0.5), "vacc"),
    ])
    family = PolicyFamily(
        verb="set_lockdown",
        template="{a} if I > {thr} else 0.0",
        grid={"a": [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9],
              "thr": [0.002, 0.005, 0.01, 0.02, 0.04, 0.08, 0.15]},
    )
    return dict(
        name=f"SIR efficacy-collapse (lam={lam}, eff {1.0}->{efficacy})",
        pre=_mk(SIRSystem, base), shocked=_mk(SIRSystem, shocked), iface=iface,
        objective=EpidemicLoss(lam=lam, post_shock_step=SHOCK_STEP),
        family=family, schedule=EveryN(10), horizon=200,
        calib_metric="loss", score_metric="post_loss",
    )


def domain_sir(lam: float = 0.02):
    """Epidemic governance: a threshold lockdown institution meets a more transmissible variant.

    Bounded by construction (S+I+R is conserved), so no arm can 'diverge' — the comparison is
    always between two real policies, never between two blow-ups. The policy family is a
    *threshold institution* ('lock down at intensity a whenever prevalence exceeds θ'), which is
    what an actual public-health rule looks like and what a code-emitting regent can express
    natively.

    ``lam`` prices lockdown against infection. It has to be tuned until the optimum is interior:
    at λ=1.0 the cheapest thing to do is lock down maximally forever, which is regime-invariant and
    therefore has no headroom (and is also not a description of any real polity).
    """
    # ENDEMIC configuration: waning immunity + a trickle of imported cases, so the disease is still
    # circulating when the variant lands. Without this the epidemic can burn out before the shock
    # and "do nothing" wins the post-shock window by emptying it.
    base = {"beta0": 0.35, "gamma": 0.10, "noise_sigma": 0.0, "shock_step": 10_000,
            "waning": 0.02, "import_rate": 0.0005}
    shocked = dict(base, shock_step=SHOCK_STEP, shock_factor=1.8)
    iface = ScalarLeverInterface([
        Lever("set_lockdown", (0.0, 0.9), "lockdown"),
        Lever("set_vaccination", (0.0, 0.5), "vacc"),
    ])
    family = PolicyFamily(
        verb="set_lockdown",
        template="{a} if I > {thr} else 0.0",
        grid={"a": [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9],
              "thr": [0.002, 0.005, 0.01, 0.02, 0.04, 0.08, 0.15]},
    )
    return dict(
        name=f"SIR (epidemic governance, lam={lam})",
        pre=_mk(SIRSystem, base), shocked=_mk(SIRSystem, shocked), iface=iface,
        objective=EpidemicLoss(lam=lam, post_shock_step=SHOCK_STEP),
        family=family, schedule=EveryN(10), horizon=200,
        calib_metric="loss", score_metric="post_loss",
    )


DOMAINS = [
    domain_cubic, domain_coupled,
    lambda: domain_sir(0.02),
    lambda: domain_sir_efficacy(0.02, 0.50),
    lambda: domain_sir_efficacy(0.02, 0.25),
    lambda: domain_sir_efficacy(0.02, 0.00),
    lambda: domain_sir_efficacy(0.08, 0.25),
]


def audit(spec: dict, seeds: list[int]) -> dict:
    from govsim.analysis.calibration import _score_expr

    fam, iface, sched, hz = spec["family"], spec["iface"], spec["schedule"], spec["horizon"]
    cm, sm = spec["calib_metric"], spec["score_metric"]
    print(f"\n=== {spec['name']} — calibrating {fam.size()} laws x 2 regimes x {len(seeds)} seeds ===")
    sys.stdout.flush()

    # frozen: optimal on the stationary PRE-shock world, on the full-horizon loss.
    frozen_cal = calibrate(fam, system_factory=spec["pre"], action_interface=iface,
                           objective=spec["objective"], schedule=sched, seeds=seeds,
                           horizon=hz, metric=cm)
    # oracle: optimal on the post-shock WINDOW of the world that actually gets shocked. This is the
    # honest clairvoyant: same family, same window, same seeds — it differs from `frozen` only in
    # *when* it was allowed to look.
    oracle_cal = calibrate(fam, system_factory=spec["shocked"], action_interface=iface,
                           objective=spec["objective"], schedule=sched, seeds=seeds,
                           horizon=hz, metric=sm)
    print(f"  frozen (pre-shock optimal):  {frozen_cal.best_expr}")
    print(f"  oracle (post-shock optimal): {oracle_cal.best_expr}")

    kw = dict(verb=fam.verb, system_factory=spec["shocked"], action_interface=iface,
              objective=spec["objective"], schedule=sched, seeds=seeds, horizon=hz, metric=sm)
    frozen_loss = _score_expr(frozen_cal.best_expr, **kw)
    oracle_loss = _score_expr(oracle_cal.best_expr, **kw)
    h = headroom(frozen_loss, oracle_loss)
    same = frozen_cal.best_expr == oracle_cal.best_expr
    print(f"  on the SHOCKED world ({sm}): frozen={frozen_loss:.5f}  oracle={oracle_loss:.5f}  "
          f"=> HEADROOM = {h:.2f}x  (frozen carries {100*(h-1):.0f}% excess loss)"
          f"{'   [!] identical policies — the shock does not move the optimum' if same else ''}")
    sys.stdout.flush()
    return {
        "domain": spec["name"], "frozen_expr": frozen_cal.best_expr,
        "oracle_expr": oracle_cal.best_expr, "frozen_loss": frozen_loss,
        "oracle_loss": oracle_loss, "headroom": h, "metric": sm,
        "family_size": fam.size(), "same_policy": same,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--json", help="write results here")
    ap.add_argument("--only", help="audit a single domain by substring match")
    args = ap.parse_args()
    seeds = list(range(args.seeds))

    specs = [d() for d in DOMAINS]
    if args.only:
        specs = [s for s in specs if args.only.lower() in s["name"].lower()]
    rows = [audit(s, seeds) for s in specs]

    print("\n=== headroom ranking (higher = more room for adaptation to matter) ===")
    for r in sorted(rows, key=lambda r: -r["headroom"]):
        print(f"  {r['headroom']:>8.2f}x  {r['domain']}")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
