"""
Regenerate ``govsim/docs_gates/calibration.json`` — the frozen/oracle reference policies.

These two laws anchor every normalized-regret number the paper reports, so they are produced by a
script with its seed count and search grid recorded alongside them, and never typed by hand.

    uv run python scripts/recalibrate.py --seeds 12

Re-run this after ANY change to a regime in ``govsim/domains/scalar/regimes.py`` — a stale
calibration silently re-anchors the whole scale.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from govsim.analysis import PolicyFamily, calibrate, calibrate_families, headroom
from govsim.analysis.calibration import _score_expr
from govsim.core.schedule import EveryN
from govsim.domains.scalar import EpidemicLoss, Lever, ScalarLeverInterface, StabilizationLoss
from govsim.domains.scalar import regimes as R

EPIDEMIC_IFACE = ScalarLeverInterface([
    Lever("set_lockdown", (0.0, 0.9), "lockdown"),
    Lever("set_vaccination", (0.0, 0.5), "vacc"),
])
# The reference family is a THRESHOLD INSTITUTION: "lock down at intensity a whenever prevalence
# exceeds θ". It is what a public-health rule actually looks like, and it is deliberately the same
# language for both references, so the comparison is about *when* the parameters were chosen and
# never about who had the richer vocabulary.
EPIDEMIC_FAMILY = PolicyFamily(
    verb="set_lockdown",
    template="{a} if I > {thr} else 0.0",
    grid={"a": [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9],
          "thr": [0.002, 0.01, 0.03, 0.06, 0.10, 0.16, 0.25, 0.40, 0.70]},
)

# The WIDE reference vocabulary. A regent that emits code is not confined to the threshold form, so
# an oracle that is only allowed thresholds can be beaten on shape rather than on adaptation — and
# a reviewer will say so. These add the two shapes a regent actually reaches for: a proportional
# response, and a threshold with a non-zero floor. Reporting regret against both the narrow
# institutional family and this wider one keeps "beat the rule-maker" separate from "beat the best
# policy we can construct".
EPIDEMIC_FAMILIES = {
    "threshold": EPIDEMIC_FAMILY,
    "proportional": PolicyFamily(
        verb="set_lockdown",
        template="{g} * I + {b}",
        grid={"g": [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
              "b": [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75]},
    ),
    "threshold_floor": PolicyFamily(
        verb="set_lockdown",
        template="{a} if I > {thr} else {b}",
        grid={"a": [0.15, 0.45, 0.75, 0.9],
              "thr": [0.01, 0.06, 0.16, 0.40],
              "b": [0.0, 0.05, 0.15, 0.3]},
    ),
}

SCALAR_IFACE = ScalarLeverInterface([Lever("set_control_input", R.SCALAR_U_RANGE, "current_u")])
SCALAR_FAMILY = PolicyFamily(
    verb="set_control_input",
    template="-({k} * current_x + {c} * current_x ** 3)",
    grid={"k": [0.0, 0.6, 1.3, 1.9, 2.6, 3.6, 5.0, 8.0],
          "c": [0.0, 0.3, 1.0, 3.0]},
)


def _regime_spec(name: str) -> dict:
    if name == "epidemic":
        return dict(
            family=EPIDEMIC_FAMILY, families=EPIDEMIC_FAMILIES, iface=EPIDEMIC_IFACE,
            pre=R.sir_factory(R.EPIDEMIC_PRE), shocked=R.sir_factory(R.EPIDEMIC_SHOCKED),
            objective=EpidemicLoss(lam=R.EPIDEMIC_LAMBDA, post_shock_step=R.EPIDEMIC_SHOCK_STEP),
            schedule=EveryN(R.EPIDEMIC_DECIDE_EVERY), horizon=R.EPIDEMIC_HORIZON,
        )
    if name == "scalar":
        return dict(
            family=SCALAR_FAMILY, families={"cubic_gain": SCALAR_FAMILY}, iface=SCALAR_IFACE,
            pre=R.cubic_factory(R.SCALAR_PRE), shocked=R.cubic_factory(R.SCALAR_SHOCKED),
            objective=StabilizationLoss(lam=R.SCALAR_LAMBDA, post_shock_step=R.SCALAR_SHOCK_STEP),
            schedule=EveryN(R.SCALAR_DECIDE_EVERY), horizon=R.SCALAR_HORIZON,
        )
    raise KeyError(name)


def run(name: str, seeds: list[int]) -> dict:
    s = _regime_spec(name)
    fam, iface, sched, hz, obj = s["family"], s["iface"], s["schedule"], s["horizon"], s["objective"]
    print(f"\n=== {name}: {fam.size()} laws x 2 regimes x {len(seeds)} seeds ===")
    sys.stdout.flush()

    frozen = calibrate(fam, system_factory=s["pre"], action_interface=iface, objective=obj,
                       schedule=sched, seeds=seeds, horizon=hz, metric="loss")
    oracle = calibrate(fam, system_factory=s["shocked"], action_interface=iface, objective=obj,
                       schedule=sched, seeds=seeds, horizon=hz, metric="post_loss")

    # The WIDE oracle: same clairvoyance, but allowed to pick its functional form as well as its
    # parameters. This is the reference that makes "the regent beat the oracle" mean something.
    wide_name, wide = calibrate_families(
        s["families"], system_factory=s["shocked"], action_interface=iface, objective=obj,
        schedule=sched, seeds=seeds, horizon=hz, metric="post_loss")

    skw = dict(verb=fam.verb, system_factory=s["shocked"], action_interface=iface, objective=obj,
               schedule=sched, seeds=seeds, horizon=hz, metric="post_loss")
    fl = _score_expr(frozen.best_expr, **skw)
    ol = _score_expr(oracle.best_expr, **skw)
    wl = _score_expr(wide.best_expr, **skw)
    h, hw = headroom(fl, ol), headroom(fl, wl)
    print(f"  frozen      : {frozen.best_expr}")
    print(f"  oracle      : {oracle.best_expr}   (narrow family)")
    print(f"  oracle_wide : {wide.best_expr}   (best of {sorted(s['families'])} -> '{wide_name}')")
    print(f"  post_loss on the shocked world: frozen={fl:.5f}  oracle={ol:.5f}  oracle_wide={wl:.5f}")
    print(f"  headroom (narrow)={h:.3f}x   headroom (wide)={hw:.3f}x")
    return {
        "verb": fam.verb,
        "frozen": {"expr": frozen.best_expr, "params": frozen.best_params, "post_loss": fl},
        "oracle": {"expr": oracle.best_expr, "params": oracle.best_params, "post_loss": ol},
        "oracle_wide": {"expr": wide.best_expr, "params": wide.best_params, "post_loss": wl,
                        "family": wide_name},
        "headroom": h,
        "headroom_wide": hw,
        "provenance": {
            "n_seeds": len(seeds), "seeds": seeds, "family_template": fam.template,
            "family_grid": fam.grid, "family_size": fam.size(),
            "wide_families": {n: f.template for n, f in s["families"].items()},
            "horizon": hz, "decide_every": sched.n if hasattr(sched, "n") else None,
            "calibrated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--regimes", nargs="+", default=["epidemic", "scalar"])
    args = ap.parse_args()
    seeds = list(range(args.seeds))

    out = {name: run(name, seeds) for name in args.regimes}
    path = R.CALIBRATION_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    existing.update(out)
    path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
