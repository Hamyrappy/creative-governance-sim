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

from govsim.analysis import (
    PolicyFamily, calibrate, calibrate_families, calibrate_switching, diagnose, headroom,
)
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
# The family spans BOTH instruments. This is not an embellishment: the regent arms can set
# vaccination as well as lockdown, so a reference confined to lockdown is not a weaker opponent, it
# is an unfair one — any arm would beat it partly by using a lever the reference was forbidden. It
# also turns out to matter substantively, though not in the way an earlier version of this comment
# claimed: calibrated jointly, the clairvoyant law holds vaccination at the SAME level in both legs
# (byte-identical), so the correct response to the instrument failure is pure withdrawal of the
# broken lever, not substitution toward the surviving one. What differs is that the frozen rule
# never used vaccination at all — a difference in the rule as written, which the decomposition
# assigns to robustness rather than adaptation.
EPIDEMIC_FAMILY = PolicyFamily(
    verb="set_lockdown",
    template="{a} if I > {thr} else 0.0",
    grid={"a": [0.0, 0.3, 0.45, 0.6, 0.75, 0.9],
          "thr": [0.002, 0.01, 0.03, 0.06, 0.10, 0.16, 0.25, 0.40],
          "v": [0.0, 0.1, 0.25, 0.5]},
    extra_laws={"set_vaccination": "{v}"},
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
        grid={"g": [0.0, 0.5, 1.0, 2.0, 4.0, 8.0],
              "b": [0.0, 0.05, 0.15, 0.35, 0.6],
              "v": [0.0, 0.1, 0.25, 0.5]},
        extra_laws={"set_vaccination": "{v}"},
    ),
    "threshold_floor": PolicyFamily(
        verb="set_lockdown",
        template="{a} if I > {thr} else {b}",
        grid={"a": [0.45, 0.75, 0.9],
              "thr": [0.01, 0.06, 0.16, 0.40],
              "b": [0.0, 0.05, 0.15, 0.3],
              "v": [0.0, 0.25, 0.5]},
        extra_laws={"set_vaccination": "{v}"},
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
            shock_step=R.EPIDEMIC_SHOCK_STEP,
        )
    if name == "epidemic_severe":
        return dict(
            family=EPIDEMIC_FAMILY, families=EPIDEMIC_FAMILIES, iface=EPIDEMIC_IFACE,
            pre=R.sir_factory(R.EPIDEMIC_PRE), shocked=R.sir_factory(R.EPIDEMIC_SEVERE_SHOCKED),
            objective=EpidemicLoss(lam=R.EPIDEMIC_SEVERE_LAMBDA,
                                   post_shock_step=R.EPIDEMIC_SHOCK_STEP),
            schedule=EveryN(R.EPIDEMIC_DECIDE_EVERY), horizon=R.EPIDEMIC_HORIZON,
            shock_step=R.EPIDEMIC_SHOCK_STEP,
        )
    if name == "epidemic_pricey":
        return dict(
            family=EPIDEMIC_FAMILY, families=EPIDEMIC_FAMILIES, iface=EPIDEMIC_IFACE,
            pre=R.sir_factory(R.EPIDEMIC_PRE), shocked=R.sir_factory(R.EPIDEMIC_PRICEY_SHOCKED),
            objective=EpidemicLoss(lam=R.EPIDEMIC_PRICEY_LAMBDA,
                                   post_shock_step=R.EPIDEMIC_SHOCK_STEP),
            schedule=EveryN(R.EPIDEMIC_DECIDE_EVERY), horizon=R.EPIDEMIC_HORIZON,
            shock_step=R.EPIDEMIC_SHOCK_STEP,
        )
    if name == "scalar":
        return dict(
            family=SCALAR_FAMILY, families={"cubic_gain": SCALAR_FAMILY}, iface=SCALAR_IFACE,
            pre=R.cubic_factory(R.SCALAR_PRE), shocked=R.cubic_factory(R.SCALAR_SHOCKED),
            objective=StabilizationLoss(lam=R.SCALAR_LAMBDA, post_shock_step=R.SCALAR_SHOCK_STEP),
            schedule=EveryN(R.SCALAR_DECIDE_EVERY), horizon=R.SCALAR_HORIZON,
            shock_step=R.SCALAR_SHOCK_STEP,
        )
    raise KeyError(name)


def run(name: str, seeds: list[int]) -> dict:
    """Calibrate the four references the full-horizon comparison needs.

    ``frozen``      best fixed law given only the pre-break world, held through the break.
    ``best_fixed``  best fixed law over the WHOLE broken horizon, chosen in hindsight. This is the
                    non-adaptive ceiling, and it is the reference that makes an adaptation claim
                    falsifiable: no fixed law can be optimal on both sides of a break that moves the
                    optimum, so beating it requires actually changing behaviour.
    ``oracle``      best fixed law for the post-break window (kept for continuity of the post-window
                    reporting; on its own it is NOT a sound target, see below).
    ``switching``   pre-break optimum until the break, post-break optimum after — the clairvoyant
                    adaptor, and the achievable end of the full-horizon scale.

    Why the extra references: scoring only the post-break window rewards *passivity*. A policy that
    never intervenes is wrong before the break and, when the break disables the instrument, nearly
    right after it, so it scores well on a post-window metric without having adapted to anything. We
    found this when the no-harness control arm scored suspiciously close to the post-window oracle.
    Over the full horizon that free lunch disappears.
    """
    s = _regime_spec(name)
    fam, iface, sched, hz, obj = s["family"], s["iface"], s["schedule"], s["horizon"], s["objective"]
    # The banner reports the size of the space EVERY reference searches. It said fam.size() while the
    # comparators searched three families, which is exactly the kind of stale launch banner that hid
    # the asymmetry documented below.
    n_laws = sum(f.size() for f in s["families"].values())
    print(f"\n=== {name}: {n_laws} laws across {len(s['families'])} families x {len(seeds)} seeds ===")
    sys.stdout.flush()

    # ALL FOUR REFERENCES SEARCH THE SAME POLICY VOCABULARY. This is load-bearing and was wrong for
    # several revisions: ``frozen`` and ``oracle`` searched the single flagship family (192 laws)
    # while ``best_fixed`` and ``switching`` searched all three (456). The papers' caption claims the
    # references "differ only in what they were allowed to know", and under the old asymmetry that
    # was false — the frozen rule was also handicapped in what it was allowed to SAY, over 42% of the
    # space its comparators got. Every ratio with ``frozen`` in the numerator (staleness, robustness
    # headroom) was therefore inflated by an unknown amount of pure vocabulary difference.
    #
    # Adaptation headroom, L(best_fixed)/L(switching), was never affected: both sides always used
    # calibrate_families. That is why the asymmetry survived so long — it left the headline number
    # alone and moved the two numbers around it.
    frozen_name, frozen = calibrate_families(
        s["families"], system_factory=s["pre"], action_interface=iface, objective=obj,
        schedule=sched, seeds=seeds, horizon=hz, metric="loss")
    oracle_name, oracle = calibrate_families(
        s["families"], system_factory=s["shocked"], action_interface=iface, objective=obj,
        schedule=sched, seeds=seeds, horizon=hz, metric="post_loss")
    # The non-adaptive ceiling, over the full horizon, chosen with hindsight.
    best_fixed_name, best_fixed = calibrate_families(
        s["families"], system_factory=s["shocked"], action_interface=iface, objective=obj,
        schedule=sched, seeds=seeds, horizon=hz, metric="loss")

    skw = dict(verb=fam.verb, system_factory=s["shocked"], action_interface=iface, objective=obj,
               schedule=sched, seeds=seeds, horizon=hz)
    fl = _score_expr(frozen.best_laws, metric="loss", **skw)
    bl = _score_expr(best_fixed.best_laws, metric="loss", **skw)
    fl_post = _score_expr(frozen.best_laws, metric="post_loss", **skw)
    ol_post = _score_expr(oracle.best_laws, metric="post_loss", **skw)

    # The clairvoyant adaptor: (pre-leg, post-leg) searched JOINTLY, because the pre-leg decides
    # the state the post-leg inherits. Composing it from two separately-optimal legs produced a
    # "clairvoyant" reference that a fixed law could beat — impossible for a real upper bound.
    pre_laws, post_laws, sl = calibrate_switching(
        s["families"], switch_step=s["shock_step"], system_factory=s["shocked"],
        action_interface=iface, objective=obj, schedule=sched, seeds=seeds, horizon=hz,
        metric="loss")

    d = diagnose(fl, bl, sl)
    h_full, h_vs_fixed = d["staleness"], d["adaptation_headroom"]
    print(f"  frozen      : {frozen.best_laws}   (pre-break only, '{frozen_name}')")
    print(f"  best_fixed  : {best_fixed.best_laws}   (hindsight, whole horizon, '{best_fixed_name}')")
    print(f"  oracle(post): {oracle.best_laws}   ('{oracle_name}')")
    print(f"  switching   : {pre_laws}")
    print(f"                ->  {post_laws}   at t={s['shock_step']}")
    print(f"  full-horizon loss: frozen={fl:.4f}  best_fixed={bl:.4f}  switching={sl:.4f}")
    print(f"  staleness (frozen/switch)      = {d['staleness']:.3f}x   "
          f"<- what the stale rule costs")
    print(f"  ADAPTATION headroom (fixed/sw) = {d['adaptation_headroom']:.3f}x   "
          f"<- recoverable ONLY by changing behaviour")
    print(f"  robustness headroom (fro/fix)  = {d['robustness_headroom']:.3f}x   "
          f"<- recoverable by legislating a better standing rule")
    if h_vs_fixed < 1.0:
        print("  [!!] the clairvoyant adaptor is WORSE than the best fixed law. That is impossible "
              "for a genuine upper bound (switching subsumes not-switching), so the switching "
              "search is under-powered — widen top_k or the vocabulary before trusting this cell.")
    elif h_vs_fixed < 1.05:
        print(f"  [!] adaptation buys ~nothing here: the best FIXED law matches the adaptor. The "
              f"{100 * (d['staleness'] - 1):.0f}% the stale rule costs is a RULE-DESIGN problem, "
              f"not an adaptation problem — a better standing rule recovers it without adapting.")
    return {
        "verb": fam.verb,
        "switch_step": s["shock_step"],
        "frozen": {"expr": frozen.best_expr, "laws": frozen.best_laws,
                   "params": frozen.best_params, "loss": fl, "post_loss": fl_post,
                   "family": frozen_name},
        "best_fixed": {"expr": best_fixed.best_expr, "laws": best_fixed.best_laws,
                       "params": best_fixed.best_params, "loss": bl, "family": best_fixed_name},
        "oracle": {"expr": oracle.best_expr, "laws": oracle.best_laws,
                   "params": oracle.best_params, "post_loss": ol_post, "family": oracle_name},
        "switching": {"pre_expr": pre_laws[fam.verb], "post_expr": post_laws[fam.verb],
                      "pre_laws": pre_laws, "post_laws": post_laws, "loss": sl},
        "headroom": h_full,               # == staleness, kept for artifact compatibility
        "headroom_vs_best_fixed": h_vs_fixed,
        "diagnosis": d,
        "provenance": {
            "n_seeds": len(seeds), "seeds": seeds, "family_template": fam.template,
            "family_grid": fam.grid, "family_size": fam.size(),
            "wide_families": {n: f.template for n, f in s["families"].items()},
            "horizon": hz, "decide_every": sched.n if hasattr(sched, "n") else None,
            "primary_metric": "loss (full horizon)",
            "vocabulary_symmetric": True,   # all four references search s["families"]
            "search_size": sum(f.size() for f in s["families"].values()),
            "calibrated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    }


def _score_switching(pre_expr, post_expr, s: dict, seeds: list[int], metric: str) -> float:
    """Score the clairvoyant adaptor through the normal Runner path, so it is comparable."""
    import math
    import statistics as st

    from govsim.core.experiment import Experiment, Hypothesis
    from govsim.core.runner import Runner
    from govsim.regents import SwitchingRegent

    exp = Experiment(
        name="calib:switching", system_factory=s["shocked"], action_interface=s["iface"],
        regents={"regent:0": SwitchingRegent(s["family"].verb, pre_expr, post_expr, s["shock_step"])},
        objectives={"regent:0": s["objective"]}, schedule=s["schedule"], seeds=seeds,
        horizon=s["horizon"],
        hypothesis=Hypothesis(id="calibration", claim="clairvoyant adaptor reference",
                              baseline="the same family without the switch", primary_metric=metric),
    )
    vals = [r.components["regent:0"][metric] for r in Runner().run(exp)]
    return float("inf") if any(not math.isfinite(v) for v in vals) else st.fmean(vals)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--regimes", nargs="+", default=["epidemic", "epidemic_severe", "epidemic_pricey", "scalar"])
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
