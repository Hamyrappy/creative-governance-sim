"""Decompose every candidate world into ADAPTATION vs ROBUSTNESS headroom.

``headroom_audit.py`` ranks worlds by ``frozen / post-shock-optimal`` on the post-break window. That
ratio answers "how much does staleness cost here", and it is the wrong question for choosing a
flagship. It bundles two effects that call for opposite institutional conclusions:

    staleness  =  adaptation_headroom  x  robustness_headroom
    L(frozen)     L(best_fixed)             L(frozen)
    ---------  =  -------------      x     -------------
    L(switch)     L(switch)                L(best_fixed)

Only the FIRST factor is a case for a government that pays attention. The second says the incumbent
rule was badly chosen to begin with — a *better constant* would have fixed it, no adaptation
required. A benchmark built on a world where the ratio is mostly robustness is asking agents to
demonstrate a capability the world does not reward.

We have already been caught by exactly this: the epidemic at high intervention price scores 1.44x
staleness with adaptation headroom of 1.000. Every unit of that 44% is recoverable by picking a
better fixed rule. An agent that adapts brilliantly and an agent that cannot adapt at all score the
same there, and the loss table would have looked like a real result.

Run: ``uv run python scripts/decompose_library.py --seeds 8``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from govsim.analysis.calibration import calibrate, calibrate_switching, diagnose


def decompose(spec: dict, seeds: list[int], switch_step: int) -> dict:
    """The three-way split for one world.

    Every reference is scored on the SHOCKED world over the FULL horizon, because that is the world
    the agents actually run in and a post-window-only metric rewards passivity (preregistration
    section 8). This is the one place the audit and the decomposition deliberately disagree: the
    audit scores on ``post_loss`` to isolate the break, while a reference an agent is compared
    against must be scored the way the agent is.
    """
    from govsim.analysis.calibration import _score_expr

    fam, iface, sched, hz = spec["family"], spec["iface"], spec["schedule"], spec["horizon"]
    common = dict(action_interface=iface, objective=spec["objective"], schedule=sched,
                  seeds=seeds, horizon=hz, metric="loss")

    # (1) best_fixed: the best SINGLE law on the shocked world, chosen with full knowledge of the
    #     break. The non-adaptive ceiling — what a very well-advised authority that legislates once
    #     and never revisits achieves.
    best_fixed = calibrate(fam, system_factory=spec["shocked"], **common)

    # (2) switching: the best (pre, post) PAIR, searched JOINTLY. Composing it from two separately
    #     calibrated legs can produce a "bound" below the thing it bounds, because the pre-leg
    #     determines the state the post-world is entered in.
    pre_laws, post_laws, switch_loss = calibrate_switching(
        {fam.verb: fam}, switch_step=switch_step, system_factory=spec["shocked"], **common)

    # (3) frozen: the law that was optimal BEFORE the break, still in force after it. Calibrated on
    #     the stationary pre-break world, then scored on the shocked one.
    frozen = calibrate(fam, system_factory=spec["pre"], **common)
    # Re-score with the FULL law set (``best_laws``), never ``best_expr``: the latter renders only
    # the family's primary verb and silently drops every other lever on a multi-instrument family,
    # so the reference would govern with one hand while the calibration that chose it used both.
    frozen_loss = _score_expr(frozen.best_laws, verb=fam.verb,
                              system_factory=spec["shocked"], **common)

    d = diagnose(frozen_loss, best_fixed.best_loss, switch_loss)
    return {
        "name": spec["name"],
        "frozen_law": frozen.best_laws,
        "best_fixed_law": best_fixed.best_laws,
        "switching_law": {"pre": pre_laws, "post": post_laws},
        "frozen_loss": frozen_loss,
        "best_fixed_loss": best_fixed.best_loss,
        "switching_loss": switch_loss,
        **d,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--only", nargs="*", default=None, help="substring filter on domain name")
    ap.add_argument("--json", default="logs/decomposition.json")
    args = ap.parse_args()

    from scripts.headroom_audit import DOMAINS

    seeds = list(range(args.seeds))
    rows = []
    for factory in DOMAINS:
        spec = factory()
        if args.only and not any(s in spec["name"] for s in args.only):
            continue
        step = spec.get("switch_step")
        if step is None:
            print(f"\n=== {spec['name']} ===\n  SKIPPED: no switch_step declared, so 'when the "
                  f"clairvoyant is allowed to change its mind' is undefined for this world.")
            continue
        print(f"\n=== {spec['name']} (switch at step {step}) ===", flush=True)
        try:
            row = decompose(spec, seeds, step)
        except Exception as exc:  # a world that cannot be decomposed must say so, not vanish
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            continue
        rows.append(row)
        print(f"  frozen     {row['frozen_loss']:>12.4f}   {row['frozen_law']}")
        print(f"  best_fixed {row['best_fixed_loss']:>12.4f}   {row['best_fixed_law']}")
        print(f"  switching  {row['switching_loss']:>12.4f}   {row['switching_law']}")
        if row["ratios_valid"]:
            print(f"  staleness={row['staleness']:.3f} = adaptation={row['adaptation_headroom']:.3f}"
                  f" x robustness={row['robustness_headroom']:.3f}")
            if row["adaptation_headroom"] < 1.0 or row["staleness"] < 1.0:
                print("  [!!] a headroom below 1.0 is IMPOSSIBLE — the clairvoyant reference lost")
                print("       to a reference it dominates by construction. The search is broken,")
                print("       not the world.")
        else:
            print("  RATIOS UNDEFINED (a loss is <= 0 — this is a net welfare objective, so a good")
            print("  policy legitimately scores below zero). Absolute gaps instead:")
            print(f"    staleness gap={row['staleness_gap']:.4f}  "
                  f"adaptation gap={row['adaptation_gap']:.4f}  "
                  f"robustness gap={row['robustness_gap']:.4f}")

    # Rank ONLY on valid ratios. A nan or inf sorts unpredictably and can take first place while
    # meaning nothing — commons did exactly that at inf x, above a world with a real 1.355x.
    ranked = [r for r in rows if r["ratios_valid"]]
    unranked = [r for r in rows if not r["ratios_valid"]]
    ranked.sort(key=lambda r: r["adaptation_headroom"], reverse=True)
    print("\n=== ranked by ADAPTATION headroom (the only factor an adaptive agent can claim) ===")
    print(f"{'adapt':>7} {'robust':>8} {'stale':>7}  world")
    for r in ranked:
        print(f"{r['adaptation_headroom']:>6.3f}x {r['robustness_headroom']:>7.3f}x "
              f"{r['staleness']:>6.3f}x  {r['name']}")
    for r in unranked:
        print(f"{'  n/a':>7} {'n/a':>8} {'n/a':>7}  {r['name']}  "
              f"(adaptation gap={r['adaptation_gap']:.4f} — ratios undefined, NOT ranked)")
    rows = ranked + unranked

    Path(args.json).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
