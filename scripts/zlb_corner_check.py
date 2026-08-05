"""Is the monetary world's clairvoyant optimum a real constraint, or an edge of our search grid?

The paper reports adaptation headroom of 1.359x on the monetary world, and its clairvoyant reference
sits at the grid's LOWEST intercept. A corner optimum is exactly the thing that makes a headroom
number an artifact: the search wanted to go further and could not, so the reference is weaker than
the world allows and the headroom is a lower bound of unknown tightness.

Here the corner has an economic name --- the zero lower bound --- but the intercept of a Taylor rule
is not the rate itself, and a rule with a NEGATIVE intercept is a real policy ("stay at the floor
through mild inflation"). So the corner has to be tested rather than explained away.

This script re-runs the decomposition with the intercept grid extended well below zero and reports
both numbers. It exists because the paper previously quoted the extended-grid figure from an ad-hoc
computation that was never saved --- a number in a manuscript with no artifact behind it, which is
the same defect as a stale one and harder to notice.

Run: ``uv run python scripts/zlb_corner_check.py --seeds 8``
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from govsim.analysis.calibration import PolicyFamily  # noqa: E402

TEMPLATE = "{base} + {g} * (inflation - inflation_target + 0.5 * output_gap)"
GAINS = [0.0, 0.5, 1.0, 2.0]
#: The grid the paper's headline uses: intercepts at or above the zero lower bound.
NARROW = [0.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]
#: The same grid extended below the bound. An intercept of -8 with a positive gain is a rule that
#: holds the floor through a substantial inflation overshoot; it is expressible policy, not nonsense.
WIDE = [-8.0, -4.0, -2.0] + NARROW


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--out", default="logs/zlb_corner.json")
    args = ap.parse_args()

    from decompose_library import decompose
    from headroom_audit import domain_monetary

    seeds = list(range(args.seeds))
    rows = {}
    for label, grid in (("narrow (ZLB floor)", NARROW), ("extended (below ZLB)", WIDE)):
        spec = dict(domain_monetary(),
                    family=PolicyFamily(verb="set_policy_rate", template=TEMPLATE,
                                        grid={"base": grid, "g": GAINS}))
        r = decompose(spec, seeds, spec["switch_step"])
        rows[label] = {k: r[k] for k in
                       ("frozen_loss", "best_fixed_loss", "switching_loss",
                        "adaptation_headroom", "robustness_headroom", "staleness")}
        rows[label]["switching_post_law"] = r["switching_law"]["post"]
        print(f"\n=== {label} ===")
        print(f"  frozen {r['frozen_loss']:.3f}  best_fixed {r['best_fixed_loss']:.3f}  "
              f"switching {r['switching_loss']:.3f}")
        print(f"  adaptation={r['adaptation_headroom']:.4f}  "
              f"robustness={r['robustness_headroom']:.4f}  staleness={r['staleness']:.4f}")
        print(f"  clairvoyant post-break law: {r['switching_law']['post']}")

    a = rows["narrow (ZLB floor)"]["adaptation_headroom"]
    b = rows["extended (below ZLB)"]["adaptation_headroom"]
    rows["verdict"] = {
        "narrow": a, "extended": b, "gain_from_extending": b - a,
        "corner_is_binding": abs(b - a) > 0.02,
    }
    print(f"\nextending the grid below the ZLB moves adaptation headroom "
          f"{a:.3f} -> {b:.3f} ({b - a:+.4f})")
    print("  => the corner is a REAL constraint, not a grid edge" if abs(b - a) <= 0.02
          else "  => the corner WAS binding; the headline number is a lower bound")

    Path(args.out).write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
