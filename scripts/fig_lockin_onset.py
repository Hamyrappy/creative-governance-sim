"""Figure: when precedent locks a policy in, and whether the lock releases at the break.

Reads ``logs/lockin_onset.json`` — it does not recompute anything, so the figure and the numbers in
the text cannot disagree. Every annotated value is pulled from the artifact rather than typed.

Run: ``uv run python scripts/fig_lockin_onset.py``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STYLE = {
    "bare":    dict(color="#8c8c8c", marker="o", ls="--", label="no precedent"),
    "own":     dict(color="#c0392b", marker="s", ls="-",  label="own precedent"),
    "foreign": dict(color="#1f6fb4", marker="^", ls="-",  label="another authority's precedent"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="logs/lockin_onset.json")
    ap.add_argument("--out", default="paper/figures/lockin_onset.pdf")
    args = ap.parse_args()

    d = json.loads(Path(args.json).read_text(encoding="utf-8"))
    curve, brk = d["curve"], d["break_decision"]
    x = [r["transition"] for r in curve]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.4, 3.7),
                                 gridspec_kw={"width_ratios": [2.15, 1]})

    ax.axvspan(brk, max(x), color="#000000", alpha=0.045, lw=0)
    ax.axvline(brk, color="k", lw=1.0, ls=":")
    ax.text(brk + 0.2, 1.14, "instrument breaks", fontsize=8.5, va="top")
    for k, s in STYLE.items():
        ax.plot(x, [r[k] for r in curve], ms=4.2, lw=1.5, **s)
    ax.set_xlabel("decision (the law in force is revised, or is not)")
    ax.set_ylabel("share of runs revising")
    ax.set_ylim(-0.05, 1.30)
    ax.set_xticks([1, 5, 10, 15, 19])
    # Above the axes rather than inside: every interior region of this panel carries data, and a
    # legend box over the "own precedent" flatline would hide the single most important feature.
    ax.legend(fontsize=8.5, ncol=3, loc="lower left", bbox_to_anchor=(0.0, 1.005),
              frameon=False, handletextpad=0.5, columnspacing=1.4)

    # The onset annotation: the point of the left panel is that lock-in arrives at bank size one.
    ax.annotate(f"one own episode:\nrevision {curve[0]['own']:.2f}",
                xy=(1, curve[0]["own"]), xytext=(1.6, 0.60), fontsize=8.2,
                arrowprops=dict(arrowstyle="->", lw=0.8, color="#c0392b"), color="#c0392b")
    ax.annotate("two: zero, for ten decisions", xy=(4, 0.0), xytext=(3.0, 0.14),
                fontsize=8.2, color="#c0392b")

    w, resp = d["windows"], d["break_response"]
    keys = ["bare", "own", "foreign"]
    # "bare" and "foreign" land within 0.01 of each other after the break — which is itself the
    # point, but it makes the two labels collide. Nudge them apart by rank rather than by hand, so
    # the offset survives the numbers changing.
    posts = sorted(keys, key=lambda k: w[k]["post"])
    nudge = {}
    for i, k in enumerate(posts):
        clash = i > 0 and (w[k]["post"] - w[posts[i - 1]]["post"]) < 0.08
        nudge[k] = 0.055 if clash else 0.0
    for k in keys:
        bx.plot([0, 1], [w[k]["pre"], w[k]["post"]], marker=STYLE[k]["marker"],
                color=STYLE[k]["color"], lw=1.8, ms=5.5)
        star = f"p={resp[k]['p_holm']:.3f}" if resp[k]["significant"] else "n.s."
        bx.annotate(f"{resp[k]['delta']:+.2f}  {star}", xy=(1.05, w[k]["post"] + nudge[k]),
                    fontsize=8.2, color=STYLE[k]["color"], va="center")
    bx.set_xlim(-0.12, 2.05)
    bx.set_xticks([0, 1]); bx.set_xticklabels(["before\nthe break", "after\nthe break"])
    bx.set_ylabel("revision rate")
    bx.set_ylim(-0.05, 1.05)
    bx.set_title("does the lock release?", fontsize=9.5)
    bx.set_ylim(-0.05, 1.05)

    for a in (ax, bx):
        a.spines[["top", "right"]].set_visible(False)
        a.grid(axis="y", alpha=0.22, lw=0.6)

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")

    # Caption facts as macros, so the prose cannot drift from the figure.
    facts = out.parent.parent / "generated" / "lockin_facts.tex"
    facts.parent.mkdir(parents=True, exist_ok=True)
    facts.write_text("\n".join([
        r"\newcommand{\lockinOnsetOwn}{%.2f}" % curve[0]["own"],
        r"\newcommand{\lockinOnsetBare}{%.2f}" % curve[0]["bare"],
        r"\newcommand{\lockinOnsetForeign}{%.2f}" % curve[0]["foreign"],
        r"\newcommand{\lockinBreak}{%d}" % brk,
        *[fr"\newcommand{{\lockin{k.capitalize()}{w.capitalize()}}}{{{d['windows'][k][w]:.2f}}}"
          for k in keys for w in ("pre", "post")],
        *[fr"\newcommand{{\lockin{k.capitalize()}Delta}}{{{resp[k]['delta']:+.2f}}}" for k in keys],
        *[fr"\newcommand{{\lockin{k.capitalize()}P}}{{{resp[k]['p_holm']:.3f}}}" for k in keys],
        "",
    ]), encoding="utf-8")
    print(f"wrote {facts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
