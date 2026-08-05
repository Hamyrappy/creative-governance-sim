"""Figure: episodic memory suppresses policy revision, and outcome feedback restores it.

Two panels sharing an x-axis of the eight harness cells, ordered by churn so the collapse is the
shape of the figure rather than something the caption has to assert:

  (a) policy churn — the share of reviews at which the enacted law changed, with per-seed spread;
  (b) full-horizon loss on the same cells, against the three calibrated references.

The point the figure has to make is the DISSOCIATION between them, so both panels are drawn on the
same ordering and the reference lines in (b) let a reader see that even the best arm sits above the
non-adaptive ceiling. It deliberately does not draw a regression line through churn against loss:
the two-way fixed-effects correlation is -0.11, and a fitted line would assert a within-condition
relationship the data does not support (see ``scripts/churn_loss_panel.py``).

Run: ``uv run python scripts/fig_lockin.py``
"""

from __future__ import annotations

import argparse
import itertools
import statistics as st
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from govsim.core.result_store import ResultStore  # noqa: E402
from policy_churn import arm_name, churn_by_seed  # noqa: E402

C_MEM, C_OUT, C_PLAIN = "#B03A2E", "#1F4E79", "#7F8C8D"
plt.rcParams.update({
    "font.size": 9, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 160,
})

SHORT = {
    "bare": "none", "trace": "T", "outcome": "O", "memory": "M",
    "trace_outcome": "T+O", "trace_memory": "T+M", "outcome_memory": "O+M",
    "trace_outcome_memory": "T+O+M",
}


def _label(cell: tuple[bool, ...]) -> str:
    on = [f for f, b in zip(("trace", "outcome", "memory"), cell) if b]
    return SHORT["_".join(on) if on else "bare"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="logs/runs_v3")
    ap.add_argument("--out", default="paper/generated/fig_lockin.pdf")
    args = ap.parse_args()

    store = ResultStore(args.store)
    rows = []
    for cell in itertools.product((False, True), repeat=3):
        name = arm_name("epidemic", cell)
        churn = churn_by_seed(store, name, carry_forward=True)
        loss = {}
        for r in store.query(experiment=name):
            c = (r.get("components") or {}).get("regent:0", {})
            if "loss" in c:
                loss[int(r["seed"])] = float(c["loss"])
        shared = sorted(set(churn) & set(loss))
        if len(shared) < 5:
            continue
        rows.append({
            "label": _label(cell), "memory": cell[2], "outcome": cell[1],
            "churn": [churn[k] for k in shared], "loss": [loss[k] for k in shared],
        })
    if len(rows) < 8:
        print(f"only {len(rows)}/8 cells present; not drawing")
        return 1

    rows.sort(key=lambda r: -st.fmean(r["churn"]))
    x = range(len(rows))
    # Colour by which channel the cell carries: the figure's claim is about memory, so memory-
    # bearing cells must be identifiable without reading the tick labels.
    colours = [C_MEM if r["memory"] else (C_OUT if r["outcome"] else C_PLAIN) for r in rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.8, 4.6), sharex=True,
                                   gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.18})

    ax1.bar(x, [st.fmean(r["churn"]) for r in rows], color=colours, width=0.62, zorder=2)
    for i, r in enumerate(rows):  # per-seed spread, so the bars are not read as point estimates
        ax1.plot([i] * len(r["churn"]), r["churn"], ".", color="black", ms=2.0, alpha=0.35, zorder=3)
    ax1.set_ylabel("policy churn")
    ax1.set_ylim(0, 1.0)
    ax1.set_title("Episodic memory stops the institution revising", fontsize=9.5, loc="left")

    ax2.bar(x, [st.fmean(r["loss"]) for r in rows], color=colours, width=0.62, zorder=2)
    for i, r in enumerate(rows):
        ax2.plot([i] * len(r["loss"]), r["loss"], ".", color="black", ms=2.0, alpha=0.35, zorder=3)
    ax2.set_ylabel("full-horizon loss")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels([r["label"] for r in rows])
    ax2.set_xlabel("harness cell   (T = failure trace, O = outcome feedback, M = episodic memory)")

    anchors = {}
    for name, style in (("epidemic_switching", (0, (4, 2))), ("epidemic_best_fixed", (0, (1, 1.5))),
                        ("epidemic_frozen", "solid")):
        vals = [(r.get("components") or {}).get("regent:0", {}).get("loss")
                for r in store.query(experiment=name)]
        vals = [v for v in vals if v is not None]
        if vals:
            anchors[name] = st.fmean(vals)
            ax2.axhline(anchors[name], color="#444444", lw=0.8, ls=style, zorder=1)
    ymin = min(min(r["loss"]) for r in rows)
    ax2.set_ylim(min(ymin, min(anchors.values(), default=ymin)) - 1.0, None)
    for name, text in (("epidemic_switching", "clairvoyant"), ("epidemic_best_fixed", "best fixed"),
                       ("epidemic_frozen", "frozen")):
        if name in anchors:
            ax2.annotate(text, xy=(len(rows) - 0.45, anchors[name]), fontsize=7,
                         color="#444444", va="bottom", ha="right")

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (C_MEM, C_OUT, C_PLAIN)]
    ax1.legend(handles, ["carries memory", "carries outcome (no memory)", "neither"],
               fontsize=7, frameon=False, loc="lower left", ncol=1)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
    for r in rows:
        print(f"  {r['label']:<8} churn={st.fmean(r['churn']):.3f}  loss={st.fmean(r['loss']):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
