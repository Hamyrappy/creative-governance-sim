"""
Render the paper's figures. Key-free: everything here is calibration and replay, no live model calls.

    uv run python scripts/make_figures.py --surface      # recompute the headroom surface (slow)
    uv run python scripts/make_figures.py                # render from the cached artifact

Figures:

1. ``headroom_surface`` — headroom over (cost weight lambda) x (post-shock instrument efficacy),
   with the operating point marked. This is the argument of the paper in one panel: the usable band
   is narrow, it is bounded on BOTH sides, and where it vanishes it vanishes for a reason that can
   be named (a corner optimum is regime-invariant).

2. ``policy_divergence`` — prevalence and enacted lockdown over time for the frozen institution, the
   clairvoyant oracle, and the harnessed regent. The point is visual: after the break the frozen rule
   keeps buying an instrument that stopped working, and the two adaptive traces separate from it.

3. ``regret_by_arm`` — normalized regret per harness configuration with bootstrap intervals.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from govsim.analysis import PolicyFamily, calibrate, headroom, normalized_regret  # noqa: E402
from govsim.analysis.calibration import _score_expr  # noqa: E402
from govsim.core.schedule import EveryN  # noqa: E402
from govsim.domains.scalar import EpidemicLoss, Lever, ScalarLeverInterface, SIRSystem  # noqa: E402
from govsim.domains.scalar import regimes as R  # noqa: E402

FIG = Path(__file__).resolve().parent.parent / "paper" / "generated"
ART = Path(__file__).resolve().parent.parent / "logs"

# A restrained, print-safe palette; distinguishable in greyscale by line style as well as hue.
C_FROZEN, C_ORACLE, C_REGENT = "#B03A2E", "#1F4E79", "#1E8449"
plt.rcParams.update({
    "font.size": 9, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 160,
})

IFACE = ScalarLeverInterface([
    Lever("set_lockdown", (0.0, 0.9), "lockdown"),
    Lever("set_vaccination", (0.0, 0.5), "vacc"),
])
FAMILY = PolicyFamily(
    verb="set_lockdown", template="{a} if I > {thr} else 0.0",
    grid={"a": [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9],
          "thr": [0.002, 0.01, 0.03, 0.06, 0.10, 0.16, 0.25, 0.40, 0.70]},
)


# ---------------------------------------------------------------------------------------------
# 1. headroom surface
# ---------------------------------------------------------------------------------------------

def compute_surface(lams, effs, n_seeds: int = 8) -> list[dict]:
    seeds = list(range(n_seeds))
    rows = []
    for lam in lams:
        for eff in effs:
            base = dict(R.EPIDEMIC_PRE)
            shocked = dict(base, shock_step=R.EPIDEMIC_SHOCK_STEP, shock_factor=1.0,
                           shock_params={"lockdown_efficacy": eff})
            obj = EpidemicLoss(lam=lam, post_shock_step=R.EPIDEMIC_SHOCK_STEP)
            kw = dict(action_interface=IFACE, objective=obj, schedule=EveryN(R.EPIDEMIC_DECIDE_EVERY),
                      seeds=seeds, horizon=R.EPIDEMIC_HORIZON)
            fz = calibrate(FAMILY, system_factory=R.sir_factory(base), metric="loss", **kw)
            orc = calibrate(FAMILY, system_factory=R.sir_factory(shocked), metric="post_loss", **kw)
            skw = dict(verb=FAMILY.verb, system_factory=R.sir_factory(shocked),
                       action_interface=IFACE, objective=obj,
                       schedule=EveryN(R.EPIDEMIC_DECIDE_EVERY), seeds=seeds,
                       horizon=R.EPIDEMIC_HORIZON, metric="post_loss")
            fl, ol = _score_expr(fz.best_expr, **skw), _score_expr(orc.best_expr, **skw)
            rows.append({"lam": lam, "eff": eff, "headroom": headroom(fl, ol),
                         "frozen": fz.best_expr, "oracle": orc.best_expr,
                         "frozen_loss": fl, "oracle_loss": ol,
                         "same_policy": fz.best_expr == orc.best_expr})
            print(f"  lam={lam:<5} eff={eff:<5} headroom={rows[-1]['headroom']:.3f}"
                  f"{'  [corner: shock does not move the optimum]' if rows[-1]['same_policy'] else ''}",
                  flush=True)
    return rows


def plot_surface(rows: list[dict]) -> Path:
    lams = sorted({r["lam"] for r in rows})
    effs = sorted({r["eff"] for r in rows}, reverse=True)
    grid = np.full((len(effs), len(lams)), np.nan)
    for r in rows:
        grid[effs.index(r["eff"]), lams.index(r["lam"])] = r["headroom"]

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    im = ax.imshow(grid, aspect="auto", cmap="YlGnBu", vmin=1.0,
                   vmax=max(1.05, float(np.nanmax(grid))))
    ax.set_xticks(range(len(lams)), [f"{v:g}" for v in lams])
    ax.set_yticks(range(len(effs)), [f"{v:g}" for v in effs])
    ax.set_xlabel(r"cost weight $\lambda$  (price of intervention vs infection)")
    ax.set_ylabel("post-shock instrument efficacy")
    ax.grid(False)
    for i, eff in enumerate(effs):
        for j, lam in enumerate(lams):
            v = grid[i, j]
            if np.isnan(v):
                continue
            same = next(r["same_policy"] for r in rows if r["lam"] == lam and r["eff"] == eff)
            ax.text(j, i, ("=" if same else f"{v:.2f}"), ha="center", va="center",
                    fontsize=7.5, color="white" if v > 1.45 else "#22333B")
    # Mark the pre-registered operating point.
    if R.EPIDEMIC_LAMBDA in lams and 0.25 in effs:
        ax.add_patch(plt.Rectangle((lams.index(R.EPIDEMIC_LAMBDA) - 0.5, effs.index(0.25) - 0.5),
                                   1, 1, fill=False, edgecolor="#B03A2E", lw=2.0))
    fig.colorbar(im, ax=ax, label=r"headroom $L^{\rm post}_{\rm frozen}/L^{\rm post}_{\rm oracle}$")
    ax.set_title("Adaptation headroom is available only in a narrow band", fontsize=9.5, pad=8)
    out = FIG / "fig_headroom_surface.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------------------------
# 2. policy divergence
# ---------------------------------------------------------------------------------------------

def _run_expr(expr: str, seed: int) -> list[dict]:
    """Replay a fixed policy on the shocked world and return its trajectory."""
    from govsim.core.experiment import Experiment, Hypothesis
    from govsim.core.regent import ScriptedRegent
    from govsim.core.runner import Runner

    exp = Experiment(
        name="traj", system_factory=R.sir_factory(R.EPIDEMIC_SHOCKED), action_interface=IFACE,
        regents={"regent:0": ScriptedRegent(verb="set_lockdown", expr=expr)},
        objectives={"regent:0": EpidemicLoss(lam=R.EPIDEMIC_LAMBDA,
                                             post_shock_step=R.EPIDEMIC_SHOCK_STEP)},
        schedule=EveryN(R.EPIDEMIC_DECIDE_EVERY), seeds=[seed], horizon=R.EPIDEMIC_HORIZON,
        hypothesis=Hypothesis(id="traj", claim="trajectory rendering", baseline="n/a",
                              primary_metric="post_loss"),
    )
    return Runner().run(exp)[0].metrics_series


def plot_divergence(regent_expr: str | None, seed: int = 0) -> Path:
    traces = [("frozen institution", R.reference_expr("epidemic", "frozen"), C_FROZEN, "-"),
              ("clairvoyant oracle", R.reference_expr("epidemic", "oracle"), C_ORACLE, "--")]
    if regent_expr:
        traces.append((f"regent: {regent_expr}", regent_expr, C_REGENT, "-."))

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5.6, 5.4), sharex=True,
                                        gridspec_kw={"height_ratios": [1.2, 1, 1]})
    for label, expr, color, ls in traces:
        rows = _run_expr(expr, seed)
        t = [r["t"] for r in rows]
        ax1.plot(t, [r["infected"] for r in rows], color=color, ls=ls, lw=1.4, label=label)
        ax2.plot(t, [r.get("lockdown", 0.0) for r in rows], color=color, ls=ls, lw=1.4)
        ax3.plot(t, [r.get("cum_cost", 0.0) for r in rows], color=color, ls=ls, lw=1.4)

    for ax in (ax1, ax2, ax3):
        ax.axvline(R.EPIDEMIC_SHOCK_STEP, color="#666", lw=0.9, ls=":")
    ax1.set_ylim(top=ax1.get_ylim()[1] * 1.45)  # headroom so the legend never sits on a curve
    ax1.annotate("instrument efficacy collapses\n(unannounced, unobservable)",
                 xy=(R.EPIDEMIC_SHOCK_STEP + 4, ax1.get_ylim()[1] * 0.97),
                 fontsize=7.5, color="#444", va="top")
    ax1.set_ylabel("prevalence $I_t$")
    ax2.set_ylabel("enacted lockdown")
    ax3.set_ylabel("cumulative cost")
    ax3.set_xlabel("step")
    ax1.legend(fontsize=7.5, frameon=False, loc="upper left", ncol=1)
    ax1.set_title("After the break the frozen rule pays more and buys less", fontsize=9.5, pad=6)
    out = FIG / "fig_policy_divergence.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------------------------
# 3. regret by arm
# ---------------------------------------------------------------------------------------------

def plot_regret(analysis: dict) -> Path | None:
    arms = analysis.get("arms") or []
    if not arms:
        return None
    labels = {"epidemic_llm_bare": "none", "epidemic_llm_trace": "T", "epidemic_llm_outcome": "O",
              "epidemic_llm_memory": "M", "epidemic_llm_trace_outcome": "T+O",
              "epidemic_llm_trace_memory": "T+M", "epidemic_llm_outcome_memory": "O+M",
              "epidemic_llm_trace_outcome_memory": "T+O+M", "epidemic_opro": "OPRO",
              "epidemic_llm_critic": "T+O+M+C"}
    order = [a for a in labels if any(r["arm"] == a for r in arms)]
    vals = {r["arm"]: r for r in arms}

    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    xs = range(len(order))
    ys = [vals[a].get("R") for a in order]
    cols = ["#8E9AAF" if "opro" in a else ("#1E8449" if vals[a].get("R", 9) < 0.5 else "#1F4E79")
            for a in order]
    ax.bar(xs, [y if y is not None else 0 for y in ys], color=cols, width=0.62)
    ax.axhline(1.0, color=C_FROZEN, lw=1.1, ls="--")
    ax.axhline(0.0, color=C_ORACLE, lw=1.1, ls="--")
    ax.text(len(order) - 0.4, 1.0, " frozen", color=C_FROZEN, fontsize=7.5, va="center")
    ax.text(len(order) - 0.4, 0.0, " oracle", color=C_ORACLE, fontsize=7.5, va="center")
    ax.set_xticks(list(xs), [labels[a] for a in order], fontsize=8)
    ax.set_ylabel(r"normalized regret $\mathcal{R}$")
    ax.set_xlabel("harness configuration  (T=trace, O=outcome, M=memory, C=critic)")
    ax.set_title("How much of the available headroom each harness recovers", fontsize=9.5, pad=6)
    out = FIG / "fig_regret_by_arm.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--surface", action="store_true", help="recompute the headroom surface (slow)")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--analysis", default="logs/analysis.json")
    ap.add_argument("--regent-expr", default=None,
                    help="an enacted regent policy to overlay on the divergence figure")
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    surface_path = ART / "headroom_surface.json"

    if args.surface or not surface_path.exists():
        print("computing the headroom surface (key-free, but a full calibration per cell)…")
        rows = compute_surface([0.02, 0.04, 0.08, 0.15, 0.30], [0.75, 0.5, 0.25, 0.0], args.seeds)
        surface_path.parent.mkdir(parents=True, exist_ok=True)
        surface_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    rows = json.loads(surface_path.read_text(encoding="utf-8"))
    print("wrote", plot_surface(rows))
    print("wrote", plot_divergence(args.regent_expr))

    analysis = json.loads(Path(args.analysis).read_text(encoding="utf-8")) \
        if Path(args.analysis).exists() else {}
    p = plot_regret(analysis)
    print("wrote", p) if p else print("(regret figure skipped: no analysis artifact yet)")

    usable = [r for r in rows if not r["same_policy"] and r["headroom"] >= 1.5]
    print(f"\nsurface: {len(usable)}/{len(rows)} cells with headroom >= 1.5x and a moved optimum")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
