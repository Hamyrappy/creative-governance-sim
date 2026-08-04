"""
Generate the paper's LaTeX tables from the analysis artifacts.

    uv run python scripts/analyze_matrix.py --store logs/runs logs/runs_* \
        --cross-model --json logs/analysis.json
    uv run python scripts/make_tables.py --analysis logs/analysis.json

No number in the paper is typed by hand: every table here is derived from a committed JSON artifact,
so a table and the run that produced it cannot drift apart. Tables for data that does not exist yet
render as an explicit "pending" note rather than an empty tabular, so a draft compiled mid-sweep
says so on the page instead of looking finished.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "paper" / "generated"

ARM_LABELS = {
    "epidemic_llm_bare": "no harness",
    "epidemic_llm_trace": "trace",
    "epidemic_llm_outcome": "outcome",
    "epidemic_llm_memory": "memory",
    "epidemic_llm_trace_outcome": "trace + outcome",
    "epidemic_llm_trace_memory": "trace + memory",
    "epidemic_llm_outcome_memory": "outcome + memory",
    "epidemic_llm_trace_outcome_memory": "all three",
    "epidemic_opro": "OPRO (budget-matched rival)",
    "epidemic_llm_critic": "all three + critic\\,$^{\\dagger}$",
}


def esc(s: str) -> str:
    return str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def num(v, p: int = 3, dash: str = "---") -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return dash
    return f"{v:.{p}f}"


def pending(name: str, caption: str, label: str) -> str:
    return (
        "\\begin{table}[t]\\centering\\small\n"
        f"\\caption{{{caption}}}\\label{{{label}}}\n"
        # NB: inside an f-string, `}}}}` renders as `}}` — the two braces that close \textcolor
        # and \emph. Writing `}}` here emits only one and leaves the float unterminated.
        "\\emph{\\textcolor{red}{Pending: the analysis artifact does not yet contain this table "
        f"({esc(name)}).}}}}\n"
        "\\end{table}\n"
    )


# ---------------------------------------------------------------------------------------------


def headroom_table(calib: dict, surface: list[dict] | None) -> str:
    rows = []
    for regime, d in sorted(calib.items()):
        rows.append(
            f"{esc(regime)} & \\texttt{{{esc(d['frozen']['expr'])}}} & "
            f"\\texttt{{{esc(d['oracle']['expr'])}}} & "
            f"{num(d['frozen']['post_loss'])} & {num(d['oracle']['post_loss'])} & "
            f"\\textbf{{{num(d['headroom'], 2)}}} \\\\"
        )
    body = "\n".join(rows)
    return f"""\\begin{{table}}[t]\\centering\\small
\\caption{{\\textbf{{Adaptation headroom by regime.}} Both references are calibrated by exhaustive
search inside one shared policy family and scored on the same broken world over the same seeds; they
differ only in when they were allowed to look. $\\headroom \\approx 1$ means the break does not move
the optimum, so no controller could show an effect there.}}
\\label{{tab:headroom}}
\\begin{{tabular}}{{@{{}}l l l r r r@{{}}}}
\\toprule
regime & frozen (pre-shock optimal) & oracle (post-shock optimal) &
$\\Lpost_{{\\text{{frozen}}}}$ & $\\Lpost_{{\\text{{oracle}}}}$ & $\\headroom$ \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def regime_table(calib: dict) -> str:
    ep = calib.get("epidemic")
    if not ep:
        return pending("epidemic calibration", "The flagship regime.", "tab:regime")
    p = ep["provenance"]
    wide = ep.get("oracle_wide", {})
    return f"""\\begin{{table}}[t]\\centering\\small
\\caption{{\\textbf{{The flagship regime, calibrated.}} The oracle's answer is the substantive one:
it keeps the intensity and raises the trigger, i.e.\\ it stops paying for an instrument that no
longer works except in a genuine emergency. Widening the reference vocabulary to include
proportional and floored-threshold forms does not change it.}}
\\label{{tab:regime}}
\\begin{{tabular}}{{@{{}}l l@{{}}}}
\\toprule
frozen (optimal before the break) & \\texttt{{{esc(ep['frozen']['expr'])}}} \\\\
oracle (optimal after it, narrow family) & \\texttt{{{esc(ep['oracle']['expr'])}}} \\\\
oracle (widened vocabulary) & \\texttt{{{esc(wide.get('expr', 'n/a'))}}}
  {{\\footnotesize (best of {esc(', '.join(sorted(p.get('wide_families', {}))))})}} \\\\
\\midrule
post-break loss, frozen / oracle & {num(ep['frozen']['post_loss'])} / {num(ep['oracle']['post_loss'])} \\\\
headroom $\\headroom$ (narrow / wide) & \\textbf{{{num(ep['headroom'], 2)}}} /
  {num(ep.get('headroom_wide'), 2)} \\\\
calibration grid & {p['family_size']} laws, {p['n_seeds']} seeds, horizon {p['horizon']} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def arms_table(a: dict) -> str:
    arms = a.get("arms") or []
    if not arms:
        return pending("arms", "Arm results.", "tab:arms")
    anchors = a.get("anchors", {})
    order = list(ARM_LABELS)
    arms = sorted(arms, key=lambda r: order.index(r["arm"]) if r["arm"] in order else 99)
    rows = [
        f"\\textit{{frozen}} (R\\,{{=}}\\,1 anchor) & --- & {len(arms) and ''}"
        f"{num(anchors.get('frozen_mean'))} & --- & \\textit{{1.000}} \\\\",
        f"\\textit{{oracle}} (R\\,{{=}}\\,0 anchor) & --- & {num(anchors.get('oracle_mean'))} & --- & "
        f"\\textit{{0.000}} \\\\",
        "\\midrule",
    ]
    for r in arms:
        label = ARM_LABELS.get(r["arm"], esc(r["arm"]))
        rows.append(
            f"{label} & {r['n']} & {num(r['mean'])} & {num(r['sd'])} & "
            f"\\textbf{{{num(r.get('R'))}}} \\\\"
        )
    body = "\n".join(rows)
    return f"""\\begin{{table}}[t]\\centering\\small
\\caption{{\\textbf{{Post-break governance loss and normalized regret.}} $\\Rreg=0$ is the
clairvoyant oracle, $\\Rreg=1$ is the pre-break rule held unchanged. Lower is better; $\\Rreg$ is
computed per seed against the paired anchors before averaging.
$^{{\\dagger}}$the critic arm is not budget-matched.}}
\\label{{tab:arms}}
\\begin{{tabular}}{{@{{}}l r r r r@{{}}}}
\\toprule
arm & $n$ & $\\Lpost$ & sd & $\\Rreg$ \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def factorial_table(a: dict) -> str:
    f = a.get("factorial")
    if not f:
        return pending("factorial", "Harness component attribution.", "tab:factorial")
    rows = []
    for k, v in sorted(f.items(), key=lambda kv: (kv[1]["order"], kv[1]["p"])):
        sig = r"\textbf{yes}" if v.get("significant") else "no"
        rows.append(
            f"{esc(k)} & {v['order']} & {v['effect']:+.3f} & "
            f"$[{v['ci_low']:+.3f}, {v['ci_high']:+.3f}]$ & {num(v['p'], 4)} & "
            f"{num(v.get('p_adj'), 4)} & {sig} \\\\"
        )
    body = "\n".join(rows)
    return f"""\\begin{{table}}[t]\\centering\\small
\\caption{{\\textbf{{Factorial attribution of the harness.}} $\\pm1$ contrast coding with contrasts
formed within each seed, so world variance cancels. A \\emph{{negative}} effect lowers post-break
loss, i.e.\\ the component helped. Interactions are estimated, not assumed to be zero; $p$ is a
two-sided bootstrap value and $p_{{\\text{{Holm}}}}$ corrects across the whole family of seven
terms.}}
\\label{{tab:factorial}}
\\begin{{tabular}}{{@{{}}l c r c r r c@{{}}}}
\\toprule
term & order & effect & 95\\% CI & $p$ & $p_{{\\text{{Holm}}}}$ & significant \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def crossmodel_table(a: dict) -> str:
    cm = a.get("cross_model")
    if not cm:
        return pending("cross_model", "Cross-model replication.", "tab:crossmodel")
    rungs = ["epidemic_llm_bare", "epidemic_llm_outcome", "epidemic_llm_trace_outcome_memory"]
    heads = ["no harness", "outcome only", "all three"]
    rows = []
    for model, r in sorted(cm.items()):
        cells = []
        for arm in rungs:
            d = r.get(arm)
            cells.append(num(d["R"]) if isinstance(d, dict) and d.get("R") is not None else "---")
        rows.append(f"\\texttt{{{esc(model)}}} & " + " & ".join(cells) + " \\\\")
    body = "\n".join(rows)
    cols = " ".join("r" for _ in heads)
    return f"""\\begin{{table}}[t]\\centering\\small
\\caption{{\\textbf{{Cross-model replication.}} Mean normalized regret $\\Rreg$ on the same paired
seeds and the same anchors. The question is whether the harness ordering replicates outside a single
model, not whether the models are equally strong.}}
\\label{{tab:crossmodel}}
\\begin{{tabular}}{{@{{}}l {cols}@{{}}}}
\\toprule
model & {" & ".join(heads)} \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def repro(calib: dict, a: dict, commit: str) -> str:
    ep = calib.get("epidemic", {})
    p = ep.get("provenance", {})
    return f"""\\noindent Everything below is in the repository at commit \\texttt{{{esc(commit)}}}.

\\begin{{description}}[leftmargin=0em,style=nextline]
\\item[Calibrate the anchors] \\texttt{{uv run python scripts/recalibrate.py --seeds {p.get('n_seeds', 20)}}}
  \\\\writes \\texttt{{govsim/docs\\_gates/calibration.json}} (the artifact the paper's anchors are read from).
\\item[Measure headroom across regimes] \\texttt{{uv run python scripts/headroom\\_audit.py --seeds 8}}
\\item[Run the matrix] \\texttt{{uv run python scripts/run\\_matrix.py --arms epidemic --seeds 20}}
  \\\\resumable: the model-call tape makes an interrupted sweep free to restart.
\\item[Analyse] \\texttt{{uv run python scripts/analyze\\_matrix.py --store logs/runs --cross-model --json logs/analysis.json}}
\\item[Regenerate these tables] \\texttt{{uv run python scripts/make\\_tables.py --analysis logs/analysis.json}}
\\item[Replay without an API key] set \\texttt{{GOVSIM\\_LLM\\_MODE=replay}}. Every reported run is
  served from the committed tape; a wrong key changes nothing.
\\item[Tests] \\texttt{{uv run pytest}} --- key-free, including the factorial estimator checked
  against data with a known generating effect.
\\end{{description}}
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analysis", default="logs/analysis.json")
    ap.add_argument("--calibration", default="govsim/docs_gates/calibration.json")
    ap.add_argument("--commit", default="see git log")
    args = ap.parse_args()

    calib = json.loads(Path(args.calibration).read_text(encoding="utf-8")) \
        if Path(args.calibration).exists() else {}
    analysis = json.loads(Path(args.analysis).read_text(encoding="utf-8")) \
        if Path(args.analysis).exists() else {}

    OUT.mkdir(parents=True, exist_ok=True)
    written = {
        "headroom_table.tex": headroom_table(calib, None) if calib else pending(
            "calibration.json", "Adaptation headroom by regime.", "tab:headroom"),
        "regime_table.tex": regime_table(calib),
        "arms_table.tex": arms_table(analysis),
        "factorial_table.tex": factorial_table(analysis),
        "crossmodel_table.tex": crossmodel_table(analysis),
        "repro.tex": repro(calib, analysis, args.commit),
    }
    for name, text in written.items():
        (OUT / name).write_text(text, encoding="utf-8")
        print(f"wrote paper/generated/{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
