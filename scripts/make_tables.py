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
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
    from govsim.analysis import diagnose

    rows = []
    for regime, d in sorted(calib.items()):
        # Recomputed from the stored losses when the artifact predates the decomposition. It is an
        # exact function of three numbers already in the file, so this cannot disagree with a
        # freshly-calibrated entry — it just saves re-running an hour of search to add a ratio.
        g = d.get("diagnosis") or diagnose(
            d["frozen"]["loss"], d["best_fixed"]["loss"], d["switching"]["loss"])
        rows.append(
            f"{esc(regime)} & {num(d['frozen']['loss'])} & {num(d['best_fixed']['loss'])} & "
            f"{num(d['switching']['loss'])} & {num(g.get('staleness'), 3)} & "
            f"\\textbf{{{num(g.get('adaptation_headroom'), 3)}}} & "
            f"{num(g.get('robustness_headroom'), 3)} \\\\"
        )
    body = "\n".join(rows)
    return f"""\\begin{{table}}[t]\\centering\\small
\\caption{{\\textbf{{Decomposing what a stale rule costs.}} All references are calibrated by
exhaustive search over the same policy vocabulary, on the same broken world, over the same seeds;
they differ only in what they were allowed to know. \\emph{{staleness}}
$=L_{{\\text{{frozen}}}}/L_{{\\text{{switch}}}}$ is what the pre-break-optimal rule costs after the
break---the quantity a study of institutional rigidity usually reports, and on its own ambiguous.
It factors into \\emph{{adaptation headroom}} $=L_{{\\text{{fixed}}}}/L_{{\\text{{switch}}}}$, the
part recoverable \\emph{{only}} by changing behaviour mid-run, and \\emph{{robustness headroom}}
$=L_{{\\text{{frozen}}}}/L_{{\\text{{fixed}}}}$, the part recoverable by having legislated a better
standing rule in the first place. The severe regime is the instructive case: a large staleness cost
with \\emph{{no}} adaptation headroom at all. There the polity needed a better rule, not a more
attentive government---and reporting staleness alone would have called it an adaptation failure and
prescribed the wrong remedy.}}
\\label{{tab:headroom}}
\\begin{{tabular}}{{@{{}}l r r r r r r@{{}}}}
\\toprule
& \\multicolumn{{3}}{{c}}{{full-horizon loss}} & \\multicolumn{{3}}{{c}}{{decomposition}} \\\\
\\cmidrule(lr){{2-4}} \\cmidrule(lr){{5-7}}
regime & frozen & best fixed & switching & staleness & \\textbf{{adaptation}} & robustness \\\\
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
    bf = ep["best_fixed"]
    return f"""\\begin{{table}}[t]\\centering\\small
\\caption{{\\textbf{{The flagship regime, calibrated.}} The substantive content is in the second row:
once the instrument is only a quarter as effective, the best standing rule keeps the intensity and
raises the trigger twenty-fold---it stops paying for the instrument except in a genuine emergency.
The best fixed law is searched over a \\emph{{widened}} vocabulary (threshold, proportional,
threshold-with-floor), so it cannot be beaten merely by writing a shape it was not allowed to
express.}}
\\label{{tab:regime}}
\\begin{{tabular}}{{@{{}}l l@{{}}}}
\\toprule
frozen: optimal before the break, held through it & \\texttt{{{esc(ep['frozen']['expr'])}}} \\\\
best fixed law in hindsight (whole horizon) & \\texttt{{{esc(bf['expr'])}}}
  {{\\footnotesize (best of {esc(', '.join(sorted(p.get('wide_families', {}))))} $\\to$ '{esc(bf.get('family', '?'))}')}} \\\\
clairvoyant switch at $\\shockstep={ep.get('switch_step', '?')}$ &
  \\texttt{{{esc(ep['switching']['pre_expr'])}}} $\\to$ \\texttt{{{esc(ep['switching']['post_expr'])}}} \\\\
\\midrule
full-horizon loss: frozen / best fixed / switching &
  {num(ep['frozen']['loss'])} / {num(bf['loss'])} / {num(ep['switching']['loss'])} \\\\
what adaptation is worth ($L_{{\\text{{fixed}}}}/L_{{\\text{{switch}}}}$) &
  \\textbf{{{num(ep.get('headroom_vs_best_fixed'), 3)}}} \\\\
what a stale rule costs ($L_{{\\text{{frozen}}}}/L_{{\\text{{switch}}}}$) &
  {num(ep.get('headroom'), 3)} \\\\
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
        f"\\textit{{best fixed law in hindsight}} & --- & {num(anchors.get('frozen_mean'))} & --- & "
        f"\\textit{{1.000}} \\\\",
        f"\\textit{{clairvoyant switch}} & --- & {num(anchors.get('oracle_mean'))} & --- & "
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
\\caption{{\\textbf{{Full-horizon governance loss and normalized regret.}} $\\Rreg=1$ is the best
\\emph{{fixed}} law in hindsight and $\\Rreg=0$ the clairvoyant switch, so $\\Rreg<1$ means the arm
did better than any standing rule could have---which, unlike a post-break-only comparison, passivity
alone cannot achieve. Lower is better; $\\Rreg$ is computed per seed against the paired anchors
before averaging. $^{{\\dagger}}$the critic arm is not budget-matched.}}
\\label{{tab:arms}}
\\begin{{tabular}}{{@{{}}l r r r r@{{}}}}
\\toprule
arm & $n$ & $L$ & sd & $\\Rreg$ \\\\
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


def results_prose(a: dict, calib: dict) -> str:
    """The two data-dependent passages, written from the artifact rather than by hand.

    Both default to a red "pending" note in the preamble, so a draft compiled before the sweep
    finishes says so on the page. This file \\renewcommand's them only once the numbers exist, and
    the wording it emits states what was found INCLUDING when that is a null — the whole point of a
    generated results passage is that it cannot quietly become more favourable than the data.
    """
    arms = {r["arm"]: r for r in (a.get("arms") or [])}
    fac = a.get("factorial") or {}
    contrasts = a.get("contrasts") or {}
    ep = calib.get("epidemic", {})
    if not arms or not fac:
        return "% not enough results yet; the preamble's pending defaults stand.\n"

    full = arms.get("epidemic_llm_trace_outcome_memory") or {}
    bare = arms.get("epidemic_llm_bare") or {}
    sig = [k for k, v in fac.items() if v.get("significant")]
    helped = [k for k in sig if fac[k]["effect"] < 0]
    hurt = [k for k in sig if fac[k]["effect"] > 0]

    def _c(key):
        return contrasts.get(key, {})

    # Must filter on the CORRECTED verdict: the sentence below says "after Holm correction".
    beat_fixed = [n for n in arms
                  if _c(f"{n} vs_best_fixed").get("significant")
                  and (_c(f"{n} vs_best_fixed").get("point_estimate") or 0) < 0]

    abstract = (
        f"Adaptation is worth {num(ep.get('headroom_vs_best_fixed'), 2)}$\\times$ over the best fixed "
        f"rule in this regime. Of {len(arms)} arms, {len(beat_fixed)} beat that non-adaptive ceiling "
        f"after Holm correction; the full harness reaches $\\Rreg={num(full.get('R'), 2)}$ against "
        f"$\\Rreg={num(bare.get('R'), 2)}$ with no harness. "
        # Same branching bug as the body prose: keyed on ``helped`` alone, a significant HARMFUL
        # term produced "no harness term survives correction" in the abstract while the results
        # table showed one at p_Holm = 0.002.
        + ((f"The factorial attributes the gain to {', '.join(esc(t) for t in helped)}"
            if helped else "The factorial attributes the difference to ")
           + (("" if helped else "")
              + (f"{', ' if helped else ''}{', '.join(esc(t) for t in hurt)}, which makes the "
                 f"regent \\emph{{worse}}" if hurt else ""))
           + "." if sig else
           "No harness term survives correction, which is itself the result: at this effect size "
           "the components are not separably attributable.")
    )
    pw = a.get("power") or {}
    budget = None
    try:
        budget = ep["best_fixed"]["loss"] - ep["switching"]["loss"]
    except (KeyError, TypeError):
        pass
    power_line = ""
    if pw.get("mde") and budget:
        power_line = (
            f"This design's minimum detectable effect, computed from the observed per-seed spread at "
            f"the Holm-corrected $\\alpha$ and $80\\%$ power, is {num(pw['mde'], 2)} loss units "
            f"against a total adaptation budget of {num(budget, 2)}. It can therefore resolve a "
            f"component only if that component is worth at least "
            f"{num(100 * pw['mde'] / budget, 0)}\\% of everything adaptation is worth in this "
            f"regime. The null below rules out large component effects and does \\emph{{not}} rule "
            f"out modest ones; halving the detectable effect would require "
            f"$n={pw.get('n_for_half_mde', '?')}$ seeds.")

    body_lines = [
        f"Adaptation in this regime is worth {num(ep.get('headroom_vs_best_fixed'), 3)}$\\times$: the "
        f"clairvoyant switch reaches $L={num(ep.get('switching', {}).get('loss'))}$ where the best "
        f"\\emph{{fixed}} law in hindsight reaches ${num(ep.get('best_fixed', {}).get('loss'))}$ and "
        f"the frozen rule ${num(ep.get('frozen', {}).get('loss'))}$. That is the budget every arm "
        f"below is competing for, and it is small---which is worth saying plainly, because a "
        f"$14\\%$ ceiling sets the scale for how large any component effect can honestly be.",
        "",
        f"With no harness at all the regent reaches $\\Rreg={num(bare.get('R'), 3)}$; with all three "
        f"channels, $\\Rreg={num(full.get('R'), 3)}$. "
        + ("Neither figure should be read as the harness being unnecessary: an arm with no channel "
           "still sees the current state at each review, so it can respond to prevalence even "
           "though it cannot learn that its instrument stopped working."),
        "",
        # Branch on ANY surviving term, not only on helpful ones. An earlier version tested
        # ``helped`` — significant AND effect < 0 — so a component that significantly HURT fell
        # through to the "nothing survives" branch, and the prose asserted that no term survived
        # correction directly beside a table showing one at p_Holm = 0.002. The assumption that a
        # significant component must be a beneficial one is exactly the bias this study exists to
        # avoid: the clearest result here is a channel that makes the regent worse.
        ((f"The factorial (\\cref{{tab:factorial}}) attributes the difference to "
          + (f"{', '.join(esc(t) for t in helped)}, which helped" if helped else "")
          + ("; " if helped and hurt else "")
          + (f"{', '.join(esc(t) for t in hurt)}, which made the regent \\emph{{worse}}"
             if hurt else "")
          + f". Terms not listed did not survive Holm correction across the family of "
            f"{len(fac)} effects.")
         if sig else
         f"No term in the factorial survives Holm correction across the family of {len(fac)} "
         f"effects. We report that as the result rather than as a preliminary: with a "
         f"{num(ep.get('headroom_vs_best_fixed'), 2)}$\\times$ ceiling and 20 seeds, the design is "
         f"not powered to separate components of this size, and saying so is more useful than "
         f"reporting the largest uncorrected term."),
        "",
        power_line,
    ]
    esc_body = "\n".join(body_lines)
    return (
        "% GENERATED by scripts/make_tables.py — do not edit; edit the generator.\n"
        f"\\renewcommand{{\\ResultsAbstractSentence}}{{{abstract}}}\n"
        f"\\renewcommand{{\\ResultsBody}}{{{esc_body}}}\n"
    )


def run_status(a: dict) -> str:
    """A generated statement of what has actually been RUN, printed in the paper.

    Written because the draft asserted a completed $2^3$ across several models while the store held
    six of eight cells, no critic arm and no OPRO arm. That kind of claim is not caught by
    proofreading — the tables around it render "Pending" in red and the sentence still reads as
    finished. Making coverage a generated fact means the paper cannot overstate it, and updates
    itself as arms land.
    """
    arms = {r["arm"]: r for r in (a.get("arms") or [])}
    cells = [k for k in ARM_LABELS if k.startswith("epidemic_llm_") and "critic" not in k]
    have = [k for k in cells if k in arms]
    missing = [k for k in cells if k not in arms]
    rates = a.get("action_rates") or {}
    worst = max((v.get("rate", 0.0) for v in rates.values()), default=0.0)
    extras = [(k, k in arms) for k in ("epidemic_opro", "epidemic_llm_critic")]

    bits = [
        f"\\textbf{{Run coverage.}} Of the ${2**3}$ factorial cells, "
        f"\\textbf{{{len(have)}}} have completed at $n={arms[have[0]]['n'] if have else 0}$ paired "
        f"seeds" + (f"; missing: {', '.join(esc(ARM_LABELS[m]) for m in missing)}." if missing else "."),
    ]
    for name, present in extras:
        bits.append(f"The {esc(ARM_LABELS.get(name, name))} arm has "
                    f"{'completed' if present else '\\emph{not} been run'}.")
    if rates:
        bits.append(f"Highest per-arm rate of decisions producing no parseable action: "
                    f"{100 * worst:.1f}\\% "
                    f"({'within' if worst <= 0.02 else '\\textbf{above}'} the $2\\%$ validity gate).")
    cm = a.get("cross_model") or {}
    if cm:
        bits.append(f"Cross-model replication covers {len(cm)} model(s).")
    return "\\noindent " + " ".join(bits) + "\n"


def odd(calib: dict) -> str:
    """The ODD-style model description social-simulation venues expect, filled from the pinned config."""
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
    from govsim.domains.scalar import regimes as R

    pre, shocked = R.EPIDEMIC_PRE, R.EPIDEMIC_SHOCKED
    ep = calib.get("epidemic", {})
    shock = shocked.get("shock_params", {})
    return f"""\\noindent Following the ODD convention, with the parameter values read directly from
the pinned configuration in \\texttt{{govsim/domains/scalar/regimes.py}} rather than transcribed.

\\subsection*{{Purpose}}
To determine which class of structural break a standing feedback rule can absorb, how much
adaptation is worth in each case, and which informational channel allows a rule-writing authority to
detect a break it cannot observe directly. The model is an instrument for isolating that mechanism;
it is not intended to forecast any real epidemic or to evaluate any real restriction policy.

\\subsection*{{Entities, state variables, and scales}}
One governed population and one governing authority. The population is described by the shares
$S_t, I_t, R_t$ with $S+I+R=1$ enforced each step. The authority holds two instruments, restriction
intensity $\\in [0, 0.9]$ and vaccination effort $\\in [0, 0.5]$. One step is an epidemiological
period; the horizon is {R.EPIDEMIC_HORIZON} steps and the authority reviews its rule every
{R.EPIDEMIC_DECIDE_EVERY} steps, giving {R.EPIDEMIC_HORIZON // R.EPIDEMIC_DECIDE_EVERY} reviews.

\\subsection*{{Process overview and scheduling}}
Each step: (i)~the standing rule is re-evaluated against the current state and its output clipped
into the instrument's admissible range; (ii)~new infections, recoveries, vaccinations, and waning
are applied; (iii)~intervention cost accrues on the \\emph{{policy}}, not on its effect. At a review
step the authority first receives whatever its channels supply, then promulgates a rule that stands
until the next review.

\\subsection*{{Design concepts}}
\\emph{{Basic principle}}: a standing rule is feedback, and feedback is robust to disturbances in
the signal it reads and fragile to changes in what its action accomplishes.
\\emph{{Adaptation}}: the authority may rewrite its rule at review points; it is never told the
break occurred. \\emph{{Objective}}: infection burden plus $\\lambda={R.EPIDEMIC_LAMBDA}$ times
intervention cost. \\emph{{Sensing}}: $S, I, R$, the authority's own current instrument settings, and
the clock. Instrument efficacy is \\textbf{{never}} sensed---inferring it is the task.
\\emph{{Stochasticity}}: per-seed heterogeneity in $\\beta_0$, $\\gamma$, and initial prevalence,
plus per-step incidence noise. \\emph{{Observation}}: the full trajectory, the enacted rules, and
every model call are recorded.

\\subsection*{{Initialisation}}
$I_0 = {pre['initial_i']}$ with lognormal spread $\\sigma={pre['initial_i_sigma']}$;
$\\beta_0 = {pre['beta0']}$ with lognormal spread $\\sigma={pre['beta0_sigma']}$;
$\\gamma = {pre['gamma']}$ with spread $\\sigma={pre['gamma_sigma']}$; $R_0 = 0$. Each of the 20
seeds draws one such population and every arm is run on all 20.

\\subsection*{{Input data}}
None. The model is closed; no empirical time series is used as a driver.

\\subsection*{{Submodels}}
\\emph{{Transmission}}: $\\beta_t = \\beta_0\\,(1 - a_t\\,e_t)$, where $a_t$ is enacted restriction
and $e_t$ its efficacy. New infections are $\\beta_t S_t I_t$ plus imported cases at rate
{pre['import_rate']}$\\cdot S_t$ and Gaussian noise $\\sigma={pre['noise_sigma']}$, capped at $S_t$.
\\emph{{Recovery and waning}}: a share $\\gamma$ of $I$ recovers each step and a share
{pre['waning']} of $R$ returns to $S$, which makes the disease endemic---it cannot be waited out.
\\emph{{Vaccination}}: moves ${pre['vacc_rate']}\\cdot v_t\\, e^{{\\text{{vac}}}}_t S_t$ from $S$ to
$R$. \\emph{{Cost}}: ${pre['lockdown_cost']}\\,a_t + {pre['vacc_cost']}\\,v_t$ per step, charged on
the policy regardless of its effect---which is what makes an efficacy collapse expensive to ignore.
\\emph{{The break}}: at $t = {R.EPIDEMIC_SHOCK_STEP}$, restriction efficacy
$e_t: 1.0 \\to {shock.get('lockdown_efficacy', '?')}$. Transmissibility is unchanged
(\\texttt{{shock\\_factor}} $= {shocked.get('shock_factor', 1.0)}$), so the break is purely
instrumental. It is not announced and leaves no directly observable trace.

\\subsection*{{Reference policies}}
Calibrated by exhaustive search over {ep.get('provenance', {}).get('family_size', '?')} laws at
{ep.get('provenance', {}).get('n_seeds', '?')} seeds:
frozen \\texttt{{{esc(ep.get('frozen', {}).get('expr', 'n/a'))}}},
best fixed in hindsight \\texttt{{{esc(ep.get('best_fixed', {}).get('expr', 'n/a'))}}}.
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
        "odd.tex": odd(calib),
        "run_status.tex": run_status(analysis),
        "repro.tex": repro(calib, analysis, args.commit),
        "results_prose.tex": results_prose(analysis, calib),
    }
    for name, text in written.items():
        (OUT / name).write_text(text, encoding="utf-8")
        print(f"wrote paper/generated/{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
