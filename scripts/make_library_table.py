"""Emit ``paper/generated/library_table.tex`` from ``logs/decomposition.json``.

The headroom table in the paper decomposes four regimes of one flagship. This one decomposes ten
worlds with unrelated mechanics — a monetary economy, an opinion polity, a fishery, a supply chain,
a stock-flow-consistent fiscal economy, and five configurations of an epidemic — under the same
shock taxonomy, the same reference construction and the same seed set.

That is the difference between "adaptation headroom is scarce in this epidemic" and "adaptation
headroom is scarce", which is the paper's actual claim. Generated rather than hand-written so the
table cannot drift from the artifact it reports.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

#: Human-readable names. A LaTeX table that says ``SIR efficacy-collapse (lam=0.08, eff 1.0->0.25)``
#: is a log line, not a table.
PRETTY = {
    "monetary": ("monetary", "policy rate", "transmission collapse"),
    "opinion": ("opinion", "moderation", "moderation collapse"),
    "commons": ("commons", "quota + reserve", "enforcement collapse"),
    "supply_chain": ("supply chain", "order quantity", "lead time lengthens"),
    "fiscal": ("fiscal", "tax + transfer", "compliance collapse"),
}


def _label(name: str) -> tuple[str, str, str]:
    for key, val in PRETTY.items():
        if name.startswith(key):
            return val
    if "SIR efficacy-collapse" in name:
        inner = name.split("(", 1)[1].rstrip(")")
        return (f"epidemic ({inner})".replace("lam=", r"$\lambda$="). replace("->", r"$\to$"),
                "lockdown + vaccination", "instrument efficacy")
    if "SIR" in name:
        return ("epidemic (state shock)", "lockdown + vaccination", "transmissibility")
    if "cubic" in name:
        return ("cubic plant", "control input", "plant dynamics")
    if "coupled" in name:
        return ("coupled plant", "control input", "plant dynamics")
    return (name.replace("_", r"\_"), "--", "--")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="logs/decomposition.json")
    ap.add_argument("--out", default="paper/generated/library_table.tex")
    args = ap.parse_args()

    rows = json.loads(Path(args.json).read_text(encoding="utf-8"))
    valid = [r for r in rows if r.get("ratios_valid")]
    invalid = [r for r in rows if not r.get("ratios_valid")]
    valid.sort(key=lambda r: r["adaptation_headroom"], reverse=True)

    # WE DO NOT REPORT A FRACTION-OF-WORLDS STATISTIC, and this block is why.
    #
    # The paper used to say "only 3 of 10 worlds clear 1.1x adaptation headroom". Five of those ten
    # rows are the SAME SIR model at different (lambda, efficacy) settings, three of which return
    # exactly 1.000 because the break provably cannot move the optimum there. So the denominator was
    # padded with near-nulls from one world, in the direction of the paper's own thesis — the
    # direction that most needs guarding against.
    #
    # De-padding does not fix it, it exposes it. The fraction is 3/9 = 30% over rows, 3/5 = 60% over
    # distinct worlds taking each world's best configuration, and 2/5 = 40% taking each world's
    # median. A statistic that ranges 30-60% under three defensible conventions is not a result; it
    # is an artifact of a library WE chose the contents of. There is no sampling frame over "worlds",
    # so no fraction computed here estimates anything.
    #
    # Two quantities survive every convention, and we report those instead:
    #   (1) the MAXIMUM adaptation headroom anywhere in the library — a bound, not a frequency;
    #   (2) the epidemic panel's WITHIN-WORLD span, which shows headroom is a property of the
    #       (world, shock, price) triple rather than of the world, and therefore that "how many
    #       worlds have headroom" is not a well-posed question in the first place.
    def _world_of(name: str) -> str:
        if name.startswith("SIR"):
            return "epidemic"
        return name.split(" (", 1)[0]

    by_world: dict[str, list[float]] = {}
    for r in valid:
        by_world.setdefault(_world_of(r["name"]), []).append(r["adaptation_headroom"])
    n_defined = len(by_world)
    n_total = n_defined + len({_world_of(r["name"]) for r in invalid})

    adapt_all = [r["adaptation_headroom"] for r in valid]
    max_adapt = max(adapt_all)
    max_world = _label(max(valid, key=lambda r: r["adaptation_headroom"])["name"])[0]
    # The three counting conventions, all reported, precisely so no single one can be quoted as if
    # it were the finding.
    frac_rows = (sum(1 for v in adapt_all if v >= 1.1), len(adapt_all))
    frac_best = (sum(1 for v in by_world.values() if max(v) >= 1.1), n_defined)
    frac_med = (sum(1 for v in by_world.values() if median(v) >= 1.1), n_defined)
    epi = sorted(by_world["epidemic"])

    lines = [
        r"\begin{table}[t]\centering\small",
        r"\caption{\textbf{Adaptation headroom across the world library.} Ten worlds with unrelated "
        r"mechanics, each decomposed by the same construction as \cref{tab:headroom}: every "
        r"reference is calibrated by exhaustive search over that world's own policy vocabulary, on "
        r"the same broken world, over the same seeds, and the three differ only in what they were "
        r"allowed to know. Ranked by \emph{adaptation} headroom, because that is the only factor an "
        rf"adaptive agent can claim. \textbf{{We deliberately report no fraction-of-worlds "
        rf"statistic.}} Only ${n_defined}$ of these rows are mechanically distinct worlds: the five "
        rf"\textsc{{sir}} rows are one world at different severities and instrument prices (the "
        rf"within-world panel of \cref{{fig:surface}}). The share clearing $1.1\times$ is therefore "
        rf"${frac_rows[0]}/{frac_rows[1]}$ counting rows, ${frac_best[0]}/{frac_best[1]}$ counting "
        rf"each world at its best configuration, and ${frac_med[0]}/{frac_med[1]}$ counting each at "
        rf"its median --- and since we chose the library's contents there is no sampling frame for "
        rf"any of the three to estimate. What does not depend on the convention is the \emph{{bound}}: "
        rf"the largest adaptation headroom anywhere in the library is ${max_adapt:.3f}\times$ "
        rf"({max_world}), so even a clairvoyant re-legislator recovers at most "
        rf"${100 * (1 - 1 / max_adapt):.0f}\%$ of the incumbent rule's cost. The two declared "
        rf"nulls are informative rather "
        r"than disappointing: the fiscal world's optimum is $\tau^\ast=0$ and its shock enters as "
        r"$\tau_{\text{eff}} = c\cdot\tau$, so a compliance collapse multiplies zero and is "
        r"\emph{arithmetically} inert; and the supply chain answers the one shock family "
        r"(\textsc{delay}) our taxonomy had never measured, showing it is absorbed by an order-up-to "
        r"rule with a free target exactly as a state shock would be. The commons is reported as an "
        r"absolute gap: its objective is net welfare, so a good policy scores below zero and the "
        r"ratio is undefined rather than merely imprecise.}",
        r"\label{tab:library}",
        r"\begin{tabular}{@{}l l r r r@{}}",
        r"\toprule",
        r"world & instrument & staleness & \textbf{adaptation} & robustness \\",
        r"\midrule",
    ]
    for r in valid:
        world, instrument, _shock = _label(r["name"])
        bold = r"\textbf{%.3f}" % r["adaptation_headroom"]
        lines.append(f"{world} & {instrument} & {r['staleness']:.3f} & {bold} & "
                     f"{r['robustness_headroom']:.3f} \\\\")
    for r in invalid:
        world, instrument, _shock = _label(r["name"])
        lines.append(f"{world} & {instrument} & \\multicolumn{{3}}{{c}}{{ratio undefined; "
                     f"adaptation \\emph{{gap}} ${r['adaptation_gap']:.2f}$}} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out} ({len(valid)} ranked + {len(invalid)} unranked worlds)")

    # A macro file so prose cannot drift from the table it describes.
    facts = Path(out.parent / "library_facts.tex")
    best = valid[0]
    facts.write_text("\n".join([
        r"\newcommand{\libWorlds}{%d}" % n_total,
        r"\newcommand{\libRows}{%d}" % len(rows),
        r"\newcommand{\libDefined}{%d}" % n_defined,
        # The three conventions, as macros, so any sentence quoting a fraction must quote which one.
        r"\newcommand{\libFracRows}{%d/%d}" % frac_rows,
        r"\newcommand{\libFracBest}{%d/%d}" % frac_best,
        r"\newcommand{\libFracMedian}{%d/%d}" % frac_med,
        # The convention-invariant claims that replaced the fraction.
        r"\newcommand{\libMaxAdapt}{%.3f}" % max_adapt,
        r"\newcommand{\libMaxRecovered}{%.0f}" % (100 * (1 - 1 / max_adapt)),
        r"\newcommand{\libEpiMin}{%.3f}" % epi[0],
        r"\newcommand{\libEpiMax}{%.3f}" % epi[-1],
        r"\newcommand{\libEpiConfigs}{%d}" % len(epi),
        r"\newcommand{\libBestWorld}{%s}" % _label(best["name"])[0],
        r"\newcommand{\libBestAdapt}{%.3f}" % best["adaptation_headroom"],
        r"\newcommand{\libBestStale}{%.3f}" % best["staleness"],
        "",
    ]), encoding="utf-8")
    print(f"wrote {facts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
