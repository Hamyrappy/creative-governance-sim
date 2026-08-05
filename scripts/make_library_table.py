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

    n_total = len(rows)
    n_over_11 = sum(1 for r in valid if r["adaptation_headroom"] >= 1.1)
    n_over_12 = sum(1 for r in valid if r["adaptation_headroom"] >= 1.2)

    lines = [
        r"\begin{table}[t]\centering\small",
        r"\caption{\textbf{Adaptation headroom across the world library.} Ten worlds with unrelated "
        r"mechanics, each decomposed by the same construction as \cref{tab:headroom}: every "
        r"reference is calibrated by exhaustive search over that world's own policy vocabulary, on "
        r"the same broken world, over the same seeds, and the three differ only in what they were "
        r"allowed to know. Ranked by \emph{adaptation} headroom, because that is the only factor an "
        rf"adaptive agent can claim. Only ${n_over_11}/{n_total}$ worlds clear $1.1\times$ and "
        rf"${n_over_12}/{n_total}$ clears $1.2\times$. The two declared nulls are informative rather "
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
        r"\newcommand{\libOverEleven}{%d}" % n_over_11,
        r"\newcommand{\libOverTwelve}{%d}" % n_over_12,
        r"\newcommand{\libBestWorld}{%s}" % _label(best["name"])[0],
        r"\newcommand{\libBestAdapt}{%.3f}" % best["adaptation_headroom"],
        r"\newcommand{\libBestStale}{%.3f}" % best["staleness"],
        "",
    ]), encoding="utf-8")
    print(f"wrote {facts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
