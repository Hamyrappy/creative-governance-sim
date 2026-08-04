"""
Rebuild every reported artifact from the stores, in the right order, in one command.

    uv run python scripts/finalize.py --store logs/runs_gemma logs/runs_gemma26 --model gemma-4-31b-it

The order is not cosmetic. Reference arms must be re-run after any recalibration or the anchors in
the store belong to a previous calibration and every normalized regret is silently rescaled; tables
must be generated after the analysis JSON exists or they render their "pending" stubs; the papers
must be built after both or they embed stale numbers. Doing this by hand at the end of a long session
is exactly when a step gets skipped, so it is a script.

Each step reports what it did and the run keeps going on failure, so one broken step does not hide
the state of the others.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(label: str, cmd: list[str], cwd: Path = ROOT, env: dict | None = None) -> bool:
    print(f"\n{'=' * 78}\n== {label}\n{'=' * 78}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=cwd, env={**os.environ, **(env or {})},
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = (proc.stdout or "") + (proc.stderr or "")
    tail = "\n".join(out.strip().splitlines()[-25:])
    print(tail, flush=True)
    ok = proc.returncode == 0
    print(f"-- {'ok' if ok else 'FAILED (rc=%d)' % proc.returncode} in {time.time() - t0:.1f}s",
          flush=True)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", nargs="+", default=["logs/runs_gemma"])
    ap.add_argument("--model", default="gemma-4-31b-it")
    ap.add_argument("--skip-refs", action="store_true",
                    help="skip re-running the key-free reference arms (only if calibration is unchanged)")
    ap.add_argument("--analysis", default="logs/analysis.json")
    args = ap.parse_args()

    py = [sys.executable]
    results: dict[str, bool] = {}

    # 1. Reference arms, into EVERY store, so each store's anchors match the current calibration.
    if not args.skip_refs:
        for store in args.store:
            results[f"refs:{store}"] = run(
                f"reference arms -> {store}",
                py + ["scripts/run_matrix.py", "--arms", "epidemic-refs", "--seeds", "20",
                      "--models", args.model, "--store", store,
                      "--summary", f"logs/refs_{Path(store).name}.json"])

    # 2. Analysis (needs the refs), then the policy audit (independent, reads llm_io artifacts).
    results["analysis"] = run(
        "analysis",
        py + ["scripts/analyze_matrix.py", "--store", *args.store, "--model", args.model,
              "--cross-model", "--json", args.analysis])
    results["policy_audit"] = run(
        "policy audit",
        py + ["scripts/policy_audit.py", "--store", *args.store,
              "--json", "logs/policy_audit.json"])

    # 3. Tables and figures (need the analysis JSON).
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip() or "unknown"
    results["tables"] = run("tables", py + ["scripts/make_tables.py", "--analysis", args.analysis,
                                            "--commit", commit])
    results["figures"] = run("figures", py + ["scripts/make_figures.py", "--analysis", args.analysis])

    # 4. Both manuscripts, twice through bibtex so cross-references resolve.
    for doc in ("main", "social"):
        ok = True
        for step in (["pdflatex", "-interaction=nonstopmode", f"{doc}.tex"],
                     ["bibtex", doc],
                     ["pdflatex", "-interaction=nonstopmode", f"{doc}.tex"],
                     ["pdflatex", "-interaction=nonstopmode", f"{doc}.tex"]):
            proc = subprocess.run(step, cwd=ROOT / "paper", capture_output=True, text=True,
                                  encoding="utf-8", errors="replace")
            ok = ok and (proc.returncode == 0 or step[0] == "bibtex")
        pdf = ROOT / "paper" / f"{doc}.pdf"
        log = (ROOT / "paper" / f"{doc}.log").read_text(encoding="utf-8", errors="replace")
        pages = [ln for ln in log.splitlines() if "Output written on" in ln]
        undefined = log.count("Citation") and log.count("undefined")
        print(f"\n== paper/{doc}.pdf: {'built' if pdf.exists() else 'MISSING'}"
              f"  {pages[-1].strip() if pages else ''}"
              f"  undefined-citation lines: {undefined}", flush=True)
        results[f"paper:{doc}"] = pdf.exists()

    print(f"\n{'=' * 78}\n== summary")
    for k, v in results.items():
        print(f"  {'ok  ' if v else 'FAIL'}  {k}")
    failed = [k for k, v in results.items() if not v]
    print(f"== {len(results) - len(failed)}/{len(results)} steps ok")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
