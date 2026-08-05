"""Hard numbers in the manuscripts must match the artifacts they came from.

The repository claims "no number in the paper is typed by hand". That is true of the generated
tables and false of the prose, and the gap cost real defects: a figure caption asserting "no arm
reaches the best fixed rule" while two arms did, a churn table printing one convention while the
prose two paragraphs later claimed the other, the clairvoyant's post-break rate appearing as both
$1.53$ and $1.47$ in the same document, and a headroom figure quoted from a computation that was
never saved.

Those were found by a reviewer reading the artifacts alongside the PDF. This file does the same
check on every subsequent edit, for the numbers that carry an argument.

DESIGN NOTE. This deliberately does NOT try to parse every numeral out of the LaTeX. That would be
brittle and would fail on section numbers, page counts and parameter values, producing noise that
gets suppressed — and a suppressed check is worse than none. Instead each entry names one claim, the
artifact it must agree with, and the tolerance. Adding a claim is one line; the cost of not adding it
is that the number is unchecked, which is the status quo for the rest.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
MAIN = (REPO / "paper" / "main.tex").read_text(encoding="utf-8")
ANALYSIS = REPO / "logs" / "analysis_v3.json"
DECOMP = REPO / "logs" / "decomposition.json"
CHURN = REPO / "logs" / "churn.json"
ZLB = REPO / "logs" / "zlb_corner.json"

pytestmark = pytest.mark.skipif(
    not ANALYSIS.exists(),
    reason="analysis artifact absent; run scripts/analyze_matrix.py --json logs/analysis_v3.json",
)


def _analysis() -> dict:
    return json.loads(ANALYSIS.read_text(encoding="utf-8"))


def _appears(value: float, decimals: int) -> bool:
    """Is this value, formatted as the paper formats it, present in main.tex?

    Matching the rendered string rather than parsing arithmetic keeps the check honest: it fails when
    the PDF a reader sees disagrees with the artifact, which is the failure that matters.
    """
    return f"{value:.{decimals}f}" in MAIN


# --- factorial effects ---------------------------------------------------------------------------

@pytest.mark.parametrize("term,decimals", [
    ("memory", 3),
    ("outcome:memory", 3),
])
def test_reported_factorial_effects_match_the_artifact(term, decimals):
    fac = _analysis()["factorial"]
    assert term in fac, f"{term} absent from the analysis artifact"
    assert _appears(abs(fac[term]["effect"]), decimals), (
        f"main.tex does not contain the {term} effect "
        f"{abs(fac[term]['effect']):.{decimals}f} from the artifact"
    )


def test_the_memory_effect_is_still_the_only_one_that_clears_its_own_mde():
    """The paper's central harness claim. If another term becomes resolvable, or memory stops being
    so, several sentences need rewriting and this is where that surfaces."""
    a = _analysis()
    fac, mde = a["factorial"], a.get("mde_per_term") or {}
    assert mde, "per-term MDEs absent; re-run analyze_matrix"
    resolvable = sorted(t for t, m in mde.items()
                        if abs(fac[t]["effect"]) >= m["mde"])
    assert resolvable == ["memory"], resolvable


# --- the no-action / collapse contamination the paper reports as a limitation ---------------------

def test_the_contaminated_cells_are_exactly_the_outcome_bearing_ones():
    """The paper's limitation section rests on this being a clean split. If a memory-only arm ever
    starts collapsing, the confound stops being perfectly aligned with the factor and the argument
    in that section changes."""
    rates = _analysis()["action_rates"]
    over = {k for k, v in rates.items() if v["rate"] > 0.02}
    assert over, "no arm exceeds the 2% gate; the limitation section is now stale"
    assert all("outcome" in k for k in over), sorted(over)
    under = {k: v["rate"] for k, v in rates.items() if "outcome" not in k}
    assert all(r <= 0.0025 for r in under.values()), under


# --- claims a reader can check against the arms table ---------------------------------------------

def test_the_arms_that_beat_the_best_fixed_rule_are_the_two_the_caption_names():
    """A caption previously asserted that NO arm reached the best fixed rule while two did."""
    arms = {r["arm"]: r["R"] for r in _analysis()["arms"] if r.get("R") is not None}
    beat = sorted(k for k, v in arms.items() if v < 1.0)
    assert beat == ["epidemic_llm_bare", "epidemic_llm_trace"], beat
    for name in beat:
        assert _appears(arms[name], 2), f"R={arms[name]:.2f} for {name} is not stated in main.tex"


# --- the churn table must print the convention the prose claims -----------------------------------

@pytest.mark.skipif(not CHURN.exists(), reason="churn artifact absent")
def test_the_churn_table_uses_the_conservative_convention():
    """The paper states it counts a collapsed decision as a NON-revision, so the table must show
    churn_carry and NOT churn_skip.

    Asserting the correct value is present is not enough, and this test found that out: 0.847 occurs
    twice in main.tex, so swapping the churn row back to the permissive 0.850 left the other
    occurrence and the check passed on a document that had the defect. What identifies the mistake is
    the presence of the WRONG value, so that is what is asserted.
    """
    churn = json.loads(CHURN.read_text(encoding="utf-8"))
    for arm in ("epidemic_llm_bare", "epidemic_llm_outcome"):
        assert arm in churn, arm
        carry, skip = churn[arm]["churn_carry"], churn[arm]["churn_skip"]
        assert _appears(carry, 3), (
            f"{arm}: main.tex should print the conservative churn {carry:.3f}"
        )
        if abs(carry - skip) > 5e-4:
            assert not _appears(skip, 3), (
                f"{arm}: main.tex contains the PERMISSIVE churn {skip:.3f} while the prose claims "
                f"the conservative convention ({carry:.3f})"
            )


# --- numbers that must have an artifact behind them ------------------------------------------------

@pytest.mark.skipif(not ZLB.exists(), reason="ZLB artifact absent; run scripts/zlb_corner_check.py")
def test_the_zlb_corner_numbers_are_sourced():
    """This pair was quoted from a computation that was never saved. A manuscript figure with no
    artifact behind it is the same defect as a stale one and harder to notice."""
    z = json.loads(ZLB.read_text(encoding="utf-8"))["verdict"]
    assert _appears(z["narrow"], 4) and _appears(z["extended"], 4), z
    assert not z["corner_is_binding"], (
        "the grid corner now binds; the paper's claim that the headline headroom is not a lower "
        "bound no longer holds"
    )


@pytest.mark.skipif(not DECOMP.exists(), reason="decomposition artifact absent")
def test_the_library_headline_matches_the_decomposition():
    rows = json.loads(DECOMP.read_text(encoding="utf-8"))
    valid = [r for r in rows if r.get("ratios_valid")]
    best = max(valid, key=lambda r: r["adaptation_headroom"])
    assert "monetary" in best["name"], best["name"]
    assert _appears(best["adaptation_headroom"], 3), best["adaptation_headroom"]


# --- internal consistency, independent of any artifact --------------------------------------------

def test_no_quantity_is_stated_with_two_different_values():
    """The clairvoyant's post-break rate once appeared as both 1.53 and 1.47 in the same document."""
    for label, pattern, allowed in [
        ("clairvoyant post-break rate", r"clairvoyant'?s?\s+\$?(\d\.\d\d)\$?", {"1.53"}),
    ]:
        found = set(re.findall(pattern, MAIN))
        assert found <= allowed, f"{label}: conflicting values in main.tex: {sorted(found)}"


def test_the_gate_count_is_stated_consistently():
    counts = set(re.findall(r"(?:^|\s)(Four|Five|Six|four|five|six) validity gates", MAIN))
    assert len({c.lower() for c in counts}) <= 1, f"main.tex claims several gate counts: {counts}"
