"""Pin the foreign-memory result and the three checks that make it survivable.

`ForeignMemory` was built to separate anchoring from commitment. It did not settle that question and
the papers no longer claim it does: the component inherits its rendering from the parent, so the
donor's episodes reach the prompt as "when [state] YOU DID [law]" and the agent has no authorship
signal at all. What the arm does establish is a PROXIMITY effect -- own precedent is retrieved 2.4x
nearer the current state -- plus two mechanism-independent facts (the onset and the break response).

The retracted claim is pinned here as hard as the surviving ones. A test that only guards the current
story lets the old one creep back during an edit, and this particular error survived a full commit,
two abstracts and a figure caption before the corrected measurement caught it.

Each test names one claim the papers make, so a failure points at a sentence rather than at a number.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
FOREIGN = REPO / "logs" / "foreign_memory.json"
LOSS = REPO / "logs" / "foreign_loss.json"
ONSET = REPO / "logs" / "lockin_onset.json"
SIMILARITY = REPO / "logs" / "retrieval_similarity.json"
PAPERS = (REPO / "paper" / "main.tex", REPO / "paper" / "social.tex")

pytestmark = pytest.mark.skipif(
    not FOREIGN.exists(),
    reason="foreign-memory artifact absent; run the epidemic_llm_foreign arm and the analysis",
)


def _load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


# --- the pre-registered claim ----------------------------------------------------------------------

def test_swapping_the_bank_recovers_most_of_the_suppressed_revision():
    """The pre-registered contrast. `ForeignMemory`'s docstring predicted churn would rise materially
    above 0.082 toward the no-memory arm's 0.847 if the effect needed the agent's own episodes, and
    stay near 0.082 if any exemplar anchored. It rose, which rules out plain anchoring.

    It does NOT identify what about one's own episodes matters -- authorship is excluded by the
    rendering (see test_neither_paper_claims_the_retracted_authorship_mechanism) and the papers
    attribute it to retrieval proximity instead."""
    means = _load(FOREIGN)["means"]
    assert means["own"] < 0.15, means
    assert means["bare"] > 0.75, means
    assert means["own"] + 0.30 < means["foreign"] < means["bare"] - 0.10, means


def test_all_three_churn_contrasts_survive_holm_correction():
    d = _load(FOREIGN)
    for name, c in d["contrasts"].items():
        assert c["p_holm"] <= 0.05, (name, c)
        assert c["robust_agreement"], (name, "bootstrap and Wilcoxon disagree")


def test_the_bank_substitution_recovers_a_minority_of_the_suppression():
    """Both papers say foreign precedent reproduces roughly a quarter of own-precedent's suppression.
    If that crosses a half, "most of the suppressed revision" is the wrong phrase in both."""
    d = _load(FOREIGN)
    assert 0.0 < d["fraction_reproduced"] < 0.5, d["fraction_reproduced"]


# --- the direction of the loss claim ----------------------------------------------------------------

@pytest.mark.skipif(not LOSS.exists(), reason="loss artifact absent")
def test_foreign_removes_the_harm_but_is_not_claimed_to_help():
    """Both papers state this asymmetry explicitly, and it is the easiest thing to overstate:
    foreign beats OWN memory significantly, and does NOT beat the no-memory arm."""
    c = _load(LOSS)["contrasts"]
    assert c["foreign - own"]["sig"] and c["foreign - own"]["delta"] < 0, c["foreign - own"]
    assert not c["foreign - bare"]["sig"], (
        "foreign now beats the no-memory arm; both papers say it does not, and the sentence "
        "'harmless where own precedent is harmful' understates the result"
    )
    for paper in PAPERS:
        text = paper.read_text(encoding="utf-8")
        if "another authority" in text:
            assert "0.61" in text, f"{paper.name} should state the null against the no-memory arm"


# --- the three alternative explanations -------------------------------------------------------------

@pytest.mark.skipif(not SIMILARITY.exists(), reason="similarity artifact absent")
def test_own_precedent_is_retrieved_from_nearer_the_current_state():
    """The proximity mechanism the papers now report. An earlier version of this test asserted the
    OPPOSITE (ratio < 1) because the probe reconstructed a 180-episode donor pool instead of reading
    the 8-episode bank the arm receives; a larger bank has combinatorially nearer neighbours, so the
    reconstruction measured itself. Both papers quote the corrected factor, so it is pinned."""
    d = _load(SIMILARITY)
    assert d["ratio"] > 1.5, (
        f"own precedent is no longer retrieved appreciably nearer ({d['ratio']:.2f}x); the proximity "
        f"account in both papers rests on this gap"
    )
    for paper in PAPERS:
        text = paper.read_text(encoding="utf-8")
        if "lockinForeignPre" in text:
            assert f"{d['ratio']:.1f}" in text or f"{d['own_mean']:.2f}" in text, (
                f"{paper.name} does not state the measured retrieval distances"
            )


@pytest.mark.skipif(not SIMILARITY.exists(), reason="similarity artifact absent")
def test_the_similarity_probe_reads_the_arms_real_donor_bank():
    """The defect that caused the retraction. The probe must import the experiment's own loader
    rather than rebuild the bank, because a rebuilt bank of a different size answers a different
    question and does so silently."""
    src = (REPO / "scripts" / "retrieval_similarity.py").read_text(encoding="utf-8")
    assert "_donor_bank" in src and "FOREIGN_DONOR_SEEDS" in src, (
        "the probe no longer loads the bank from the experiment definition; it can now drift from "
        "what the arm actually received, which is the error this test exists to prevent"
    )


def test_neither_paper_claims_the_retracted_authorship_mechanism():
    """`ForeignMemory` does not override on_observe, so the donor's episodes are rendered in the
    parent's second person and the agent cannot tell whose they are. Any sentence attributing the
    effect to self-consistency or to recognising foreign authorship is unsupported."""
    from govsim.harness.components import EpisodicMemory, ForeignMemory
    assert ForeignMemory.on_observe is EpisodicMemory.on_observe, (
        "ForeignMemory now renders its own prompt. If it signals authorship, the papers' argument "
        "that authorship CANNOT be the mechanism no longer holds and must be revisited."
    )
    banned = ("self-consistency pressure", "consistency pressure", "recognises the precedent as")
    for paper in PAPERS:
        text = paper.read_text(encoding="utf-8")
        for phrase in banned:
            if phrase in text:
                # allowed only where the paper is explicitly naming the hypothesis it rejects
                idx = text.index(phrase)
                window = text[max(0, idx - 400):idx + 200]
                assert "not" in window or r"Under \emph{commitment}" in window, (
                    f"{paper.name} appears to assert the retracted authorship mechanism: "
                    f"...{text[max(0, idx - 120):idx + 120]}..."
                )


@pytest.mark.skipif(not ONSET.exists(), reason="onset artifact absent")
def test_lockin_arrives_before_the_bank_can_be_repetitive():
    """The papers rule out 'the bank became repetitive' by pointing at the FIRST transition, where
    own memory holds exactly one episode. That argument needs the onset to be immediate."""
    curve = _load(ONSET)["curve"]
    first = curve[0]
    assert first["own_bank_size"] == 1, first
    assert first["own"] < first["bare"] - 0.3, first
    # NB this rules out accounts needing the bank to ACCUMULATE; it does not discriminate proximity
    # from anything else, because a one-episode own bank holds the immediately preceding decision,
    # which is the nearest precedent obtainable. The papers say so.
    # and total by two episodes, which is the second half of the sentence
    assert curve[1]["own"] == 0.0, curve[1]


@pytest.mark.skipif(not ONSET.exists(), reason="onset artifact absent")
def test_only_the_foreign_arm_responds_to_the_break():
    """The exploratory finding both papers report. It is labelled exploratory precisely because it
    was not pre-registered, but it is still a claim in the PDFs."""
    resp = _load(ONSET)["break_response"]
    assert resp["foreign"]["significant"] and resp["foreign"]["delta"] > 0.2, resp["foreign"]
    assert not resp["bare"]["significant"], resp["bare"]
    assert not resp["own"]["significant"], resp["own"]


@pytest.mark.skipif(not ONSET.exists(), reason="onset artifact absent")
def test_the_exploratory_label_is_actually_in_both_papers():
    """A finding this quotable is exactly the one that loses its hedge during an edit."""
    for paper in PAPERS:
        text = paper.read_text(encoding="utf-8")
        if "lockinForeignPre" not in text:
            continue
        assert "exploratory" in text, (
            f"{paper.name} reports the break-alignment result without labelling it exploratory"
        )
