"""Pin the foreign-memory result and the three checks that make it survivable.

`ForeignMemory` answers a question the presentation repairs could not: is the lock-in caused by
exposure to any concrete precedent (anchoring), or specifically by the agent's own record
(commitment)? The answer is commitment, and both manuscripts now carry a design prescription that
rests on it. These tests fail if the artifact stops supporting the sentences in the PDFs.

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

def test_the_prediction_registered_before_the_arm_ran_is_the_one_confirmed():
    """`ForeignMemory`'s docstring says: under commitment churn rises materially above 0.082 toward
    the no-memory arm's 0.847; under anchoring it stays near 0.082. Both papers report commitment."""
    d = _load(FOREIGN)
    means = d["means"]
    assert means["own"] < 0.15, means
    assert means["bare"] > 0.75, means
    # "materially above own, and well short of bare" is what commitment predicts and anchoring does
    # not. A foreign arm that landed at either endpoint would change the papers' reading.
    assert means["own"] + 0.30 < means["foreign"] < means["bare"] - 0.10, means


def test_all_three_churn_contrasts_survive_holm_correction():
    d = _load(FOREIGN)
    for name, c in d["contrasts"].items():
        assert c["p_holm"] <= 0.05, (name, c)
        assert c["robust_agreement"], (name, "bootstrap and Wilcoxon disagree")


def test_the_authorship_residual_is_the_larger_share():
    """The papers say foreign precedent reproduces roughly a quarter of the suppression and the
    authorship residual is the rest. If that ratio crosses a half the wording is wrong."""
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
def test_the_retrieval_similarity_confound_still_runs_backwards():
    """Both papers claim foreign precedent is retrieved CLOSER, not further. If that ever inverts,
    the commitment reading is confounded and the claim must be withdrawn, not softened."""
    d = _load(SIMILARITY)
    assert d["ratio"] < 1.0, (
        f"foreign precedent is now retrieved {d['ratio']:.2f}x FURTHER away; the papers' rebuttal "
        f"of the similarity confound no longer holds"
    )


@pytest.mark.skipif(not ONSET.exists(), reason="onset artifact absent")
def test_lockin_arrives_before_the_bank_can_be_repetitive():
    """The papers rule out 'the bank became repetitive' by pointing at the FIRST transition, where
    own memory holds exactly one episode. That argument needs the onset to be immediate."""
    curve = _load(ONSET)["curve"]
    first = curve[0]
    assert first["own_bank_size"] == 1, first
    assert first["own"] < first["bare"] - 0.3, first
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
