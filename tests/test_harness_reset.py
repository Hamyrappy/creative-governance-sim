"""Harness component state must not leak between seeds.

`Runner.run` executes every seed against the SAME Experiment object. Any component that accumulates
therefore carried one run's history into the next until this was fixed: measured on three seeds of a
20-decision world, EpisodicMemory finished holding 60 episodes instead of 20 and OutcomeFeedback 57
log entries instead of 19.

That is not untidiness. It breaks the design's core assumption: the agent at seed 19 was retrieving
precedent from nineteen OTHER world realisations, and the outcome channel was reporting scores earned
in runs the current world never saw. A paired seed design assumes independent draws and every
statistic downstream assumes it too.

The bug is invisible in the outputs — every arm still produces 20 plausible losses — so it needs a
test rather than vigilance.
"""

from __future__ import annotations

from govsim.core.harness import Harness, HarnessComponent
from govsim.core.regent import MultiScriptedRegent
from govsim.core.runner import Runner
from govsim.experiments import get
from govsim.harness import (
    ContextualOutcomeFeedback,
    ContrastiveMemory,
    EpisodicMemory,
    ForeignMemory,
    OutcomeFeedback,
    UnscoredMemory,
)

DECISIONS_PER_RUN = 20  # horizon 200, EveryN(10)


def _run_seeds(components, seeds=(0, 1, 2)):
    exp = get("epidemic_llm_bare")
    exp.regents = {"regent:0": MultiScriptedRegent(
        {"set_lockdown": "0.5", "set_vaccination": "0.2"})}
    exp.harness = Harness(list(components))
    exp.seeds = list(seeds)
    Runner().run(exp)
    return exp


def test_episodic_memory_does_not_accumulate_across_seeds():
    mem = EpisodicMemory(k=4)
    _run_seeds([mem])
    assert len(mem.episodes) == DECISIONS_PER_RUN, len(mem.episodes)


def test_outcome_feedback_does_not_accumulate_across_seeds():
    out = OutcomeFeedback(k=4)
    _run_seeds([out])
    # One fewer than the decision count: the first decision has no preceding interval to score.
    assert len(out.log) == DECISIONS_PER_RUN - 1, len(out.log)


def test_contextual_outcome_clears_both_of_its_logs():
    """It keeps a second, parallel log; resetting only the inherited one would leave it stale."""
    ctx = ContextualOutcomeFeedback(k=4)
    _run_seeds([ctx])
    assert len(ctx.log) == DECISIONS_PER_RUN - 1
    assert len(ctx.log_ctx) == DECISIONS_PER_RUN - 1


def test_the_memory_variants_reset_too():
    for cls in (ContrastiveMemory, UnscoredMemory):
        c = cls(k=4)
        _run_seeds([c])
        assert len(c.episodes) == DECISIONS_PER_RUN, (cls.__name__, len(c.episodes))


def test_foreign_memory_RESTORES_its_donor_bank_rather_than_clearing_it():
    """The parent's reset empties `episodes`. Inheriting that unchanged would leave this arm with no
    memory from the second seed onward — silently a no-harness control wearing a memory label."""
    bank = [{"state": {"I": 0.1}, "actions": [{"verb": "set_lockdown", "expr": "0.5"}], "score": 1.0},
            {"state": {"I": 0.2}, "actions": [{"verb": "set_lockdown", "expr": "0.7"}], "score": 2.0}]
    fm = ForeignMemory(bank, k=2)
    fm.reset()
    assert len(fm.episodes) == 2
    _run_seeds([fm])
    assert len(fm.episodes) == 2, "the donor bank must survive every seed unchanged"


def test_foreign_memory_never_records_the_current_run():
    fm = ForeignMemory([{"state": {"I": 0.1}, "actions": [{"verb": "set_lockdown", "expr": "0.5"}],
                         "score": 1.0}], k=2)
    _run_seeds([fm])
    assert len(fm.episodes) == 1, "on_outcome must stay a no-op or the bank stops being foreign"


def test_foreign_memory_refuses_an_empty_bank():
    """An empty bank would degrade the arm into a no-harness control while still labelled memory."""
    try:
        ForeignMemory([], k=2)
    except ValueError:
        return
    raise AssertionError("an empty donor bank must raise")


def test_a_stateless_component_needs_no_reset_and_the_default_is_harmless():
    class Plain(HarnessComponent):
        name = "plain"

    Harness([Plain()]).reset()  # must not raise


def test_disabled_components_are_reset_too():
    """`enabled` is the ablation switch; a component toggled back on must not wake up holding
    another run's history."""
    mem = EpisodicMemory(k=4)
    mem.episodes.append({"state": {"I": 0.1}, "actions": [], "score": 1.0})
    mem.enabled = False
    Harness([mem]).reset()
    assert mem.episodes == []
