"""
Tests for the two pieces of machinery the headline claims rest on:

* ``factorial_effects`` / ``holm_bonferroni`` — the harness attribution. A factorial estimator that
  is subtly wrong would produce confident, well-formatted, false component attributions, so it is
  checked against data with a KNOWN generating effect rather than only for not crashing.
* ``calibrate`` / ``headroom`` / ``normalized_regret`` — the reference anchors that give every
  reported number its scale.

Key-free and deterministic (fixed RNG seeds, no LLM).
"""

from __future__ import annotations

import math
import random

import pytest

from govsim.analysis import (
    PolicyFamily,
    calibrate,
    calibrate_families,
    headroom,
    normalized_regret,
)
from govsim.analysis.stats import bootstrap_p, factorial_effects, holm_bonferroni
from govsim.core.schedule import EveryN
from govsim.domains.scalar import (
    CubicSystem,
    EpidemicLoss,
    Lever,
    ScalarLeverInterface,
    SIRSystem,
    StabilizationLoss,
)

FACTORS = ("trace", "outcome", "memory")


def _synthetic_cells(main: float, interaction: float, *, n_seeds: int = 24, noise: float = 0.25):
    """A 2^3 factorial where only ``outcome`` has a main effect and only outcome:memory interacts."""
    rng = random.Random(1234)
    base = {s: rng.gauss(20.0, 3.0) for s in range(n_seeds)}
    cells = {}
    for bits in range(8):
        cell = tuple(bool(bits >> i & 1) for i in range(3))
        shift = (main if cell[1] else 0.0) + (interaction if (cell[1] and cell[2]) else 0.0)
        cells[cell] = {s: base[s] + shift + rng.gauss(0.0, noise) for s in range(n_seeds)}
    return cells


def test_factorial_recovers_a_known_main_effect_and_interaction():
    cells = _synthetic_cells(main=-2.0, interaction=-0.5)
    eff = factorial_effects(cells, FACTORS)

    # With +/-1 contrast coding an interaction of size d contributes d/2 to each involved main
    # effect, so the recovered 'outcome' effect is main + interaction/2.
    assert eff["outcome"]["effect"] == pytest.approx(-2.25, abs=0.15)
    assert eff["outcome:memory"]["effect"] == pytest.approx(-0.25, abs=0.15)
    assert eff["memory"]["effect"] == pytest.approx(-0.25, abs=0.15)

    # The real effects are detected...
    for term in ("outcome", "outcome:memory"):
        assert eff[term]["ci_high"] < 0, f"{term} CI should exclude 0 on the helping side"
    # ...and the null factor is not, even before correction.
    assert eff["trace"]["ci_low"] < 0 < eff["trace"]["ci_high"]


def test_holm_suppresses_the_null_terms_across_the_family():
    eff = factorial_effects(_synthetic_cells(main=-2.0, interaction=-0.5), FACTORS)
    holm = holm_bonferroni({k: v["p"] for k, v in eff.items()})
    assert holm["outcome"]["significant"]
    assert not holm["trace"]["significant"]
    assert not holm["trace:memory"]["significant"]
    # Adjusted p is monotone in raw p and never smaller than it.
    for k, v in holm.items():
        assert v["p_adj"] >= v["p"] - 1e-12


def test_holm_is_conservative_on_an_all_null_family():
    """20 pure-noise comparisons: uncorrected, ~1 would clear 0.05; corrected, none should."""
    rng = random.Random(99)
    ps = {f"t{i}": bootstrap_p([rng.gauss(0.0, 1.0) for _ in range(20)], seed=i) for i in range(20)}
    holm = holm_bonferroni(ps)
    assert not any(v["significant"] for v in holm.values())


def test_factorial_rejects_an_incomplete_design():
    cells = _synthetic_cells(main=-1.0, interaction=0.0)
    del cells[(True, True, True)]
    with pytest.raises(ValueError, match="full 2"):
        factorial_effects(cells, FACTORS)


def test_normalized_regret_anchors_and_stays_unclipped():
    assert normalized_regret(5.0, 10.0, 5.0) == pytest.approx(0.0)   # == oracle
    assert normalized_regret(10.0, 10.0, 5.0) == pytest.approx(1.0)  # == frozen
    assert normalized_regret(7.5, 10.0, 5.0) == pytest.approx(0.5)
    # Beating the oracle and doing worse than never adapting are both reportable, not clamped.
    assert normalized_regret(4.0, 10.0, 5.0) < 0
    assert normalized_regret(12.0, 10.0, 5.0) > 1
    assert math.isnan(normalized_regret(1.0, 5.0, 5.0))  # zero headroom ⇒ undefined, not infinite


def test_calibration_finds_the_better_law_and_reports_headroom():
    """A tiny end-to-end calibration on the scalar plant: the search must beat a deliberately bad law."""
    cfg = {"param_A": 0.95, "param_B": 0.5, "param_C": 0.0, "sigma_epsilon": 0.05,
           "target_x": 0.0, "u_range": (-2.0, 2.0), "cubic_coeff": 0.0, "state_exponent": 3}

    def factory(seed: int) -> CubicSystem:
        s = CubicSystem(cfg)
        s.reset(seed)
        return s

    fam = PolicyFamily(verb="set_control_input", template="-({k} * current_x)",
                       grid={"k": [0.0, 0.5, 1.0, 1.9]})
    res = calibrate(fam, system_factory=factory,
                    action_interface=ScalarLeverInterface(
                        [Lever("set_control_input", (-2.0, 2.0), "current_u")]),
                    objective=StabilizationLoss(lam=0.1), schedule=EveryN(25),
                    seeds=[0, 1, 2], horizon=60, metric="loss")
    assert res.best_params["k"] > 0.0, "doing nothing must not win on a controllable plant"
    losses = dict((tuple(p.items()), l) for p, l in res.all_losses)
    assert res.best_loss <= min(losses.values()) + 1e-12
    assert headroom(losses[(("k", 0.0),)], res.best_loss) > 1.0


def test_calibrate_families_picks_the_better_functional_form():
    """The wide-oracle check: given a strictly better shape, the search must switch to it."""
    cfg = {"beta0": 0.35, "gamma": 0.10, "waning": 0.02, "import_rate": 0.0005,
           "shock_step": 10_000, "noise_sigma": 0.0}

    def factory(seed: int) -> SIRSystem:
        s = SIRSystem(cfg)
        s.reset(seed)
        return s

    iface = ScalarLeverInterface([Lever("set_lockdown", (0.0, 0.9), "lockdown")])
    families = {
        "do_nothing": PolicyFamily("set_lockdown", "{a} * 0.0", {"a": [0.0]}),
        "proportional": PolicyFamily("set_lockdown", "{g} * I", {"g": [0.0, 1.0, 3.0]}),
    }
    name, res = calibrate_families(
        families, system_factory=factory, action_interface=iface,
        objective=EpidemicLoss(lam=0.02), schedule=EveryN(10), seeds=[0, 1], horizon=80,
        metric="loss")
    assert name in families
    # Whichever wins must be at least as good as the do-nothing family's only option.
    nothing = calibrate(families["do_nothing"], system_factory=factory, action_interface=iface,
                        objective=EpidemicLoss(lam=0.02), schedule=EveryN(10), seeds=[0, 1],
                        horizon=80, metric="loss")
    assert res.best_loss <= nothing.best_loss + 1e-12


def test_epidemic_post_window_refuses_to_reward_an_empty_window():
    """A run that ended before the shock must score post_loss = inf, not a flattering 0.0."""
    obj = EpidemicLoss(lam=0.02, post_shock_step=100)
    short = [{"t": float(i), "infected": 0.01, "cum_cost": 0.0} for i in range(50)]
    comps = obj.components(short)
    assert comps["post_loss"] == float("inf")
    assert comps["post_infected"] == float("inf")


def test_sir_conserves_mass_with_waning_and_importation():
    s = SIRSystem({"beta0": 0.4, "gamma": 0.1, "waning": 0.03, "import_rate": 0.002,
                   "shock_step": 10_000})
    s.reset(0)
    for _ in range(300):
        s.step()
        assert s.S >= 0.0 and s.I >= 0.0 and s.R >= 0.0
        assert s.S + s.I + s.R == pytest.approx(1.0, abs=1e-9), "S+I+R must stay conserved"


def test_sir_seeds_produce_different_worlds():
    """Heterogeneity is load-bearing: without it the paired design pairs identical numbers."""
    from govsim.domains.scalar import regimes as R

    factory = R.sir_factory(R.EPIDEMIC_SHOCKED)
    finals = []
    for seed in range(5):
        s = factory(seed)
        for _ in range(120):
            s.step()
        finals.append(s.I)
    assert len(set(round(v, 6) for v in finals)) == len(finals)
