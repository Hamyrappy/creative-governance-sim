"""
Tests for the experiment spine: the WHAT-first gate, the paired-seed Runner loop, multi-regent
N=1 reduction, and the ResultStore (insert / query / series round-trip / plot).
"""

from __future__ import annotations

import pytest

from govsim.core import (
    CreativityMetric,
    Experiment,
    Hypothesis,
    Runner,
    ResultStore,
    EveryN,
    ScriptedRegent,
    StaticRegent,
)
from govsim.domains.scalar import CubicSystem, Lever, ScalarLeverInterface, StabilizationLoss


def _cubic_factory(seed: int) -> CubicSystem:
    sys = CubicSystem({"param_A": 0.95, "param_B": 0.5, "sigma_epsilon": 0.1, "u_range": (-2.0, 2.0)})
    sys.reset(seed)
    return sys


def _iface():
    return ScalarLeverInterface([Lever("set_control_input", (-2.0, 2.0), "current_u")])


def _hypothesis():
    return Hypothesis(
        id="H0-smoke",
        claim="a proportional regent keeps x near the target",
        baseline="no-control (StaticRegent)",
        primary_metric="mse",
    )


def _experiment(regent=None, seeds=(0, 1, 2)) -> Experiment:
    regent = regent or ScriptedRegent(verb="set_control_input", expr="-0.9 * current_x")
    return Experiment(
        name="cubic_smoke",
        system_factory=_cubic_factory,
        action_interface=_iface(),
        regents={"regent:0": regent},
        objectives={"regent:0": StabilizationLoss(lam=0.1)},
        schedule=EveryN(10),
        seeds=list(seeds),
        horizon=60,
        hypothesis=_hypothesis(),
        creativity_metric=CreativityMetric(name="none", kind="generalization_gap"),
    )


def test_runner_rejects_experiment_without_baseline():
    exp = _experiment()
    exp.hypothesis = Hypothesis(id="bad", claim="something", baseline="", primary_metric="mse")
    with pytest.raises(ValueError, match="WHAT-first gate"):
        Runner().run(exp)


def test_runner_produces_one_record_per_seed():
    records = Runner().run(_experiment(seeds=(0, 1, 2)))
    assert len(records) == 3
    assert [r.seed for r in records] == [0, 1, 2]
    for r in records:
        assert len(r.metrics_series) == 60
        assert "regent:0" in r.score
        assert "mse" in r.components["regent:0"]


def test_runner_is_deterministic_per_seed():
    a = Runner().run(_experiment(seeds=(42,)))[0]
    b = Runner().run(_experiment(seeds=(42,)))[0]
    assert a.metrics_series == b.metrics_series
    assert a.score == b.score


def test_control_beats_no_control_on_mse():
    controlled = Runner().run(_experiment(seeds=(7,)))[0]
    passive = Runner().run(_experiment(regent=StaticRegent(), seeds=(7,)))[0]
    assert controlled.components["regent:0"]["mse"] < passive.components["regent:0"]["mse"]


def test_record_carries_provenance():
    rec = Runner().run(_experiment(seeds=(0,)))[0]
    assert rec.system_id == "CubicSystem"
    assert rec.action_interface_id == "ScalarLeverInterface"
    assert rec.regent_specs["regent:0"]["type"] == "ScriptedRegent"
    assert rec.objective_ids["regent:0"] == "StabilizationLoss"
    assert rec.hypothesis_id == "H0-smoke"


# --- ResultStore -----------------------------------------------------------------------------

def test_result_store_insert_query_series(tmp_path):
    store = ResultStore(tmp_path / "runs")
    runner = Runner(result_store=store)
    runner.run(_experiment(seeds=(0, 1)))

    rows = store.query(experiment="cubic_smoke")
    assert len(rows) == 2
    assert rows[0]["system_id"] == "CubicSystem"
    assert isinstance(rows[0]["score"], dict)  # JSON column parsed back

    seeded = store.query(experiment="cubic_smoke", seed=1)
    assert len(seeded) == 1

    series = store.load_series(rows[0]["run_id"])
    assert len(series) == 60 and "current_x" in series[0]
    store.close()


def test_result_store_plot(tmp_path):
    store = ResultStore(tmp_path / "runs")
    runner = Runner(result_store=store)
    runner.run(_experiment(seeds=(0,)))
    run_id = store.query(experiment="cubic_smoke")[0]["run_id"]
    out = store.plot(run_id, keys=["current_x", "current_u"])
    assert out.exists() and out.stat().st_size > 0
    store.close()
