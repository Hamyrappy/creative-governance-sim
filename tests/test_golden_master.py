"""
Golden-master regression test (doc-09 §7.1 / doc-08 §3.2).

NOT an attempt to reproduce the thesis numbers (the thesis used a live ``temp=0.5`` model + a
different config; reproducing that is best-effort, not a unit test). This is a **deterministic
regression baseline**: the ``ScriptedRegent`` (fixed expression, NO live LLM) on the linear
``CubicSystem`` with a fixed seed must reproduce a committed trajectory. It guards the whole
Runner → ActionInterface → System → re-eval pipeline against silent drift, and runs key-free on CI.

Regenerate (only on an intentional dynamics change) with::

    python -c "from tests.test_golden_master import _regenerate; _regenerate()"
"""

from __future__ import annotations

import json
from pathlib import Path

from govsim.core.runner import Runner
from govsim.experiments import get

_REF = Path(__file__).parent / "golden" / "cubic_stabilization_seed0.json"
_TOL = 1e-7  # PCG64 + IEEE floats are stable across numpy versions to well within this


def _run_seed0():
    exp = get("cubic_stabilization")
    exp.seeds = [0]
    return Runner().run(exp)[0]


def test_golden_master_trajectory_matches_reference():
    ref = json.loads(_REF.read_text())
    rec = _run_seed0()

    assert rec.horizon == ref["horizon"]
    assert rec.system_id == ref["system_id"]
    assert len(rec.metrics_series) == len(ref["current_x"])

    for i, (row, ref_x, ref_u) in enumerate(zip(rec.metrics_series, ref["current_x"], ref["current_u"])):
        assert abs(row["current_x"] - ref_x) < _TOL, f"current_x drift at step {i}"
        assert abs(row["current_u"] - ref_u) < _TOL, f"current_u drift at step {i}"

    assert abs(rec.score["regent:0"] - ref["score"]) < _TOL
    for key, ref_val in ref["components"].items():
        assert abs(rec.components["regent:0"][key] - ref_val) < _TOL, f"component '{key}' drift"


def _regenerate() -> None:  # pragma: no cover - maintenance helper
    rec = _run_seed0()
    ref = {
        "experiment": rec.experiment,
        "seed": rec.seed,
        "horizon": rec.horizon,
        "system_id": rec.system_id,
        "git_note": "deterministic regression baseline (ScriptedRegent, no live LLM) — doc-08 §3.2",
        "current_x": [round(r["current_x"], 9) for r in rec.metrics_series],
        "current_u": [round(r["current_u"], 9) for r in rec.metrics_series],
        "score": round(rec.score["regent:0"], 9),
        "components": {k: round(v, 9) for k, v in rec.components["regent:0"].items()},
    }
    _REF.write_text(json.dumps(ref, indent=2))
