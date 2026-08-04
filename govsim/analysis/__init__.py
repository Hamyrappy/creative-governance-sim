"""
govsim.analysis — the statistics layer that turns runs into defensible claims.

This is ``govsim/docs_gates/stats-protocol.md`` made executable: a **paired, shared-seed** design
(every regent runs the same world seeds), **bootstrap CI on the per-seed difference** (a claim
"X beats Y" requires the CI to exclude 0), **variance-aware selection** (``mean − λ·std``, never the
mean alone), and a **collapse detector** (early-terminated runs are counted worst-case, not dropped —
the Vending-Bench tail-event lesson). Domain-neutral: it operates on ``RunRecord`` scores/components,
never on domain internals, so it serves every domain unchanged.

It also owns **regime calibration** (``calibration.py``): the frozen/oracle reference pair that
turns a raw loss into a normalized regret, and that decides *before* an experiment runs whether the
regime has any adaptation headroom to compete for at all.
"""

from govsim.analysis.calibration import (
    CalibrationResult,
    PolicyFamily,
    calibrate,
    headroom,
    normalized_regret,
)
from govsim.analysis.stats import (
    bootstrap_ci,
    collapse_summary,
    compare,
    infer_lower_is_better,
    metric_by_seed,
    paired_diff,
    robust_score,
    variance_aware_select,
)

__all__ = [
    "CalibrationResult",
    "PolicyFamily",
    "calibrate",
    "headroom",
    "normalized_regret",
    "bootstrap_ci",
    "collapse_summary",
    "compare",
    "infer_lower_is_better",
    "metric_by_seed",
    "paired_diff",
    "robust_score",
    "variance_aware_select",
]
