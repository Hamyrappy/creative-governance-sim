"""
govsim.regents — the controllers that decide, beyond the trivial baselines in ``govsim.core.regent``.

All are domain-BLIND: they see an ``Observation`` + an ``ActionSpace`` and return
``ActionRequest``s; none imports a domain context class (the coupling that killed the old
``IntelligentLLMAgent``). Ships:
  - ``LLMRegent``    — emits a sandboxed expression per lever via an OpenAI-compatible client
                       (native tool-calling or JSON), with a 4-source prompt assembler.
  - ``PIDRegent``    — a tuned PD/PID control law (the "tuned controller" the creativity metric
                       must beat on un-tuned regimes).
  - ``LQRRegent``    — the analytic LQR ground-truth ceiling for the linear scalar plant.
"""

from govsim.regents.llm_regent import LLMRegent, default_prompt_assembler, parse_action_requests
from govsim.regents.baselines import PIDRegent, LQRRegent

__all__ = [
    "LLMRegent",
    "default_prompt_assembler",
    "parse_action_requests",
    "PIDRegent",
    "LQRRegent",
]
