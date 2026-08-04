"""
The experiment registry — "add an experiment = one function + one ``@register``".

Each registered factory returns a fully-wired :class:`govsim.core.Experiment` (system factory ×
regents × objectives × harness × schedule × seeds × horizon × the WHAT-first hypothesis). The CLI
(``python -m govsim run <name>``) looks experiments up here. Concrete experiments live in sibling
modules (e.g. ``scalar_experiments.py``) and are imported at the bottom so registration happens on
``import govsim.experiments``.
"""

from __future__ import annotations

from typing import Callable, Dict

from govsim.core.experiment import Experiment

ExperimentFactory = Callable[[], Experiment]
_REGISTRY: Dict[str, ExperimentFactory] = {}


def register(name: str) -> Callable[[ExperimentFactory], ExperimentFactory]:
    def deco(fn: ExperimentFactory) -> ExperimentFactory:
        if name in _REGISTRY:
            raise ValueError(f"experiment '{name}' is already registered")
        _REGISTRY[name] = fn
        return fn

    return deco


def get(name: str) -> Experiment:
    if name not in _REGISTRY:
        raise KeyError(f"unknown experiment '{name}'. Available: {available()}")
    return _REGISTRY[name]()


def available() -> list[str]:
    return sorted(_REGISTRY)


# Importing these modules registers their experiments (keep at the bottom to avoid a cycle).
from govsim.experiments import scalar_experiments  # noqa: E402,F401
from govsim.experiments import governance_experiments  # noqa: E402,F401
from govsim.experiments import economy_experiments  # noqa: E402,F401
