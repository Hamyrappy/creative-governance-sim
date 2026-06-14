"""
Architectural invariants enforced as tests (doc-09 §6.4, §7.1, §8) — they run key-free on CI:

1. **no bare RNG** — every system owns a ``numpy.random.Generator`` (``self.rng``); nothing under
   ``govsim/core`` or ``govsim/domains`` draws from the module-global ``random`` / ``np.random``
   state (only ``np.random.default_rng`` / ``Generator`` are allowed). This is the HARD doc-08
   precondition for a sound ``clone()``/``rollout()`` fitness oracle.
2. **no domain noun in core** — ``govsim/core`` is domain-agnostic: no identifier (class/function/
   arg/attribute/import name) may be a domain noun (economy, tax, ledger, welfare, …). Domain
   coupling lives below the ``ActionInterface`` seam, in ``govsim/domains/*``. (Comments/docstrings
   may *reference* a domain as a counter-example; only code identifiers are checked.)
3. **gate docs present** — the WHAT-first spec docs exist and are non-empty ("BLOCK the phase if
   absent").
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent / "govsim"
_CORE = _ROOT / "core"
_DOMAINS = _ROOT / "domains"

# np.random / numpy.random members that ARE allowed (seeding a per-system Generator).
_ALLOWED_RANDOM_ATTRS = {"default_rng", "Generator", "SeedSequence", "PCG64", "BitGenerator"}


def _py_files(*roots: Path) -> list[Path]:
    return [p for root in roots for p in root.rglob("*.py")]


def _dotted(node: ast.Attribute) -> str:
    parts: list[str] = []
    cur: ast.AST = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    return ".".join(reversed(parts))


# --- 1. no bare module-global RNG -----------------------------------------------------------

def test_no_bare_rng_in_core_and_domains():
    offenders: list[str] = []
    for path in _py_files(_CORE, _DOMAINS):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                dotted = _dotted(node)
                if re.match(r"^(random|np\.random|numpy\.random)\.", dotted):
                    last = dotted.rsplit(".", 1)[1]
                    if last not in _ALLOWED_RANDOM_ATTRS:
                        offenders.append(f"{path.relative_to(_ROOT.parent)}:{node.lineno}: {dotted}")
            if isinstance(node, ast.ImportFrom) and node.module in {"random", "numpy.random"}:
                for alias in node.names:
                    if alias.name not in _ALLOWED_RANDOM_ATTRS:
                        offenders.append(f"{path.relative_to(_ROOT.parent)}:{node.lineno}: from {node.module} import {alias.name}")
    assert not offenders, "bare module-global RNG (use self.rng = np.random.default_rng(seed)):\n" + "\n".join(offenders)


# --- 2. no domain noun leaks into core ------------------------------------------------------

_DOMAIN_WORDS = {
    "economy", "economic", "economics", "tax", "taxes", "ledger", "treasury", "welfare",
    "gini", "firm", "firms", "market", "markets", "gdp", "mandel", "fiscal", "monetary",
    "inflation", "household", "households", "polity", "sfc", "chancery", "mediator",
}


def _words(identifier: str) -> set[str]:
    """Split snake_case + camelCase into lowercase words (so 'syntax' never matches 'tax')."""
    parts: list[str] = []
    for chunk in identifier.split("_"):
        parts.extend(re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+", chunk) or [chunk])
    return {p.lower() for p in parts if p}


def test_no_domain_nouns_in_core():
    offenders: list[str] = []
    for path in _py_files(_CORE):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.append(node.name)
            elif isinstance(node, ast.Name):
                names.append(node.id)
            elif isinstance(node, ast.arg):
                names.append(node.arg)
            elif isinstance(node, ast.Attribute):
                names.append(node.attr)
            elif isinstance(node, ast.keyword) and node.arg:
                names.append(node.arg)
            elif isinstance(node, ast.alias):
                names.append((node.asname or node.name).split(".")[0])
            for name in names:
                bad = _words(name) & _DOMAIN_WORDS
                if bad:
                    offenders.append(f"{path.relative_to(_ROOT.parent)}:{getattr(node, 'lineno', '?')}: '{name}' -> {sorted(bad)}")
    assert not offenders, "domain noun in govsim/core identifiers (move it to govsim/domains/*):\n" + "\n".join(offenders)


# --- 3. WHAT-first gate docs present --------------------------------------------------------

def test_gate_docs_present_and_nonempty():
    gates = _ROOT / "docs_gates"
    required = ["hypotheses.md", "objectives.md", "creativity-metric.md", "stats-protocol.md", "decisions.md", "STATUS.md"]
    missing = [name for name in required if not (gates / name).exists() or (gates / name).stat().st_size < 50]
    assert not missing, f"WHAT-first gate docs missing/empty (BLOCK the phase if absent): {missing}"
