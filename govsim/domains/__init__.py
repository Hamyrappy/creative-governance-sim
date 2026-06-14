"""
govsim.domains — the domain plugins.

A domain provides a concrete ``ActionInterface`` (what a regent may do + the domain
invariants), one or more ``System``s, and ``Objective``s. The core (``govsim.core``) never
imports a domain; domains import the core. This is the boundary the grand plan's leakage test
enforces (zero domain nouns in ``govsim.core``).

Shipped:
  - ``scalar``  — ``ScalarLeverInterface`` + bounded-lever dynamical systems (cubic, SIR,
    company). NO ledger. Two non-economic systems run here to prove the core is domain-general.
  - ``economy`` — (later) ``EconomyActionInterface``: the conserved Effect/Ledger/Mediator
    "Chancery" of doc-07, a Phase-4 sibling of the scalar interface.
"""
