"""Every citation the papers actually use must carry a verification note.

`paper/refs.bib` is verified-only by policy, and the policy was not enforced by anything. The cost
of that showed up as a REAL defect: `ye2026trails` credited "Junchen Ye, Yuxuan Cao, Canyu Chen" for
a paper actually written by Jinyi Ye, Lei Cao and Ding Chen — three fabricated names on a real work,
sitting in the bibliography of two manuscripts. A referee who spot-checks one reference and finds
that stops trusting the other sixty-seven.

The rule this file enforces: an entry CITED by either manuscript must record, in a `note` field, that
someone read the actual source. Uncited entries are not checked — they hurt nobody until used, and
the check fires the moment one is cited.

`GRANDFATHERED` is deliberately an explicit list rather than a date cutoff or a silent skip. It is
visible debt: it can only shrink, every removal is a real verification, and a reviewer can see
exactly how much of the bibliography is unaudited.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
BIB = REPO / "paper" / "refs.bib"
PAPERS = (REPO / "paper" / "main.tex", REPO / "paper" / "social.tex")

ENTRY_RE = re.compile(r"@\w+\{([^,]+),")
CITE_RE = re.compile(r"\\cite[a-zA-Z]*\s*(?:\[[^\]]*\])*\{([^}]+)\}")
VERIFIED_RE = re.compile(r"note\s*=\s*\{[^{}]*[Vv]erified", re.S)

#: Cited entries that predate the verified-only policy and have NOT yet been checked against their
#: source. This list may only shrink. Do not add to it — verify the entry instead.
#:
#: Pre-1990 classics are here because their metadata is not plausibly hallucinated and is checkable
#: from any library catalogue; the 2023-2026 preprints are the ones that carry real risk, and they
#: are the ones to retire first.
GRANDFATHERED = {
    # canonical, low-risk
    "lucas1976critique", "kermack1927contribution", "goodhart1984monetary",
    # tooling / prompt-optimization line
    "yang2023opro", "fernando2023promptbreeder", "khattab2023dspy", "yuksekgonul2024textgrad",
    "zhang2024aflow", "novikov2025alphaevolve",
    # harness line (the four load-bearing ones were verified 2026-08-05 and are NOT here)
    "kim2026interplay", "lin2026harnessupdating", "xu2026lifeharness",
    # code-as-policy line
    "liang2022codeaspolicies", "ma2023eureka", "maher2025llmpc", "bosio2025combining",
    "guo2026codeevolution",
    # LLM-society line
    "vezhnevets2023concordia", "li2023econagent", "piatti2024govsim", "piao2025agentsociety",
    "backlund2025vendingbench", "wang2025taxagent", "zhong2026separating", "pal2026evolutionarily",
    "decurto2026narratives", "run2026icrlsurvey", "li2026statistical", "sarangi2026ease",
    "luo2026preconditions", "li2026policypractice",
}


def _entries() -> dict[str, str]:
    text = BIB.read_text(encoding="utf-8")
    out: dict[str, str] = {}
    for chunk in re.split(r"\n(?=@)", text):
        m = ENTRY_RE.match(chunk.strip())
        if m:
            out[m.group(1).strip()] = chunk
    return out


def _cited() -> set[str]:
    keys: set[str] = set()
    for p in PAPERS:
        for m in CITE_RE.finditer(p.read_text(encoding="utf-8")):
            keys.update(k.strip() for k in m.group(1).split(",") if k.strip())
    return keys


def test_every_cited_entry_is_verified_or_explicitly_grandfathered():
    entries, cited = _entries(), _cited()
    unverified = sorted(
        k for k in cited & set(entries)
        if not VERIFIED_RE.search(entries[k]) and k not in GRANDFATHERED
    )
    assert not unverified, (
        "these entries are cited but carry no verification note:\n  "
        + "\n  ".join(unverified)
        + "\n\nVerify each against its real source and record what you read in a `note` field. "
          "Do NOT add it to GRANDFATHERED."
    )


def test_every_cited_key_exists_in_the_bibliography():
    """A missing key renders as [?] and is easy to miss in a 29-page PDF."""
    entries, cited = _entries(), _cited()
    missing = sorted(cited - set(entries))
    assert not missing, f"cited but absent from refs.bib: {missing}"


def test_grandfathered_list_contains_no_stale_names():
    """Once an entry is verified or deleted it must leave the list, or the list stops meaning
    anything and quietly re-authorises a name nobody checked."""
    entries = _entries()
    ghosts = sorted(k for k in GRANDFATHERED if k not in entries)
    assert not ghosts, f"GRANDFATHERED names no longer in refs.bib: {ghosts}"

    verified_but_listed = sorted(
        k for k in GRANDFATHERED if k in entries and VERIFIED_RE.search(entries[k])
    )
    assert not verified_but_listed, (
        f"these are verified and must be removed from GRANDFATHERED: {verified_but_listed}"
    )


def test_the_debt_does_not_grow():
    """A ratchet. If this fails because the number went UP, something was added unverified."""
    cited = _cited() & set(_entries())
    assert len(GRANDFATHERED & cited) <= 33, (
        "unverified-but-cited count rose above its recorded high-water mark; verify the new entry "
        "rather than raising this bound"
    )


@pytest.mark.parametrize("key,fragment", [
    # The specific defect that motivated this file: three fabricated author names on a real paper.
    ("ye2026trails", "Ye, Jinyi"),
    ("ye2026trails", "physics.soc-ph"),
])
def test_the_corrected_metadata_stays_corrected(key, fragment):
    entries = _entries()
    assert key in entries, f"{key} vanished from refs.bib"
    assert fragment in entries[key], f"{key} no longer contains {fragment!r}"
