"""Does state recurrence change what episodic retrieval actually returns? A key-free test.

The lock-in result has a candidate explanation the paper reports as a hypothesis, not a finding:
episodic memory suppresses revision harder in the epidemic than in the monetary world because the
epidemic's states RECUR --- it settles toward an endemic equilibrium, so successive decisions are
taken in near-identical states, retrieval returns near-identical precedent, and repeating it exactly
is easy.

That story has three links: recurrence -> retrieval returns similar precedent -> the agent repeats
it. Only the last needs a language model. The middle one is a property of the retrieval component and
the world, and it can be measured for the price of a simulation.

So this probe tests the middle link across a recurrence sweep, using the SAME ``EpisodicMemory``
the experiments use and a scripted regent in place of the LLM. If retrieval similarity does not move
with recurrence, the hypothesis is dead and no expensive LLM sweep is warranted. If it does move, the
LLM sweep becomes worth its cost --- and this fixes the operating points it should use.

The dial is the plant's ``decay``. Near 1 the state wanders and rarely revisits; near 0 it snaps back
to equilibrium and every decision is taken in almost the same place. Everything else --- the
objective, the horizon, the decision cadence, the retrieval component, the policy --- is held fixed.

Run: ``uv run python scripts/recurrence_probe.py``
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from govsim.core.action import ActionRequest, ActionSpace  # noqa: E402
from govsim.core.harness import Outcome  # noqa: E402
from govsim.core.system import Observation  # noqa: E402
from govsim.domains.diagnostics.restraint import IatrogenicPlant  # noqa: E402
from govsim.domains.scalar import Lever, ScalarLeverInterface  # noqa: E402
from govsim.harness import EpisodicMemory  # noqa: E402

DECIDE_EVERY = 10
HORIZON = 200
LAW = "0.5 * (indicator - target_x)"  # fixed policy: the probe is about retrieval, not control


def _run(decay: float, seed: int) -> tuple[list[float], list[float]]:
    """Returns (recurrence distances, retrieval distances) for one run.

    recurrence: nearest-neighbour distance between decision states, standardized per run.
    retrieval : distance from the current state to the episodes ``EpisodicMemory`` actually returns,
                in the same standardized units, so the two are directly comparable.
    """
    # decay_bounds must be widened or the dial does not turn: the plant clips decay to (0.55, 0.92)
    # by default, so a sweep from 0.20 to 0.97 silently collapses to 0.55-0.92 and the endpoints
    # return IDENTICAL numbers. That is what the first run of this probe did, and the giveaway was
    # two pairs of identical rows rather than any error. Stability only needs decay < 1.
    system = IatrogenicPlant({"seed": seed, "decay": decay, "decay_sigma": 0.0,
                              "decay_bounds": (0.01, 0.995)})
    iface = ScalarLeverInterface([Lever("set_control_input", system.u_range, "current_u")])
    iface.apply([ActionRequest("regent:0", "set_control_input", {"expr": LAW})], system)

    mem = EpisodicMemory(k=4)
    space = ActionSpace(verbs=[], context_vars=[])
    states: list[dict[str, float]] = []
    retrieved: list[list[dict[str, float]]] = []

    for step in range(HORIZON):
        if step % DECIDE_EVERY == 0:
            view = Observation(vars={"indicator": system.indicator, "current_u": system.current_u},
                               scope="regent:0", t=step)
            scratch: dict = {}
            ranked = sorted(mem.episodes, key=lambda e: mem._distance(view.vars, e["state"]))
            retrieved.append([e["state"] for e in ranked[:mem.k]])
            states.append(dict(view.vars))
            reqs = [ActionRequest("regent:0", "set_control_input", {"expr": LAW})]
            mem.on_outcome(view, reqs, Outcome(requests=reqs, metrics=system.metrics()), scratch)
        system.step()

    keys = ["indicator", "current_u"]
    arr = np.array([[s[k] for k in keys] for s in states])
    mu, sd = arr.mean(0), arr.std(0) + 1e-9
    z = (arr - mu) / sd

    rec = []
    for i in range(len(z)):
        other = np.delete(z, i, axis=0)
        rec.append(float(np.sqrt(((other - z[i]) ** 2).sum(1)).min()))

    ret = []
    for i, eps in enumerate(retrieved):
        if not eps:
            continue
        e = (np.array([[s.get(k, 0.0) for k in keys] for s in eps]) - mu) / sd
        ret.append(float(np.sqrt(((e - z[i]) ** 2).sum(1)).mean()))
    return rec, ret


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--out", default="logs/recurrence_probe.json")
    args = ap.parse_args()

    # decay near 1 = the state wanders (low recurrence); near 0 = it snaps back (high recurrence).
    grid = [0.20, 0.40, 0.60, 0.75, 0.85, 0.92, 0.97]
    rows = []
    print(f"{'decay':>7}{'recurrence dist':>18}{'retrieval dist':>17}")
    for d in grid:
        recs, rets = [], []
        for s in range(args.seeds):
            r, t = _run(d, s)
            recs.append(st.fmean(r)); rets.append(st.fmean(t))
        rows.append({"decay": d, "recurrence": st.fmean(recs), "retrieval": st.fmean(rets)})
        print(f"{d:>7.2f}{rows[-1]['recurrence']:>18.4f}{rows[-1]['retrieval']:>17.4f}")

    x = np.array([r["recurrence"] for r in rows])
    y = np.array([r["retrieval"] for r in rows])
    corr = float(np.corrcoef(x, y)[0, 1])
    spread = (y.max() - y.min()) / (y.mean() + 1e-9)
    # The CORRELATION is near-definitional and must not be read as evidence: both quantities are
    # nearest-neighbour distances computed on the same standardized trajectory, so of course they
    # move together. The informative number is the SPREAD — how much retrieval distance can be moved
    # by turning the dial across its entire feasible range.
    #
    # The comparison that matters is against the effect it is supposed to explain: episodic memory
    # suppresses churn ninefold more in the epidemic (0.847 -> 0.082) than in the monetary world
    # (0.937 -> 0.726). A mediator that moves by a fifth cannot carry a ninefold difference.
    LOCKIN_RATIO = (0.847 / 0.082) / (0.937 / 0.726)   # ~8x, the thing needing explanation
    print(f"\ncorr(recurrence, retrieval distance) = {corr:+.3f}  "
          f"(near-definitional; both are nearest-neighbour distances — do not read as evidence)")
    print(f"retrieval distance varies by {100 * spread:.0f}% across the FULL range of the dial")
    print(f"the lock-in difference needing explanation is ~{LOCKIN_RATIO:.0f}x")
    verdict = (
        f"RECURRENCE IS RULED OUT as the sole mechanism: turning it across its entire feasible range "
        f"moves retrieved-precedent distance only {100 * spread:.0f}%, against a ~{LOCKIN_RATIO:.0f}x "
        f"difference in lock-in. The link is real and far too weak."
        if spread < 0.5 else
        f"recurrence remains viable: the dial moves retrieval distance {100 * spread:.0f}%, which is "
        f"the right order to matter. An LLM sweep over this dial is now worth its cost."
    )
    print(f"=> {verdict}")

    Path(args.out).write_text(json.dumps(
        {"rows": rows, "corr": corr, "spread": spread, "verdict": verdict}, indent=1), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
