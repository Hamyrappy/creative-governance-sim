"""
Turn a ResultStore into the paper's tables: normalized regret, factorial attribution, contrasts.

    uv run python scripts/analyze_matrix.py --store logs/runs --model gemini-3.5-flash-lite

Everything here is paired on the world seed. Each arm ran the identical seeds, so per-seed
differencing removes the world variance — which in this regime is roughly as large as the effects
being measured, so an unpaired comparison would be mostly noise about the epidemic rather than
about the regent.

Three outputs, in the order a reader needs them:

1. **Arm table** — post-shock loss and normalized regret R against the calibrated anchors, where
   R=0 is the clairvoyant oracle and R=1 is the frozen pre-shock rule.
2. **Factorial attribution** — main effects *and* interactions of the harness components, Holm
   corrected across the whole family, because eight cells and seven effect terms is a family and
   reporting each at α=0.05 buys false positives at that many looks.
3. **Named contrasts** — each arm against frozen (did adaptation happen?) and against
   budget-matched OPRO (is the harness doing anything a plain score-optimizer would not?).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from govsim.analysis import compare, normalized_regret
from govsim.analysis.stats import (
    bootstrap_p, factorial_effects, holm_bonferroni, minimum_detectable_effect,
)
from govsim.core.result_store import ResultStore

FACTORS = ("trace", "outcome", "memory")
METRIC = "loss"  # FULL horizon: a post-break-only metric rewards passivity (preregistration §8)
REGENT = "regent:0"


def arm_name(cell: tuple[bool, ...]) -> str:
    on = [f for f, b in zip(FACTORS, cell) if b]
    return "epidemic_llm_" + ("_".join(on) if on else "bare")


def load_by_seed(stores: list[ResultStore], experiment: str, model: str | None) -> dict[int, float]:
    """``{seed: metric}`` for an arm, keeping the LATEST run per seed (re-runs supersede).

    Takes several stores because models run as separate processes against separate sqlite files —
    one shared file would have them contending for the write lock for hours.
    """
    out: dict[int, float] = {}
    for store in stores:
        for row in store.query(experiment=experiment):
            if model:
                spec = (row.get("regent_specs") or {}).get(REGENT, {})
                row_model = spec.get("model")
                if row_model is not None and row_model != model:
                    continue
            comps = (row.get("components") or {}).get(REGENT, {})
            if METRIC in comps:
                out[int(row["seed"])] = float(comps[METRIC])  # ORDER BY run_id ⇒ last write wins
    return out


def available_models(stores: list[ResultStore]) -> list[str]:
    """Every LLM model id that appears in any store (for the cross-model replication table)."""
    seen: set[str] = set()
    for store in stores:
        for row in store.query():
            m = ((row.get("regent_specs") or {}).get(REGENT, {}) or {}).get("model")
            if m:
                seen.add(str(m))
    return sorted(seen)


def no_action_rate(stores: list[ResultStore], experiment: str, model: str | None) -> tuple[int, int]:
    """``(calls that produced no parseable action, total calls)`` for an arm.

    This is a validity check, not a curiosity, and it is run before any result is believed. A model
    that reasons before acting can spend its whole output budget on the reasoning and be truncated
    before it emits the tool call. When that happens the decision is a silent no-op: the previously
    installed law simply stays in force, and the arm quietly becomes "sticky policy" rather than the
    treatment it is labelled as.

    The failure is *correlated with the treatment*, which is what makes it lethal here. A harness
    channel lengthens the prompt and invites longer deliberation, so exactly the arms carrying more
    information are the ones most likely to run out of budget. An ablation run this way measures
    truncation and reports it as information.

    We hit precisely this: at ``max_tokens=1500`` the outcome-feedback arm failed to act on 32.5% of
    its decisions while the no-harness arm failed on 0%.
    """
    empty = total = 0
    for store in stores:
        for row in store.query(experiment=experiment):
            if model:
                spec = (row.get("regent_specs") or {}).get(REGENT, {})
                if spec.get("model") not in (None, model):
                    continue
            path = row.get("llm_io_path")
            if not path or not Path(path).exists():
                continue
            for call in json.loads(Path(path).read_text(encoding="utf-8")):
                if call.get("regent_id") == "critic":
                    continue
                total += 1
                if not _call_produced_an_action(call):
                    empty += 1
    return empty, total


def _call_produced_an_action(call: dict) -> bool:
    """Did this recorded call yield an action the Runner would have applied?

    Must use the SAME parser the Runner uses. An earlier version counted only native
    ``tool_calls`` and therefore reported 100% no-action for a model that answers through the JSON
    fallback — which is a supported path, exercised by smaller models that cannot emit tool calls
    reliably. The gate would have disqualified a perfectly functional arm, which is the most
    expensive kind of false positive a validity gate can have: it discards real data and looks
    rigorous doing it.
    """
    from govsim.core.action import ActionSpace, VerbSpec
    from govsim.core.llm.client import LLMResponse
    from govsim.regents.llm_regent import parse_action_requests

    resp = LLMResponse(text=call.get("response_text") or "",
                       tool_calls=list(call.get("tool_calls") or []))
    # A permissive space: we are asking "did the model produce SOMETHING parseable", not "was it
    # valid for this world" — the interface rejects invalid verbs separately and that rejection is
    # itself surfaced through the trace channel.
    verbs = {tc.get("name") for tc in (call.get("tool_calls") or []) if tc.get("name")}
    verbs |= {"set_lockdown", "set_vaccination", "set_control_input"}
    space = ActionSpace(verbs=[VerbSpec(name=v) for v in sorted(verbs)], context_vars=[])
    return bool(parse_action_requests(resp, space, "regent:0"))


def collapse_rate(stores: list[ResultStore], experiment: str, model: str | None) -> tuple[int, int]:
    """``(decisions that collapsed into pure deliberation, total decisions)`` for an arm.

    Distinct from :func:`no_action_rate`, and reported separately even though collapse implies
    no-action. The no-action rate says an arm is contaminated; this says *how*, and the two have
    opposite remedies — plain truncation is a budget problem a larger ``max_tokens`` fixes, while
    collapse happens AT a large budget and a larger one only buys more loop.

    It is also a RESULT rather than only a check. Measured on ``gemma-4-31b-it``, the rate tracks one
    factor and one only:

        outcome present   7.8% / 5.8% / 3.8%
        outcome absent    0.2% / 0.2% / 0.0% / 0.0%

    A channel can degrade an agent without misinforming it, simply by inviting unbounded
    second-guessing — which is a harness-design finding, not a bug in the harness.
    """
    from govsim.core.llm.client import response_is_collapsed

    collapsed = total = 0
    for store in stores:
        for row in store.query(experiment=experiment):
            if model:
                spec = (row.get("regent_specs") or {}).get(REGENT, {})
                if spec.get("model") not in (None, model):
                    continue
            path = row.get("llm_io_path")
            if not path or not Path(path).exists():
                continue
            for call in json.loads(Path(path).read_text(encoding="utf-8")):
                if call.get("regent_id") == "critic":
                    continue
                total += 1
                if response_is_collapsed(call.get("response_text") or "",
                                         call.get("tool_calls") or [],
                                         call.get("usage") or {}):
                    collapsed += 1
    return collapsed, total


def enacted_policies(stores: list[ResultStore], experiment: str, model: str | None) -> list[str]:
    """Every policy the agent actually installed in this arm, in order, as ``verb=expr`` strings.

    The raw material for the responsiveness gate. We read what was *enacted* rather than what was
    said, because the question is whether the harness changed the agent's behaviour, and only the
    installed law reaches the world.
    """
    from govsim.core.action import ActionSpace, VerbSpec
    from govsim.core.llm.client import LLMResponse
    from govsim.regents.llm_regent import parse_action_requests

    space = ActionSpace(
        verbs=[VerbSpec(name="set_lockdown"), VerbSpec(name="set_vaccination"),
               VerbSpec(name="set_control_input")],
        context_vars=[],
    )
    out: list[str] = []
    for store in stores:
        for row in store.query(experiment=experiment):
            if model:
                spec = (row.get("regent_specs") or {}).get(REGENT, {})
                if spec.get("model") not in (None, model):
                    continue
            path = row.get("llm_io_path")
            if not path or not Path(path).exists():
                continue
            for call in json.loads(Path(path).read_text(encoding="utf-8")):
                if call.get("regent_id") == "critic":
                    continue
                resp = LLMResponse(text=call.get("response_text") or "",
                                   tool_calls=list(call.get("tool_calls") or []))
                for req in parse_action_requests(resp, space, REGENT):
                    out.append(f"{req.verb}={req.payload.get('expr')}")
    return out


#: Below this share of altered prompts, an arm-level responsiveness verdict is not identifiable: a
#: channel that rarely speaks cannot move an arm mean, so a small total-variation distance says
#: nothing about whether the model reads it. Set at 20% because the failure it guards against was
#: measured at 3% and the clear cases in hand sit at 95-100%.
_LIVENESS_FLOOR = 0.20


def _prompts_by_decision(stores: list[ResultStore], experiment: str,
                         model: str | None) -> dict[tuple[int, int], str]:
    """``{(seed, step): prompt text}`` for an arm, so arms can be compared decision-by-decision.

    Keyed rather than positional: a positional zip silently compares seed 3's step 7 against seed
    4's step 2 the moment one arm has a missing record, and reports the resulting garbage as a
    difference.
    """
    out: dict[tuple[int, int], str] = {}
    for store in stores:
        for row in store.query(experiment=experiment):
            if model:
                spec = (row.get("regent_specs") or {}).get(REGENT, {})
                if spec.get("model") not in (None, model):
                    continue
            path = row.get("llm_io_path")
            if not path or not Path(path).exists():
                continue
            seed = int(row.get("seed", -1))
            for call in json.loads(Path(path).read_text(encoding="utf-8")):
                if call.get("regent_id") == "critic":
                    continue
                text = "\n".join(
                    (m.get("content") or "") for m in (call.get("messages") or [])
                    if isinstance(m, dict)
                )
                out[(seed, int(call.get("step", -1)))] = text
    return out


def _responsiveness_report(stores: list[ResultStore], cells: dict[str, str],
                           model: str | None) -> list[str]:
    """Gate 5: did the harness change BEHAVIOUR, not merely the prompt?

    The four earlier gates all pass on a model that reads nothing. Channel liveness confirms the
    channel injected; anchor consistency confirms the references are current; the no-action gate
    confirms the agent acted; freshness confirms the code is current. A model can clear all four and
    still emit one constant on every decision, in which case every arm is the same experiment and
    the flat table is a property of the subject, not of the harness.

    We measured exactly this on a 0.8B model: 400/400 decisions were ``set_lockdown=0.5``, a constant
    that ignores the epidemic entirely, and adding episodic memory moved it to 378/400. The losses
    agreed to four decimal places across six arms. Reported as "no harness effect", that would have
    been a false null about harnesses instead of a true statement about the model.

    Returns human-readable lines; the caller decides whether to fail.
    """
    from govsim.analysis.stats import policy_responsiveness

    lines: list[str] = []
    base = enacted_policies(stores, cells["bare"], model) if "bare" in cells else []
    if not base:
        return ["  responsiveness: no baseline arm on record — cannot judge"]
    base_prompts = _prompts_by_decision(stores, cells["bare"], model)
    for label, exp in cells.items():
        if label == "bare":
            continue
        # Stage 1: did the channel reach the prompt at all? Measured by comparing the actual
        # prompts decision-by-decision rather than by grepping for a header string, which goes
        # stale the moment a component's wording changes.
        theirs = _prompts_by_decision(stores, exp, model)
        shared = set(base_prompts) & set(theirs)
        moved_keys = {k for k in shared if base_prompts[k] != theirs[k]}
        live = len(moved_keys) / len(shared) if shared else 0.0
        if 0.01 <= live < _LIVENESS_FLOOR:
            # A channel that speaks on a small minority of decisions CANNOT move the aggregate
            # policy distribution much, however attentive the model is, so a low total-variation
            # distance here is arithmetic rather than evidence. Measured: a trace channel live on 3%
            # of prompts produced TV=0.015 and was reported DEAF — a verdict about the model drawn
            # from a fact about how rarely the channel had anything to say.
            lines.append(f"  THIN   {label:<12} the channel altered only {100 * live:.0f}% of "
                         f"prompts; too rarely to test responsiveness at the arm level. Judge this "
                         f"contrast on the affected decisions, not on the arm mean.")
            continue
        if live < 0.01:
            # NOT deafness. TraceFeedback, for instance, reports rejected actions; when the agent
            # emits only valid actions it has nothing to say, and an arm whose prompt never changed
            # SHOULD score identically to bare. Calling that a null about the component would be
            # backwards — the component was never on trial.
            lines.append(f"  SILENT {label:<12} the channel altered {100 * live:.0f}% of prompts; "
                         f"it never fired, so this arm is a duplicate of bare by construction")
            continue
        # Stage 2: the channel spoke. Did the agent's enacted behaviour change?
        rep = policy_responsiveness(base, enacted_policies(stores, exp, model))
        mark = "ok    " if rep["responsive"] else "DEAF  "
        lines.append(f"  {mark}{label:<12} channel live on {100 * live:.0f}% of prompts, "
                     f"TV={rep['tv_distance']:.3f}, top-policy-share={100 * rep['top_share']:.0f}%"
                     + (f"\n         <- {rep['reason']}" if rep["reason"] else ""))
    return lines


#: Files whose content determines what a run MEANS. A record produced before any of these last
#: changed is measuring a different experiment, however plausible its numbers look.
_SEMANTIC_FILES = (
    "govsim/domains/scalar/objectives.py",   # what "loss" is, and what the outcome channel reports
    "govsim/domains/scalar/systems.py",      # the worlds themselves
    "govsim/domains/scalar/regimes.py",      # the pinned configs
    "govsim/harness/components.py",          # what each channel puts in the prompt
    "govsim/regents/llm_regent.py",          # the prompt assembler, incl. the mandate
    "govsim/core/runner.py",                 # the decision loop and the realized-score window
)


def _commit_time(rev: str) -> int | None:
    out = subprocess.run(["git", "show", "-s", "--format=%ct", rev],
                         capture_output=True, text=True, cwd=ROOT)
    try:
        return int(out.stdout.strip())
    except ValueError:
        return None


def _runs_are_fresh(stores: list[ResultStore]) -> bool:
    """Were the stored runs produced by the CURRENT semantics?

    This gate exists because of the failure it is named after. Three bugs were fixed — an outcome
    channel that reported a clock, an episodic memory that did the same, and a prompt that never
    told the regent its objective — and the paper's tables kept reporting the runs made *before* the
    fixes, because the analysis artifact was simply never regenerated. Nothing complained: the
    numbers were plausible, the tables typeset, and the only thing wrong was that they described a
    pipeline that no longer existed.

    Every RunRecord already carries the git commit it was produced at, so this is cheap: compare it
    against the last commit that touched any file determining what a run means.
    """
    newest = 0
    newest_file = ""
    for f in _SEMANTIC_FILES:
        out = subprocess.run(["git", "log", "-1", "--format=%ct", "--", f],
                             capture_output=True, text=True, cwd=ROOT)
        try:
            ts = int(out.stdout.strip())
        except ValueError:
            continue
        if ts > newest:
            newest, newest_file = ts, f
    if not newest:
        return True

    stale: list[str] = []
    for store in stores:
        for row in store.query():
            rev = row.get("git_commit")
            if not rev or rev == "unknown":
                continue
            ts = _commit_time(rev)
            if ts is not None and ts < newest:
                stale.append(f"{row['experiment']} (seed {row['seed']}) @ {rev}")
    if stale:
        uniq = sorted(set(s.split(" (")[0] for s in stale))
        print(f"\n[!!] {len(stale)} stored run(s) predate the last change to {newest_file}. "
              f"Affected arms: {', '.join(uniq[:8])}{' …' if len(uniq) > 8 else ''}.\n"
              f"     Those runs were produced under different semantics — a different loss, a "
              f"different prompt, or a different channel — so a table built from them describes a "
              f"pipeline that no longer exists. Re-run them.", file=sys.stderr)
        return False
    return True


def _anchors_match_calibration(stores: list[ResultStore]) -> bool:
    """Do the stored reference runs enact the laws the current calibration artifact specifies?

    Compares each anchor arm's persisted ``regent_specs`` against
    ``docs_gates/calibration.json``. A mismatch means the store predates a recalibration and every
    ratio computed from it is anchored to a policy that no longer exists.
    """
    from govsim.domains.scalar import regimes as R

    calib = R.calibration().get("epidemic") or {}
    if not calib:
        return True  # nothing to check against; the calibration gate elsewhere will complain
    want = {
        "epidemic_frozen": set((calib.get("frozen", {}).get("laws") or {}).values()),
        "epidemic_best_fixed": set((calib.get("best_fixed", {}).get("laws") or {}).values()),
    }
    ok = True
    for arm, expected in want.items():
        if not expected:
            continue
        for store in stores:
            # The LATEST row per seed, exactly as ``load_by_seed`` does — a re-run supersedes.
            # Reading an arbitrary row instead was a real defect in this gate: after re-running the
            # reference arms to repair a store, the gate kept reporting the SUPERSEDED row and the
            # store could never be made to pass. The dangerous version of the same bug is quieter —
            # it would validate the anchors against a policy that is no longer the one the losses
            # were produced by, which is the exact failure this gate exists to catch.
            latest: dict[int, dict] = {}
            for row in store.query(experiment=arm):
                latest[int(row.get("seed", -1))] = row  # ORDER BY run_id ⇒ last write wins
            for row in latest.values():
                spec = (row.get("regent_specs") or {}).get(REGENT, {})
                # ScriptedRegent persists `expr`; MultiScriptedRegent persists `laws`. Accept either
                # shape, and treat a row carrying NEITHER as a failure rather than a pass: a record
                # that does not say which policy it enacted cannot be checked, and silently passing
                # such rows is how this gate spent a day being vacuous.
                if "laws" in spec:
                    got = set((spec.get("laws") or {}).values())
                elif "expr" in spec:
                    got = {spec["expr"]}
                else:
                    print(f"  anchor UNCHECKABLE in {arm} (seed {row['seed']}): the stored "
                          f"regent_spec records no policy ({sorted(spec)}). Re-run the reference "
                          f"arms with a build that persists it.", file=sys.stderr)
                    ok = False
                    break
                if not got <= expected:
                    print(f"  anchor mismatch in {arm} (seed {row['seed']}): stored "
                          f"{sorted(got)!r} is not within the calibrated {sorted(expected)!r}",
                          file=sys.stderr)
                    ok = False
                break  # one row per arm is enough to detect a stale store
    return ok


def fmt(v: float | None, w: int = 9, p: int = 4) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return f"{'—':>{w}}"
    return f"{v:>{w}.{p}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", nargs="+", default=["logs/runs"],
                    help="one or more ResultStore roots (models run into separate stores)")
    ap.add_argument("--model", default=None, help="restrict LLM arms to this model id")
    ap.add_argument("--cross-model", action="store_true",
                    help="also print the per-model replication table")
    ap.add_argument("--allow-stale", action="store_true",
                    help="analyse runs that predate the current semantics (they describe a "
                         "pipeline that no longer exists; for forensics only)")
    ap.add_argument("--json", default=None, help="write the full analysis here")
    args = ap.parse_args()

    stores = [ResultStore(p) for p in args.store]
    store = stores  # every loader takes the list
    # R is anchored on the NON-ADAPTIVE ceiling and the clairvoyant ADAPTOR, so R<1 means
    # "did better than any fixed law could have, in hindsight" rather than the much weaker
    # "did better than the stale rule", which passivity alone can achieve.
    frozen = load_by_seed(store, "epidemic_best_fixed", None)
    oracle = load_by_seed(store, "epidemic_switching", None)
    stale = load_by_seed(store, "epidemic_frozen", None)
    if not frozen or not oracle:
        print("missing calibrated anchors (epidemic_best_fixed / epidemic_switching) in the "
              "store; run `scripts/run_matrix.py --arms epidemic-refs` first", file=sys.stderr)
        return 1

    # ---- 0. ANCHOR GATE: do the stored reference runs match the current calibration? -----------
    # Every normalized regret is a ratio against these two rows. If the store holds anchors from a
    # previous calibration the whole scale is silently wrong, and nothing downstream notices: the
    # numbers stay plausible and only their meaning changes. This bit us — a store retained a
    # single-lever best_fixed law that no longer existed, which moved R from 0.81 to 1.32 and
    # flipped the sign of the headline contrast.
    if not _anchors_match_calibration(store):
        print("\n[!!] STORED ANCHORS DISAGREE WITH docs_gates/calibration.json. Every R below would "
              "be scaled against a superseded reference. Re-run "
              "`scripts/run_matrix.py --arms epidemic-refs` into this store first.", file=sys.stderr)
        return 2

    cells = {}
    for bits in range(2 ** len(FACTORS)):
        cell = tuple(bool(bits >> i & 1) for i in range(len(FACTORS)))
        by_seed = load_by_seed(store, arm_name(cell), args.model)
        if by_seed:
            cells[cell] = by_seed

    extra_arms = {name: load_by_seed(store, name, args.model)
                  for name in ("epidemic_opro", "epidemic_llm_critic")}
    extra_arms = {k: v for k, v in extra_arms.items() if v}

    # ---- 1. arm table ------------------------------------------------------------------------
    fm, om = statistics.fmean(frozen.values()), statistics.fmean(oracle.values())
    print(f"\n=== arms (metric={METRIC}, model={args.model or 'any'}) ===")
    print(f"{'arm':<38} {'n':>3} {'mean':>9} {'sd':>8} {'R':>7}   R: 0=oracle, 1=frozen")
    print(f"{'best_fixed (R=1: non-adaptive ceiling)':<38} {len(frozen):>3} {fmt(fm)} "
          f"{fmt(statistics.pstdev(frozen.values()) if len(frozen) > 1 else 0.0, 8)} {1.0:>7.3f}")
    print(f"{'switching (R=0: clairvoyant adaptor)':<38} {len(oracle):>3} {fmt(om)} "
          f"{fmt(statistics.pstdev(oracle.values()) if len(oracle) > 1 else 0.0, 8)} {0.0:>7.3f}")

    table = []
    all_arms = {arm_name(c): v for c, v in cells.items()} | extra_arms
    for name, by_seed in all_arms.items():
        shared = sorted(set(by_seed) & set(frozen) & set(oracle))
        if not shared:
            continue
        # R is computed on the SHARED seeds only, so an arm that happened to run a friendlier
        # subset of worlds cannot look better for that reason.
        #
        # The HEADLINE R is a ratio of means, not a mean of per-seed ratios. Per-seed ratios divide
        # by that seed's own (fixed - switching) gap, and in this regime that gap ranges over an
        # order of magnitude across seeds (0.43 to 7.16). A single seed with a small denominator
        # then dominates the average and moves R far more than it moves any loss. The per-seed
        # distribution is still reported — its median and spread say something the ratio of means
        # does not — but it is not what the tables lead with.
        rs = [normalized_regret(by_seed[s], frozen[s], oracle[s]) for s in shared]
        rs = [r for r in rs if math.isfinite(r)]
        mean = statistics.fmean(by_seed[s] for s in shared)
        sd = statistics.pstdev([by_seed[s] for s in shared]) if len(shared) > 1 else 0.0
        fm_shared = statistics.fmean(frozen[s] for s in shared)
        om_shared = statistics.fmean(oracle[s] for s in shared)
        row = {"arm": name, "n": len(shared), "mean": mean, "sd": sd,
               "R": normalized_regret(mean, fm_shared, om_shared),
               "R_perseed_mean": statistics.fmean(rs) if rs else None,
               "R_median": statistics.median(rs) if rs else None}
        table.append(row)
        print(f"{name:<38} {row['n']:>3} {fmt(mean)} {fmt(sd, 8)} "
              f"{(f'{row['R']:>7.3f}' if row['R'] is not None else '      —')}")

    # ---- 1b. VALIDITY GATE: did every arm actually act? ---------------------------------------
    # Run before the factorial, because a factorial over truncated arms is a table of nonsense.
    print("\n=== validity: decisions that produced NO parseable action ===")
    worst = 0.0
    action_rates = {}
    for name in list(all_arms):
        empty, tot = no_action_rate(store, name, args.model)
        if not tot:
            continue
        rate = empty / tot
        action_rates[name] = {"empty": empty, "total": tot, "rate": rate}
        worst = max(worst, rate)
        flag = "  <-- CONTAMINATED" if rate > 0.02 else ""
        print(f"  {name:<40} {empty:>4}/{tot:<5} {100 * rate:>5.1f}%{flag}")
    if worst > 0.02:
        print("\n  [!!] At least one arm silently failed to act on >2% of its decisions. A decision")
        print("       that emits nothing leaves the PREVIOUS law in force, so that arm is not the")
        print("       treatment it is labelled as. This failure correlates with the treatment —")
        print("       harness channels lengthen the prompt and invite longer reasoning — so the")
        print("       ablation would be measuring truncation. Raise GOVSIM_LLM_MAX_TOKENS and re-run")
        print("       before believing anything below.")

    # ---- 1b-ii. HOW the arms failed to act: truncation, or deliberation collapse? --------------
    # Reported next to the no-action rate because it decomposes it, and because the remedies are
    # opposite: truncation wants a bigger token budget, collapse wants a re-ask at a SMALLER one.
    print("\n=== decisions that collapsed into pure deliberation (completion_tokens == 0) ===")
    collapse_rates = {}
    for name in list(all_arms):
        c, tot = collapse_rate(store, name, args.model)
        if not tot:
            continue
        collapse_rates[name] = {"collapsed": c, "total": tot, "rate": c / tot}
        na = action_rates.get(name, {}).get("empty", 0)
        share = f"{100 * c / na:.0f}% of its no-action" if na else "—"
        print(f"  {name:<40} {c:>4}/{tot:<5} {100 * c / tot:>5.1f}%   ({share})")

    # ---- 1c. VALIDITY GATE: did the harness change BEHAVIOUR, or only the prompt? --------------
    print("\n=== validity: policy responsiveness to harness content ===")
    labels = {n.removeprefix("epidemic_llm_"): n for n in all_arms}
    resp_lines = _responsiveness_report(store, labels, args.model)
    for line in resp_lines:
        print(line)
    deaf = [ln for ln in resp_lines if ln.strip().startswith("DEAF")]
    if deaf:
        print("\n  [!!] At least one arm enacted essentially the SAME policies as the no-harness arm.")
        print("       The channel reached the prompt (see channel liveness) but not the behaviour, so")
        print("       that contrast is not a test of the component — it is a measurement of whether")
        print("       this model reads its scaffold at all. Report it as a capability floor, NOT as a")
        print("       null result about the harness.")

    # ---- 2. factorial attribution -------------------------------------------------------------
    factorial = None
    if len(cells) == 2 ** len(FACTORS):
        eff = factorial_effects(cells, FACTORS)
        holm = holm_bonferroni({k: v["p"] for k, v in eff.items()})
        for k in eff:
            eff[k].update(holm[k])
        factorial = eff
        print(f"\n=== factorial attribution on {METRIC} (negative effect = the factor HELPED) ===")
        print(f"{'term':<26} {'ord':>3} {'effect':>9} {'95% CI':>22} {'p':>7} {'p_holm':>8}  sig")
        for k, v in sorted(eff.items(), key=lambda kv: (kv[1]["order"], kv[1]["p"])):
            ci = f"[{v['ci_low']:+.4f}, {v['ci_high']:+.4f}]"
            print(f"{k:<26} {v['order']:>3} {v['effect']:>+9.4f} {ci:>22} "
                  f"{v['p']:>7.4f} {v['p_adj']:>8.4f}  {'YES' if v['significant'] else '·'}")
    else:
        print(f"\n(factorial skipped: {len(cells)}/{2 ** len(FACTORS)} cells present in the store)")

    # ---- 2b. what this design could have detected ---------------------------------------------
    # A null is only informative next to the smallest effect that would have shown up. Reported at
    # the CORRECTED alpha, because that is the bar the headline analysis actually applies.
    mde = None
    if len(cells) >= 2:
        base = cells.get(tuple(False for _ in FACTORS))
        if base:
            # Compute the MDE from the variance of a FACTORIAL CONTRAST, not from
            # (arm - best_fixed). Those are different quantities: the contrast differences two arms
            # that ran the same worlds AND received nearly the same prompts, so most of the seed
            # variance cancels; the arm-vs-reference difference retains it. Using the latter
            # overstated the detectable effect by roughly 2x, which is conservative — it can only
            # cause us to under-claim — but a power statement that is wrong in the safe direction
            # is still wrong, and it understates the design's own resolving power.
            if factorial:
                per_seed = [v for k, v in factorial.items()
                            if k in FACTORS and v.get("per_seed")]
                diffs = per_seed[0]["per_seed"] if per_seed else []
            else:
                diffs = []
            if not diffs:  # no factorial yet: fall back, and say so in the label
                ref = {s: frozen[s] for s in base if s in frozen}
                diffs = [base[s] - ref[s] for s in sorted(set(base) & set(ref))]
            n_terms = len(factorial) if factorial else 7
            mde = minimum_detectable_effect(diffs, n_comparisons=n_terms)
            budget = fm - om
            print(f"\n=== power: what this design could have detected ({METRIC}) ===")
            print(f"  per-seed sd of (arm - best_fixed) = {mde['sd']:.4f}   se = {mde['se']:.4f}   n = {mde['n']}")
            print(f"  minimum detectable effect at alpha={mde['alpha_effective']:.4f} "
                  f"(Bonferroni over {n_terms} terms), power {mde['power']:.0%}: "
                  f"{mde['mde']:.4f} {METRIC} units")
            print(f"  total adaptation budget (best_fixed - switching) = {budget:.4f}")
            if budget > 0:
                print(f"  => the design can only resolve a component worth "
                      f">= {100 * mde['mde'] / budget:.0f}% of the whole adaptation budget.")
                print(f"  => a null here rules out LARGE component effects, not small ones. "
                      f"Halving the MDE needs n={mde['n_for_half_mde']} seeds.")

    # ---- 3. named contrasts -------------------------------------------------------------------
    print(f"\n=== contrasts (paired bootstrap on {METRIC}; lower is better) ===")
    contrasts = {}
    refs = {"vs_best_fixed": frozen, "vs_switching": oracle}
    if stale:
        refs["vs_stale_rule"] = stale
    if "epidemic_opro" in all_arms:
        refs["vs_opro"] = all_arms["epidemic_opro"]
    for name, by_seed in all_arms.items():
        for ref_label, ref in refs.items():
            if by_seed is ref:
                continue
            res = compare(by_seed, ref, lower_is_better=True)
            if res["n"] < 2:
                continue
            res["p"] = bootstrap_p(res["diffs"])
            contrasts[f"{name} {ref_label}"] = {
                k: res[k] for k in ("n", "point_estimate", "ci_low", "ci_high",
                                    "a_better_than_b", "excludes_zero", "p",
                                    "p_wilcoxon", "robust_agreement", "ci_method")}
    holm_c = holm_bonferroni({k: v["p"] for k, v in contrasts.items()}) if contrasts else {}
    for k, v in contrasts.items():
        v.update(holm_c.get(k, {}))
    for k, v in sorted(contrasts.items(), key=lambda kv: kv[1]["point_estimate"]):
        # Read the CORRECTED result. The uncorrected flag sits in the same dict and was
        # what this line used to print — next to the p_holm it contradicted.
        # A claim must survive BOTH the Holm-corrected studentized CI and a distribution-free
        # rank test. At n=20 those disagreeing means the result rests on a few seeds' tails.
        sig = v.get("significant", v["excludes_zero"]) and v.get("robust_agreement", True)
        verdict = ("BETTER" if sig and v["point_estimate"] < 0 else
                   "worse" if sig else "ns")
        print(f"{k:<52} n={v['n']:>3} Δ={v['point_estimate']:>+9.4f} "
              f"[{v['ci_low']:+.4f},{v['ci_high']:+.4f}] p_holm={v.get('p_adj', float('nan')):.4f} "
              f"p_wilcox={v.get('p_wilcoxon', float('nan')):.4f}  {verdict}")

    # ---- 4. cross-model replication -----------------------------------------------------------
    cross = None
    if args.cross_model:
        cross = {}
        models = available_models(store)
        rungs = ["epidemic_llm_bare", "epidemic_llm_outcome", "epidemic_llm_trace_outcome_memory"]
        print(f"\n=== cross-model replication (mean normalized regret R; lower = closer to oracle) ===")
        print(f"{'model':<26} " + " ".join(f"{r.replace('epidemic_llm_', ''):>22}" for r in rungs))
        for m in models:
            row = {}
            cellstrs = []
            for arm in rungs:
                by_seed = load_by_seed(store, arm, m)
                shared = sorted(set(by_seed) & set(frozen) & set(oracle))
                if not shared:
                    cellstrs.append(f"{'—':>22}")
                    row[arm] = None
                    continue
                rs = [normalized_regret(by_seed[s], frozen[s], oracle[s]) for s in shared]
                rs = [r for r in rs if math.isfinite(r)]
                val = statistics.fmean(rs) if rs else None
                row[arm] = {"R": val, "n": len(shared)}
                cellstrs.append(f"{val:>16.3f} (n={len(shared):>2})" if val is not None else f"{'—':>22}")
            cross[m] = row
            print(f"{m:<26} " + " ".join(cellstrs))

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps({
            "model": args.model, "metric": METRIC,
            "anchors": {"frozen_mean": fm, "oracle_mean": om,
                        "headroom": (fm / om) if om else None},
            # per_seed vectors are kept in memory for the MDE and stripped from the artifact;
            # they are large, and everything downstream reads the summarised terms.
            "arms": table,
            "factorial": ({k: {kk: vv for kk, vv in v.items() if kk != "per_seed"}
                           for k, v in factorial.items()} if factorial else None),
            "contrasts": contrasts, "power": mde,
            "action_rates": action_rates,
            # Kept beside action_rates because it DECOMPOSES them: in the recorded sweep every
            # no-action decision was a deliberation collapse rather than plain truncation, and the
            # two call for opposite fixes.
            "collapse_rates": collapse_rates,
            "cross_model": cross,
        }, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
