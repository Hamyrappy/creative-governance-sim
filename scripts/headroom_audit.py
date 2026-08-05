"""
Cross-domain headroom audit — decide WHICH world the headline experiment should live in.

For every candidate domain we calibrate the same two references inside one shared policy family
(``govsim.analysis.calibration``):

    frozen  — optimal on the PRE-shock (stationary) world, then deployed unchanged through the break
    oracle  — optimal on the post-shock WINDOW of the actual shocked world (clairvoyant)

and report ``headroom = L(frozen)/L(oracle)`` on that window. Headroom is the whole budget an
adaptive regent competes for, so a domain with headroom ≈ 1 cannot host the experiment no matter
how good the regent is.

**Calibrate on the objective, never on one of its components.** A first pass here calibrated the
scalar arms on ``mse`` and the epidemic arm on ``total_infected`` — i.e. on the benefit side with
the cost side deleted. Every optimum then sat at a corner ("use the maximum lever"), and a corner
optimum is regime-invariant: no shock can move it, so every domain reported ≈1.0x headroom. That
was an artifact of the calibration metric, not a fact about the worlds.

Running this *before* committing to a flagship domain is the difference between "we found no
effect" and "we chose a world where no effect was findable".

**Coverage.** The scalar and epidemic domains below are written out inline. The five economy worlds
(``govsim/domains/economy``) are NOT: each reads its config, shock, levers, objective, horizon and
cadence out of the registered ``_World`` in ``govsim/experiments/economy_experiments.py`` and
supplies only a policy family. A headroom number measured on a world that has drifted from the
world the arms run in is worse than no number, because it still looks like evidence; here they are
the same object and cannot drift.

**Two numbers, not one.** ``headroom`` is a RATIO and is only interpretable while both losses are
positive. ``CommonsWelfare`` is a net welfare measure, so a good policy scores below zero there and
the ratio flips sign or runs to +inf — which reads like a spectacular result and is an artifact of
the sign. The absolute ``gap`` (``L(frozen) - L(oracle)``) is well-defined in every case and is
reported alongside; when the ratio is not valid the report says so rather than printing a number.

    uv run python scripts/headroom_audit.py --seeds 8
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from govsim.analysis import PolicyFamily, calibrate, headroom
from govsim.core.schedule import EveryN
from govsim.domains.scalar import (
    CoupledSystem,
    CubicSystem,
    EpidemicLoss,
    Lever,
    ScalarLeverInterface,
    SIRSystem,
    StabilizationLoss,
)

SHOCK_STEP = 100


# --------------------------------------------------------------------------------------------
# Domain specs: each returns (pre_factory, post_factory, shocked_factory, iface, objective,
# family, schedule, horizon, metric). ``pre``/``post`` are *stationary* worlds used only to
# calibrate the two references; ``shocked`` is the real world both are then scored on.
# --------------------------------------------------------------------------------------------


def _mk(cls, cfg):
    def factory(seed: int):
        s = cls(cfg)
        s.reset(seed)
        return s
    return factory


def domain_cubic():
    base = {"param_A": 0.95, "param_B": 0.5, "param_C": 0.0, "sigma_epsilon": 0.08,
            "target_x": 0.0, "u_range": (-2.0, 2.0), "cubic_coeff": 0.05, "state_exponent": 3}
    shocked = dict(base, shock_step=SHOCK_STEP,
                   shock_params={"param_A": 1.03, "param_B": 0.12, "cubic_coeff": 0.10},
                   shock_state_kick=0.8)
    iface = ScalarLeverInterface([Lever("set_control_input", (-2.0, 2.0), "current_u")])
    family = PolicyFamily(
        verb="set_control_input",
        template="-({k} * current_x + {c} * current_x ** 3)",
        grid={"k": [0.5, 0.9, 1.3, 1.8, 2.5, 3.5, 6.0, 10.0],
              "c": [0.0, 0.1, 0.4, 1.0, 3.0]},
    )
    return dict(
        name="cubic (scalar plant)",
        pre=_mk(CubicSystem, base), shocked=_mk(CubicSystem, shocked), iface=iface,
        objective=StabilizationLoss(lam=0.1, post_shock_step=SHOCK_STEP),
        family=family, schedule=EveryN(25), horizon=200,
        calib_metric="loss", score_metric="post_loss",
    )


def domain_coupled():
    base = {"param_A": 0.95, "param_B": 0.4, "param_C": 0.0, "sigma_epsilon": 0.10,
            "target_x": 0.0, "u_range": (-2.0, 2.0), "u_smoothing_rho": 0.70,
            "shock_period": 0, "param_B_drift_sigma": 0.0}
    shocked = dict(base, shock_period=SHOCK_STEP, shock_magnitude_aux1=1.0,
                   shock_magnitude_aux2=-0.8, param_B_drift_sigma=0.01)
    iface = ScalarLeverInterface([Lever("set_control_input", (-2.0, 2.0), "u_commanded")])
    family = PolicyFamily(
        verb="set_control_input",
        template="-({k} * current_x + {d} * (current_x - previous_x))",
        grid={"k": [0.5, 1.0, 1.5, 2.2, 3.0, 4.5, 7.0],
              "d": [0.0, 0.5, 1.5, 3.0, 5.0]},
    )
    return dict(
        name="coupled (multi-state plant)",
        pre=_mk(CoupledSystem, base), shocked=_mk(CoupledSystem, shocked), iface=iface,
        objective=StabilizationLoss(lam=0.1, post_shock_step=SHOCK_STEP),
        family=family, schedule=EveryN(20), horizon=300,
        calib_metric="loss", score_metric="post_loss",
    )


def domain_sir_efficacy(lam: float = 0.02, efficacy: float = 0.25):
    """The instrument-failure regime: lockdown keeps costing what it always cost, and stops working.

    A threshold-on-prevalence rule is feedback, so it shrugs off a transmissibility shock. It cannot
    shrug off this one: prevalence rises, the rule locks down *harder*, and every unit of lockdown
    now buys a quarter of what it used to while costing the same. The optimal response — spend less
    on the broken instrument — is the opposite of what the rule does, which is exactly the wedge
    adaptation has to earn.
    """
    base = {"beta0": 0.35, "gamma": 0.10, "noise_sigma": 0.0, "shock_step": 10_000,
            "waning": 0.02, "import_rate": 0.0005, "lockdown_efficacy": 1.0}
    shocked = dict(base, shock_step=SHOCK_STEP, shock_factor=1.0,
                   shock_params={"lockdown_efficacy": efficacy})
    iface = ScalarLeverInterface([
        Lever("set_lockdown", (0.0, 0.9), "lockdown"),
        Lever("set_vaccination", (0.0, 0.5), "vacc"),
    ])
    family = PolicyFamily(
        verb="set_lockdown",
        template="{a} if I > {thr} else 0.0",
        grid={"a": [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9],
              "thr": [0.002, 0.005, 0.01, 0.02, 0.04, 0.08, 0.15]},
    )
    return dict(
        name=f"SIR efficacy-collapse (lam={lam}, eff {1.0}->{efficacy})",
        pre=_mk(SIRSystem, base), shocked=_mk(SIRSystem, shocked), iface=iface,
        objective=EpidemicLoss(lam=lam, post_shock_step=SHOCK_STEP),
        family=family, schedule=EveryN(10), horizon=200,
        calib_metric="loss", score_metric="post_loss", switch_step=SHOCK_STEP,
    )


def domain_sir(lam: float = 0.02):
    """Epidemic governance: a threshold lockdown institution meets a more transmissible variant.

    Bounded by construction (S+I+R is conserved), so no arm can 'diverge' — the comparison is
    always between two real policies, never between two blow-ups. The policy family is a
    *threshold institution* ('lock down at intensity a whenever prevalence exceeds θ'), which is
    what an actual public-health rule looks like and what a code-emitting regent can express
    natively.

    ``lam`` prices lockdown against infection. It has to be tuned until the optimum is interior:
    at λ=1.0 the cheapest thing to do is lock down maximally forever, which is regime-invariant and
    therefore has no headroom (and is also not a description of any real polity).
    """
    # ENDEMIC configuration: waning immunity + a trickle of imported cases, so the disease is still
    # circulating when the variant lands. Without this the epidemic can burn out before the shock
    # and "do nothing" wins the post-shock window by emptying it.
    base = {"beta0": 0.35, "gamma": 0.10, "noise_sigma": 0.0, "shock_step": 10_000,
            "waning": 0.02, "import_rate": 0.0005}
    shocked = dict(base, shock_step=SHOCK_STEP, shock_factor=1.8)
    iface = ScalarLeverInterface([
        Lever("set_lockdown", (0.0, 0.9), "lockdown"),
        Lever("set_vaccination", (0.0, 0.5), "vacc"),
    ])
    family = PolicyFamily(
        verb="set_lockdown",
        template="{a} if I > {thr} else 0.0",
        grid={"a": [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9],
              "thr": [0.002, 0.005, 0.01, 0.02, 0.04, 0.08, 0.15]},
    )
    return dict(
        name=f"SIR (epidemic governance, lam={lam})",
        pre=_mk(SIRSystem, base), shocked=_mk(SIRSystem, shocked), iface=iface,
        objective=EpidemicLoss(lam=lam, post_shock_step=SHOCK_STEP),
        family=family, schedule=EveryN(10), horizon=200,
        calib_metric="loss", score_metric="post_loss", switch_step=SHOCK_STEP,
    )


# --------------------------------------------------------------------------------------------
# The economy library (``govsim/domains/economy``) — five worlds, four of them carrying an
# INSTRUMENT_EFFICACY break and one carrying a DELAY.
#
# Each spec below reads its world, its shock, its levers, its objective, its horizon and its
# decision cadence straight out of the registered ``_World`` in
# ``govsim/experiments/economy_experiments.py``, and supplies ONLY the policy family. That import
# is the point of the design: a headroom number measured on a world that has quietly drifted from
# the world the arms actually run in is worse than no number at all, because it looks like
# evidence. Here the two cannot drift — they are the same object.
#
# The family is the one thing the audit must own, because it is the shared *language* of both
# references, and "the oracle was weak" must never be the explanation for a small headroom. Each
# one below therefore spans the institutional shapes that world's authority could plausibly
# legislate AND contains both corners, so a corner optimum shows up as a corner rather than as a
# missing option.
# --------------------------------------------------------------------------------------------


def _economy(key: str, family: PolicyFamily, note: str = "") -> dict:
    from govsim.experiments.economy_experiments import WORLDS_BY_KEY

    w = WORLDS_BY_KEY[key]
    return dict(
        name=f"{key} ({w.scenario.kind.value}, {w.scenario.name}){(' — ' + note) if note else ''}",
        # ``pre`` is the SAME config with the scenario not armed, so the two references differ in
        # when they were allowed to look and in nothing else.
        pre=_mk(w.system_cls, dict(w.base)),
        shocked=_mk(w.system_cls, w.shocked_config),
        iface=ScalarLeverInterface(list(w.levers)),
        objective=w.objective(),
        family=family,
        schedule=EveryN(w.decide_every),
        horizon=w.horizon,
        calib_metric="loss", score_metric="post_loss",
        # Carried so ``decompose_library.py`` can build the clairvoyant ADAPTOR on the same world:
        # the switch step must be the scenario's own break, never a hand-copied constant.
        switch_step=w.scenario.step,
    )


def domain_monetary():
    """Policy rate against a transmission collapse. The Lucas critique's own case.

    The family is a Taylor rule with a free intercept, which matters: the correct answer to a
    disconnected instrument is to stop leaning AND to sit lower than the neutral rate would have
    you sit, and a family with the intercept pinned at neutral could not express it. ``g = 0``
    reduces the rule to a constant rate, so the pure no-feedback institution is in the language.
    """
    return _economy("monetary", PolicyFamily(
        verb="set_policy_rate",
        template="{base} + {g} * (inflation - inflation_target + 0.5 * output_gap)",
        grid={"base": [0.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0],
              "g": [0.0, 0.5, 1.0, 2.0]},
    ))


def domain_fiscal():
    """Tax rate + transfer against a compliance collapse. Two instruments, searched jointly.

    Jointly because the whole content of the break is that one instrument's price survives while
    its yield does not, so the response is a re-balancing BETWEEN the two. A family that fixed the
    tax rate and searched only the transfer could not express the response and would report the
    world as headroom-free for a reason that is about the family.
    """
    return _economy("fiscal", PolicyFamily(
        verb="set_transfer",
        template="min(40.0, {g0} + {g} * (y_potential - output))",
        grid={"g0": [0.0, 5.0, 10.0, 20.0],
              "g": [0.0, 0.25, 0.5],
              "tau": [0.0, 0.05, 0.10, 0.20, 0.30]},
        extra_laws={"set_tax_rate": "{tau}"},
    ))


def domain_commons():
    """Quota + reserve against an enforcement collapse — the instrument-SUBSTITUTION case.

    The family has to span both instruments or the audit measures the wrong thing entirely: at zero
    compliance the quota channel has gain zero, so the only response that exists is to close water
    instead, and a quota-only oracle would be as helpless as the frozen rule and report ~1.0x.
    ``h = 0`` is a total ban and ``h = 1, esc = 0`` is an unconstraining quota, so both corners of
    the harvest rule are in the language alongside the constant-escapement shapes between them.
    """
    return _economy("commons", PolicyFamily(
        verb="set_quota",
        template="max(0.0, {h} * (stock - {esc} * capacity))",
        grid={"h": [0.0, 0.25, 0.5, 1.0],
              "esc": [0.0, 0.2, 0.4, 0.6],
              "res": [0.0, 0.2, 0.4, 0.8]},
        extra_laws={"set_reserve": "{res}"},
    ))


def domain_supply_chain():
    """Order-up-to policy against a tripled lead time — the library's one DELAY arm.

    ``ShockKind.DELAY`` is listed in ``govsim/scenarios.py`` as NOT YET MEASURED, and this is the
    measurement. The family spans a constant order (``s = g = 0``), a chase-demand rule
    (``s = 1``), and order-up-to rules with a free target and a free correction gain — because a
    delay shock breaks the *gain*, and a family without a free gain could not adapt to it even in
    principle.
    """
    return _economy("supply_chain", PolicyFamily(
        verb="set_order",
        template="max(0.0, {c} + {s} * recent_demand + {g} * (backlog - inventory))",
        grid={"c": [0.0, 10.0],
              "s": [0.0, 1.0, 3.0, 5.0],
              "g": [0.0, 0.5, 1.0]},
    ))


def domain_opinion():
    """Moderation against an efficacy collapse. A threshold institution with a floor.

    ``thr = 0.0`` makes the rule fire always, so every constant level is in the family — which is
    what lets the audit find "spend nothing" if that is the post-break optimum. That matters here
    more than elsewhere: the world's own sweep says the severe-collapse answer IS zero, and zero is
    precisely the response a threshold on a rising observable cannot produce.
    """
    return _economy("opinion", PolicyFamily(
        verb="set_moderation",
        template="{a} if polarization > {thr} else {b}",
        grid={"a": [0.0, 0.2, 0.4, 0.6, 1.0],
              "thr": [0.0, 0.1, 0.25],
              "b": [0.0, 0.3]},
    ))


DOMAINS = [
    domain_cubic, domain_coupled,
    lambda: domain_sir(0.02),
    lambda: domain_sir_efficacy(0.02, 0.50),
    lambda: domain_sir_efficacy(0.02, 0.25),
    lambda: domain_sir_efficacy(0.02, 0.00),
    lambda: domain_sir_efficacy(0.08, 0.25),
    # the economy library
    domain_monetary, domain_fiscal, domain_commons, domain_supply_chain, domain_opinion,
]


def audit(spec: dict, seeds: list[int]) -> dict:
    from govsim.analysis.calibration import _score_expr

    fam, iface, sched, hz = spec["family"], spec["iface"], spec["schedule"], spec["horizon"]
    cm, sm = spec["calib_metric"], spec["score_metric"]
    print(f"\n=== {spec['name']} — calibrating {fam.size()} laws x 2 regimes x {len(seeds)} seeds ===")
    sys.stdout.flush()

    # frozen: optimal on the stationary PRE-shock world, on the full-horizon loss.
    frozen_cal = calibrate(fam, system_factory=spec["pre"], action_interface=iface,
                           objective=spec["objective"], schedule=sched, seeds=seeds,
                           horizon=hz, metric=cm)
    # oracle: optimal on the post-shock WINDOW of the world that actually gets shocked. This is the
    # honest clairvoyant: same family, same window, same seeds — it differs from `frozen` only in
    # *when* it was allowed to look.
    oracle_cal = calibrate(fam, system_factory=spec["shocked"], action_interface=iface,
                           objective=spec["objective"], schedule=sched, seeds=seeds,
                           horizon=hz, metric=sm)
    # Re-score with the FULL law set, not with ``best_expr``. ``best_expr`` renders only the
    # family's primary verb, so on a multi-instrument family (``extra_laws``) it silently drops
    # every other lever and the re-scored reference governs with one hand — while the calibration
    # that chose it used both. That is not a small error: it produced a commons headroom of 0.82x,
    # i.e. a clairvoyant oracle scoring WORSE than the frozen rule it is defined to dominate, which
    # is impossible and is the only reason the bug was visible at all. Every pre-existing domain in
    # this file has a single-verb family, so nothing caught it until the economy library landed.
    frozen_law = frozen_cal.best_laws if fam.extra_laws else frozen_cal.best_expr
    oracle_law = oracle_cal.best_laws if fam.extra_laws else oracle_cal.best_expr
    print(f"  frozen (pre-shock optimal):  {frozen_law}")
    print(f"  oracle (post-shock optimal): {oracle_law}")

    kw = dict(verb=fam.verb, system_factory=spec["shocked"], action_interface=iface,
              objective=spec["objective"], schedule=sched, seeds=seeds, horizon=hz, metric=sm)
    frozen_loss = _score_expr(frozen_law, **kw)
    oracle_loss = _score_expr(oracle_law, **kw)
    h = headroom(frozen_loss, oracle_loss)
    same = frozen_law == oracle_law
    # The RATIO is only interpretable while both losses are positive. Some objectives here are net
    # welfare measures (``CommonsWelfare`` is catch value minus costs), so a good policy can score
    # BELOW zero and the ratio then flips sign or blows up to +inf — a number that reads like an
    # enormous result and means nothing. The absolute gap is well-defined in every case, so report
    # it always and let it stand in when the ratio cannot.
    gap = frozen_loss - oracle_loss
    ratio_ok = math.isfinite(h) and oracle_loss > 0 and frozen_loss > 0
    verdict = (f"HEADROOM = {h:.2f}x  (frozen carries {100 * (h - 1):.0f}% excess loss)"
               if ratio_ok else
               f"HEADROOM RATIO UNDEFINED (a loss is <= 0 — this objective is a net welfare "
               f"measure); absolute gap = {gap:.5f}")
    print(f"  on the SHOCKED world ({sm}): frozen={frozen_loss:.5f}  oracle={oracle_loss:.5f}  "
          f"=> {verdict}"
          f"{'   [!] identical policies — the shock does not move the optimum' if same else ''}")
    sys.stdout.flush()
    return {
        "domain": spec["name"], "frozen_expr": frozen_law,
        "oracle_expr": oracle_law, "frozen_loss": frozen_loss,
        "oracle_loss": oracle_loss, "headroom": h, "headroom_ratio_valid": ratio_ok,
        "gap": gap, "metric": sm,
        "family_size": fam.size(), "same_policy": same,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--json", help="write results here")
    ap.add_argument("--only", help="audit a single domain by substring match")
    args = ap.parse_args()
    seeds = list(range(args.seeds))

    specs = [d() for d in DOMAINS]
    if args.only:
        specs = [s for s in specs if args.only.lower() in s["name"].lower()]
    rows = [audit(s, seeds) for s in specs]

    print("\n=== headroom ranking (higher = more room for adaptation to matter) ===")
    for r in sorted(rows, key=lambda r: -r["headroom"]):
        ratio = f"{r['headroom']:>8.2f}x" if r["headroom_ratio_valid"] else "     n/a "
        flag = "  [!] same policy both sides" if r["same_policy"] else ""
        print(f"  {ratio}  gap={r['gap']:>12.4f}  {r['domain']}{flag}")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
