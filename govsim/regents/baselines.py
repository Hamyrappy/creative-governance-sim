"""
Control baselines — the rivals the LLM regent must match (linear) or beat (nonlinear/shifted).

``PIDRegent`` is the *tuned controller* the creativity metric is defined against (generalization
gap = beat a tuned PID **specifically** on un-tuned regimes). ``LQRRegent`` is the analytic
ground-truth ceiling on the linear scalar plant (the "matches LQR" sanity bar, doc-09 §6.4) — it is
the full-information baseline, so it is constructed with the known dynamics, not the obfuscated view.

Both emit a single sandboxed expression that the ``ScalarLeverInterface`` re-evaluates each step, so
they are continuous controllers despite deciding only on the schedule.
"""

from __future__ import annotations

from govsim.core.action import ActionRequest, ActionSpace
from govsim.core.regent import Regent, Scratch
from govsim.core.system import Observation


class PIDRegent(Regent):
    """A proportional-derivative control law ``u = -(Kp·e + Kd·Δx)`` where ``e = x - target``.

    (The integral term is omitted by default — ``Ki`` accumulation across the decision cadence is a
    later refinement; on a first-order plant PD is the right tuned comparator.) Emits an expression
    over whatever of ``target_x`` / ``previous_x`` the action space advertises.
    """

    def __init__(self, verb: str, kp: float = 0.9, kd: float = 0.0, state_var: str = "current_x",
                 id: str = "regent:0") -> None:
        super().__init__(id)
        self.verb = verb
        self.kp = kp
        self.kd = kd
        self.state_var = state_var

    def decide(self, view: Observation, space: ActionSpace, scratch: Scratch) -> list[ActionRequest]:
        ctx = set(space.context_vars)
        x = self.state_var
        err = f"({x} - target_x)" if "target_x" in ctx else x
        terms = [f"{self.kp!r} * {err}"]
        if self.kd and "previous_x" in ctx:
            terms.append(f"{self.kd!r} * ({x} - previous_x)")
        expr = "-(" + " + ".join(terms) + ")"
        return [ActionRequest(regent_id=self.id, verb=self.verb, payload={"expr": expr})]


class LQRRegent(Regent):
    """Analytic discrete LQR for the scalar plant ``x_{k+1} = A·x + B·u`` minimizing ``Q·x² + R·u²``.

    Full-information ground truth: built with the *known* dynamics. Solves the scalar DARE by
    fixed-point iteration and emits ``u = -K·x`` (optimal only for the linear-quadratic problem;
    on the cubic arm it is the frozen pre-shock-optimal baseline H1 must beat, doc-09 §6.4)."""

    def __init__(self, verb: str, A: float, B: float, Q: float = 1.0, R: float = 1.0,
                 state_var: str = "current_x", id: str = "regent:0") -> None:
        super().__init__(id)
        self.verb = verb
        self.state_var = state_var
        self.gain = self._solve_gain(A, B, Q, R)

    @staticmethod
    def _solve_gain(A: float, B: float, Q: float, R: float) -> float:
        P = Q
        for _ in range(10_000):
            denom = R + B * B * P
            P_next = Q + A * A * P - (A * B * P) ** 2 / denom if denom != 0 else Q + A * A * P
            if abs(P_next - P) < 1e-14:
                P = P_next
                break
            P = P_next
        denom = R + B * B * P
        return (B * P * A) / denom if denom != 0 else 0.0

    def decide(self, view: Observation, space: ActionSpace, scratch: Scratch) -> list[ActionRequest]:
        expr = f"{-self.gain!r} * {self.state_var}"
        return [ActionRequest(regent_id=self.id, verb=self.verb, payload={"expr": expr})]
