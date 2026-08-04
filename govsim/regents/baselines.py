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


class SwitchingRegent(Regent):
    """Enacts one law before ``switch_step`` and another at or after it — the CLAIRVOYANT adaptor.

    This is the reference that makes an adaptation claim falsifiable when the metric spans the whole
    horizon. Scoring only the post-break window turns out to reward *passivity*: a policy that never
    intervenes is wrong before the break and, if the break makes the instrument useless, close to
    right after it, so it scores well without having adapted to anything. Over the full horizon no
    single fixed law can be optimal on both sides of a break that moves the optimum, so beating the
    best fixed law in hindsight requires actually changing behaviour.

    It is given both laws and the exact switch time, none of which any other arm can see.
    """

    def __init__(self, verb: str, pre_expr: str, post_expr: str, switch_step: int,
                 id: str = "regent:0") -> None:
        super().__init__(id)
        self.verb = verb
        self.pre_expr = pre_expr
        self.post_expr = post_expr
        self.switch_step = switch_step

    def decide(self, view: Observation, space: ActionSpace, scratch: Scratch) -> list[ActionRequest]:
        expr = self.pre_expr if view.t < self.switch_step else self.post_expr
        return [ActionRequest(regent_id=self.id, verb=self.verb, payload={"expr": expr})]


class OracleRegent(LQRRegent):
    """The CLAIRVOYANT full-information reference: it is handed the *post*-shock plant.

    It feedback-linearizes the nonlinearity and then applies the LQR gain of the post-shock
    linearization::

        u = -K·x  -  (g/B)·x**p ,      K = LQR(A_post, B, Q, R)

    so (absent clipping) the closed loop collapses to ``x_{k+1} = (A_post - B·K)·x + ε`` — the
    linear-quadratic optimum for the *shifted* plant, with the cubic term exactly cancelled.

    It is not a rival any regent could be: it reads parameters that are, by construction, invisible
    to every other arm (the shock is unseen and ``f`` is unknown under the obfuscated prompt). Its
    only job is to **anchor the achievable end of the scale**. With the frozen pre-shock LQR
    anchoring the other end, post-shock loss becomes a normalized regret

        R = (L(arm) − L(oracle)) / (L(frozen) − L(oracle))

    where R=0 is clairvoyant and R=1 is "did no better than never adapting". That turns a
    unit-bound MSE into a quantity comparable across systems, severities, and models — and it makes
    the severity knob auditable: a regime with no gap between frozen and oracle has no headroom, so
    a null result there is a statement about the *environment*, not about the regent.
    """

    def __init__(self, verb: str, A_post: float, B: float, Q: float = 1.0, R: float = 1.0,
                 cubic_coeff_post: float = 0.0, state_exponent: int = 3,
                 state_var: str = "current_x", id: str = "regent:0") -> None:
        super().__init__(verb, A_post, B, Q, R, state_var=state_var, id=id)
        if B == 0.0:
            raise ValueError("OracleRegent needs a non-zero control gain B to cancel the nonlinearity")
        self.cancel_coeff = cubic_coeff_post / B
        self.state_exponent = state_exponent

    def decide(self, view: Observation, space: ActionSpace, scratch: Scratch) -> list[ActionRequest]:
        expr = f"{-self.gain!r} * {self.state_var}"
        if self.cancel_coeff:
            expr += f" - {self.cancel_coeff!r} * {self.state_var} ** {self.state_exponent}"
        return [ActionRequest(regent_id=self.id, verb=self.verb, payload={"expr": expr})]
