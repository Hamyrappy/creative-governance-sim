# coupled_linear_stochastic_system.py
# Более сложная стохастическая система управления на базе LinearStochasticSystem
# Совместима с текущим IntelligentLLMAgent (policy_type_id остаётся "set_control_input")

import random
import math
from numbers import Real
from typing import Dict, Any, List, Optional, Tuple

from govsim.utils.interfaces import BaseEconomicSystem, Policy, AgentId, PolicyDescriptor
from govsim.utils.policy_utils import evaluate_safe_policy_code
from govsim.economic_models.linear_stochastic_system import LinearSystemAgentContext


class CoupledLinearStochasticSystem(BaseEconomicSystem):
    """
    Многомерная линейная стохастическая система со сглаживанием управления
    и дрейфом параметров. Внутреннее состояние векторное (x_main, x_aux1, x_aux2),
    но *основной* KPI и целевая переменная — x_main (совместимость с текущим агентом).

    Динамика:
      u_eff[k]   = rho_u * u_eff[k-1] + (1 - rho_u) * u_cmd[k]
      X[k+1]     = A @ X[k] + b * u_eff[k] + c + eps[k],
      где X = [x_main, x_aux1, x_aux2]^T,
          A = [[A_main,   a12,   a13],
               [a21,    gamma1,  0.0],
               [a31,      0.0, gamma2]],
          b = [B_eff, d1, d2]^T,
          c = [C_eff,  0,  0]^T,
          eps ~ N(0, diag([sigma_main^2, sigma_aux1^2, sigma_aux2^2])).

    Сложность для регулятора создают:
      • Инерция управления (rho_u).
      • Перекрёстные связи между X (матрица A).
      • Дрейф эффективности управления B_eff (случайный блуждающий дрейф).
      • Случайные шоки вспомогательных состояний и периодические «режимные» толчки.
    """

    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)

        # --- Параметры «как в LinearStochasticSystem» (для совместимости промпта/агента) ---
        self.current_x: float = float(params.get("initial_x", 0.0))  # это x_main
        self.param_A: float = float(params.get("param_A", 0.95))
        self.param_B: float = float(params.get("param_B", 0.4))      # будет дрейфовать
        self.param_C: float = float(params.get("param_C", 0.0))
        self.param_sigma_epsilon: float = float(params.get("sigma_epsilon", 0.10))  # sigma_main
        self.param_target_x: Optional[float] = params.get("target_x", 0.0)
        self.u_range: Tuple[float, float] = tuple(params.get("u_range", (-2.0, 2.0)))  # type: ignore

        # --- Новые параметры сложности ---
        # Перекрёстные связи и собственная динамика вспомогательных состояний
        self.a12: float = float(params.get("a12", 0.20))   # влияние x_aux1 на x_main
        self.a13: float = float(params.get("a13", -0.10))  # влияние x_aux2 на x_main
        self.a21: float = float(params.get("a21", 0.15))   # обратная связь x_main -> x_aux1
        self.a31: float = float(params.get("a31", -0.10))  # обратная связь x_main -> x_aux2
        self.gamma1: float = float(params.get("gamma1", 0.92))  # собственная инерция x_aux1
        self.gamma2: float = float(params.get("gamma2", 0.88))  # собственная инерция x_aux2

        # Чувствительности к управлению (вектор b)
        self.d1: float = float(params.get("d1", 0.20))
        self.d2: float = float(params.get("d2", -0.15))

        # Шумы по состояниям
        self.sigma_aux1: float = float(params.get("sigma_aux1", 0.05))
        self.sigma_aux2: float = float(params.get("sigma_aux2", 0.05))

        # Инерция (сглаживание) управления и дрейф параметров
        self.rho_u: float = float(params.get("u_smoothing_rho", 0.70))  # 0..1 (чем больше, тем инерционнее)
        self.B_drift_sigma: float = float(params.get("param_B_drift_sigma", 0.01))
        self.C_drift_sigma: float = float(params.get("param_C_drift_sigma", 0.002))
        self.target_drift_sigma: float = float(params.get("target_drift_sigma", 0.0))

        # Периодические внешние толчки (чтобы ломать наработанную политику)
        self.shock_period: int = int(params.get("shock_period", 150))
        self.shock_magnitude_aux1: float = float(params.get("shock_magnitude_aux1", 0.8))
        self.shock_magnitude_aux2: float = float(params.get("shock_magnitude_aux2", -0.6))

        # Начальные доп. состояния
        self.x_aux1: float = float(params.get("initial_x_aux1", 0.0))
        self.x_aux2: float = float(params.get("initial_x_aux2", 0.0))

        # Текущее эффективное и командуемое управление
        self.current_u: float = 0.0    # u_eff (лог/метрики совместимы с Linear)
        self.u_commanded: float = 0.0  # u_cmd (для отладки)

        # Лимиты для дрейфующих параметров (чтобы не «уплыли» слишком далеко)
        self.param_B_bounds: Tuple[float, float] = tuple(params.get("param_B_bounds", (-1.5, 1.5)))  # type: ignore
        self.param_C_bounds: Tuple[float, float] = tuple(params.get("param_C_bounds", (-1.0, 1.0)))  # type: ignore

        # Валидация и нормализация параметров
        self._validate_params()

        # Инициализация состояния и истории
        self.state = self._update_state()
        self.history.append(self.state)

    # --- Интерфейс политик ---
    def get_policy_descriptors(self) -> List[PolicyDescriptor]:
        # Разрешаем использовать как старые имена (совместимость промпта), так и новые
        allowed_vars = [
            # Базовые, которые агент уже ожидает в линейной версии
            "step", "current_x", "previous_x", "current_u", "target_x",
            "param_A", "param_B", "param_C", "sigma_epsilon",
            # Новые метрики/состояния
            "x_aux1", "x_aux2", "u_commanded", "rho_u",
            # Явные границы управления (если агент хочет учитывать их в выражении)
            "u_range_min", "u_range_max",
        ]

        return [
            PolicyDescriptor(
                policy_type_id="set_control_input",
                description=(
                    "Устанавливает управляющее воздействие u_cmd. Фактическое u_eff применяется с инерцией"
                    " u_eff = rho_u*u_prev + (1-rho_u)*u_cmd."
                ),
                target_variable_name="current_u",  # совместимость — хранит u_eff
                value_type=float,
                value_range=self.u_range,
                available_context_vars=allowed_vars,
                constraints={"note": "control is smoothed; consider future inertia"},
            )
        ]

    # --- Представление состояния для агента (совместимо с IntelligentLLMAgent) ---
    def get_state_for_agent(self) -> LinearSystemAgentContext:
        return LinearSystemAgentContext(
            # Параметры (часть дрейфует во времени)
            param_A=self.param_A,
            param_B=self.param_B,
            param_C=self.param_C,
            sigma_epsilon=self.param_sigma_epsilon,
            target_x=self.param_target_x,
            u_range=self.u_range,
            # Метрики
            current_step=self.current_step,
            current_x=self.current_x,
            previous_x=self.history[-1]["metrics"].get("current_x") if self.history else None,
            current_u=self.current_u,
        )

    # --- Метрики/лог ---
    def _update_state(self) -> Dict[str, Any]:
        serializable_policies = [p.to_dict() for p in self.active_policies]
        return {
            "step": self.current_step,
            "metrics": self.get_current_metrics(),
            "active_policies_log": serializable_policies,
        }

    def get_current_metrics(self) -> Dict[str, float]:
        return {
            "step": float(self.current_step),
            # Основной KPI-объект (как раньше)
            "current_x": float(self.current_x),
            "previous_x": float(self.history[-1]["metrics"].get("current_x", self.current_x)) if self.history else float(self.current_x),
            "current_u": float(self.current_u),     # u_eff
            "u_commanded": float(self.u_commanded), # u_cmd
            "target_x": float(self.param_target_x) if self.param_target_x is not None else 0.0,
            # Параметры (видимые в контексте)
            "param_A": float(self.param_A),
            "param_B": float(self.param_B),
            "param_C": float(self.param_C),
            "sigma_epsilon": float(self.param_sigma_epsilon),
            "u_range_min": float(self.u_range[0]),
            "u_range_max": float(self.u_range[1]),
            "rho_u": float(self.rho_u),
            # Вспомогательные состояния
            "x_aux1": float(self.x_aux1),
            "x_aux2": float(self.x_aux2),
        }

    # --- Приватные утилиты ---
    def _clip(self, v: float, lo: float, hi: float) -> float:
        if lo > hi:
            lo, hi = hi, lo
        if not math.isfinite(float(v)):
            return lo if v < lo else hi if v > hi else lo
        return lo if v < lo else hi if v > hi else v

    def _validate_params(self) -> None:
        # rho_u в [0,1]
        try:
            self.rho_u = float(self.rho_u)
        except Exception as e:
            raise ValueError(f"u_smoothing_rho must be a number: {e}")
        if not math.isfinite(self.rho_u):
            raise ValueError("u_smoothing_rho must be finite")
        self.rho_u = max(0.0, min(1.0, self.rho_u))

        # u_range порядок и различие границ
        try:
            lo, hi = float(self.u_range[0]), float(self.u_range[1])
        except Exception as e:
            raise ValueError(f"u_range must be a pair of numbers: {e}")
        if lo == hi:
            raise ValueError("u_range bounds must differ (lo != hi)")
        if lo > hi:
            lo, hi = hi, lo  # нормализуем порядок
        self.u_range = (lo, hi)

        # сигмы >= 0 и конечны
        for name in ("param_sigma_epsilon", "sigma_aux1", "sigma_aux2", "B_drift_sigma", "C_drift_sigma", "target_drift_sigma"):
            val = getattr(self, name)
            try:
                val = float(val)
            except Exception as e:
                raise ValueError(f"{name} must be a number: {e}")
            if not math.isfinite(val) or val < 0.0:
                raise ValueError(f"{name} must be finite and >= 0 (got {val})")
            setattr(self, name, val)

        # нормализуем границы дрейфа (на всякий случай)
        b_lo, b_hi = float(self.param_B_bounds[0]), float(self.param_B_bounds[1])
        if b_lo > b_hi:
            self.param_B_bounds = (b_hi, b_lo)
        c_lo, c_hi = float(self.param_C_bounds[0]), float(self.param_C_bounds[1])
        if c_lo > c_hi:
            self.param_C_bounds = (c_hi, c_lo)

    # --- Применение/замена политики ---
    def apply_policy_change(self, policy_change: Optional[Policy]) -> None:
        if not policy_change:
            return
        replaced = False
        for i, p in enumerate(self.active_policies):
            if p.policy_type == "set_control_input":
                self.active_policies[i] = policy_change
                replaced = True
                break
        if not replaced:
            self.active_policies.append(policy_change)

    # --- Один шаг динамики ---
    def step(self) -> None:
        # 1) Вычисляем новое командуемое управление u_cmd по активной политике
        active = next((p for p in self.active_policies if p.policy_type == "set_control_input"), None)
        if active and active._compiled_safe_code:
            ctx = self.get_current_metrics()
            raw_val = evaluate_safe_policy_code(active._compiled_safe_code, ctx)
            if isinstance(raw_val, Real) and math.isfinite(float(raw_val)):
                lo, hi = self.u_range
                self.u_commanded = float(self._clip(float(raw_val), lo, hi))
            else:
                # Некорректное выражение -> удерживаем прошлое u_commanded
                self.u_commanded = float(self.u_commanded)
        # если политики нет, u_commanded остаётся прежним

        # 2) Сглаживаем управление (эффективное значение, используемое в динамике)
        u_eff_prev = float(self.current_u)
        u_eff = self.rho_u * u_eff_prev + (1.0 - self.rho_u) * self.u_commanded
        self.current_u = float(self._clip(u_eff, self.u_range[0], self.u_range[1]))

        # 3) Дрейф параметров
        if self.B_drift_sigma > 0.0:
            self.param_B += random.gauss(0.0, self.B_drift_sigma)
            self.param_B = self._clip(self.param_B, self.param_B_bounds[0], self.param_B_bounds[1])
        if self.C_drift_sigma > 0.0:
            self.param_C += random.gauss(0.0, self.C_drift_sigma)
            self.param_C = self._clip(self.param_C, self.param_C_bounds[0], self.param_C_bounds[1])
        if self.target_drift_sigma > 0.0 and self.param_target_x is not None:
            self.param_target_x += random.gauss(0.0, self.target_drift_sigma)

        # 4) Периодические толчки во вспомогательных состояниях
        if self.shock_period > 0 and self.current_step > 0 and (self.current_step % self.shock_period == 0):
            self.x_aux1 += self.shock_magnitude_aux1
            self.x_aux2 += self.shock_magnitude_aux2

        # 5) Шумы
        eps_main = random.gauss(0.0, self.param_sigma_epsilon)
        eps_aux1 = random.gauss(0.0, self.sigma_aux1)
        eps_aux2 = random.gauss(0.0, self.sigma_aux2)

        # 6) Обновление состояний (матрица A и вектор b «зашиты» в параметры класса)
        x_main_prev = self.current_x
        x1 = (
            self.param_A * x_main_prev
            + self.a12 * self.x_aux1
            + self.a13 * self.x_aux2
            + self.param_B * self.current_u
            + self.param_C
            + eps_main
        )
        x2 = (
            self.a21 * x_main_prev
            + self.gamma1 * self.x_aux1
            + self.d1 * self.current_u
            + eps_aux1
        )
        x3 = (
            self.a31 * x_main_prev
            + self.gamma2 * self.x_aux2
            + self.d2 * self.current_u
            + eps_aux2
        )

        self.current_x = float(x1)
        self.x_aux1 = float(x2)
        self.x_aux2 = float(x3)

        # 7) Завершение шага
        self.current_step += 1
        self.state = self._update_state()
        self.history.append(self.state)

    # --- Пока не реализовано ---
    def emulate_policy(self, policy: Policy, duration: int, agents_subset: Optional[List[AgentId]] = None) -> Dict[str, Any]:
        print("Предупреждение: emulate_policy вызван, но не реализован для CoupledLinearStochasticSystem.")
        raise NotImplementedError("Метод emulate_policy не реализован для CoupledLinearStochasticSystem.")
