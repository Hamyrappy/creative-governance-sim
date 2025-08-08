# linear_stochastic_system

import random
from typing import Dict, Any, List, Optional, Set, Tuple

from govsim.utils.interfaces import BaseEconomicSystem, Policy, AgentId, PolicyDescriptor
from govsim.utils.policy_utils import evaluate_safe_policy_code

from pydantic import BaseModel, Field

# --- Модель контекста, специфичная для LinearStochasticSystem ---

class LinearSystemAgentContext(BaseModel):
    """Структурированный пакет данных от LinearStochasticSystem для агента."""
    # Параметры модели
    param_A: float
    param_B: float
    param_C: float
    sigma_epsilon: float
    target_x: Optional[float]
    u_range: Tuple[float, float]

    # Текущие метрики
    current_step: int
    current_x: float
    previous_x: Optional[float] = None
    current_u: float

    # KPI (рассчитываются агентом, но модель знает, что они нужны)
    # Эти поля агент заполнит позже, но они часть общего контракта.
    # Поэтому пока сделаем их опциональными.
    perf_window: Optional[int] = None
    current_mse: Optional[float] = None
    current_msu: Optional[float] = None

    # Текстовые блоки для промпта
    history_text: Optional[str] = None

    # Ключевые текстовые блоки интерфейса
    policy_descriptors_text: Optional[str] = None
    all_available_context_vars_global: Optional[List[str]] = None


# --- Сам класс LinearStochasticSystem ---

class LinearStochasticSystem(BaseEconomicSystem):
    """
    Реализует простую линейную стохастическую динамическую систему:
    x_{k+1} = A * x_k + B * u_k + C + epsilon_k
    Предназначена для тестирования способности LLM-агента генерировать
    правила управления (обратной связи) для простой динамики.
    """
    def __init__(self, params: Dict[str, Any]):
        """
        Инициализация модели.
        Args:
            params: Словарь параметров из config.py. Ожидает:
                    'initial_x' (float): Начальное состояние x_0.
                    'param_A' (float): Коэффициент авторегрессии.
                    'param_B' (float): Коэффициент эффективности управления.
                    'param_C' (float): Константа (дрейф).
                    'sigma_epsilon' (float): Стандартное отклонение шока epsilon_k.
                    'target_x' (float, optional): Целевое значение для x_k (для передачи агенту).
                    'u_range' (Tuple[float, float], optional): Допустимый диапазон для u_k, например, (-1.0, 1.0).
        """
        super().__init__(params) # Инициализация базового класса

        # Параметры динамики
        self.param_A: float = params.get("param_A", 0.95)
        self.param_B: float = params.get("param_B", 0.5)
        self.param_C: float = params.get("param_C", 0.0)
        self.param_sigma_epsilon: float = params.get("sigma_epsilon", 0.1)
        self.param_target_x: Optional[float] = params.get("target_x", None) # Целевое значение (если есть)

        # Диапазон управления (важно для LLM и для клиппинга)
        default_u_range = (-1.0, 1.0)
        self.u_range: Tuple[float, float] = params.get("u_range", default_u_range)
        if not (isinstance(self.u_range, tuple) and len(self.u_range) == 2 and
                isinstance(self.u_range[0], (int, float)) and
                isinstance(self.u_range[1], (int, float)) and
                self.u_range[0] <= self.u_range[1]):
             print(f"Предупреждение: Некорректный 'u_range' {self.u_range} в параметрах. Используется дефолтный {default_u_range}.")
             self.u_range = default_u_range

        # Текущее состояние
        self.current_x: float = params.get("initial_x", 0.0)
        # Текущее управляющее воздействие (будет обновляться политикой)
        self.current_u: float = 0.0

        # Инициализация начального состояния и истории
        self.state = self._update_state()
        self.history.append(self.state)
        print(f"LinearStochasticSystem: Инициализирована. x_0={self.current_x:.3f}, A={self.param_A}, B={self.param_B}, C={self.param_C}, sigma={self.param_sigma_epsilon}, u_range={self.u_range}")

    def get_policy_descriptors(self) -> List[PolicyDescriptor]:
        """Возвращает список поддерживаемых дескрипторов политик (только один)."""
        # Определяем переменные, доступные для LLM в выражении для u_k
        available_vars = ["step", "current_x"]
        if self.param_target_x is not None:
            available_vars.append("target_x")
        # Добавим предыдущее значение x, если оно есть в истории
        if len(self.history) > 0:
             available_vars.append("previous_x")
        # Добавим текущее значение u, чтобы его можно было использовать в выражении (например, для плавности)
        available_vars.append("current_u")


        control_policy_descriptor = PolicyDescriptor(
            policy_type_id="set_control_input",
            description=f"Устанавливает уровень управляющего воздействия u_k в диапазоне {self.u_range}.",
            target_variable_name="current_u", # Внутренняя переменная для хранения u_k
            value_type=float,
            value_range=self.u_range, # Передаем диапазон из параметров
            available_context_vars=list(set(available_vars)), # Убираем дубликаты, если есть
            constraints={} # Пока без доп. ограничений
        )
        return [control_policy_descriptor]

    def _update_state(self) -> Dict[str, Any]:
        """Собирает полное текущее состояние модели для логирования."""
        serializable_policies = [p.to_dict() for p in self.active_policies]
        state_dict = {
            "step": self.current_step,
            "metrics": self.get_current_metrics(),
            "active_policies_log": serializable_policies,
        }
        # Добавим предыдущее состояние для информации, если оно есть
        if len(self.history) > 0:
            state_dict["previous_metrics"] = self.history[-1].get("metrics", {})
        return state_dict

    def get_current_metrics(self) -> Dict[str, float]:
        """Возвращает основные метрики модели (текущее состояние)."""
        metrics = {
            "step": float(self.current_step),
            "current_x": self.current_x,
            "current_u": self.current_u # Логируем и текущее управление
        }
        if self.param_target_x is not None:
             metrics["target_x"] = self.param_target_x # Добавляем цель, если она задана

        if len(self.history) >= 2:
            # В начале шага k в history[-1] уже лежит x_k, так что previous = x_{k-1} = history[-2]
            prev_metrics = self.history[-2].get("metrics", {})
            metrics["previous_x"] = prev_metrics.get("current_x", self.current_x)
        else:
            metrics["previous_x"] = self.current_x
        return metrics

    def get_state_for_agent(self) -> LinearSystemAgentContext:
        """Возвращает состояние, видимое агенту, в виде СТРОГО ТИПИЗИРОВАННОГО объекта."""
        
        # Получаем предыдущее значение x, если оно есть
        prev_x = None
        if self.history:
            prev_metrics = self.history[-1].get("metrics")
            if prev_metrics and "current_x" in prev_metrics:
                prev_x = prev_metrics["current_x"]
        
        # Создаем и возвращаем экземпляр нашего Pydantic-класса
        return LinearSystemAgentContext(
            param_A=self.param_A,
            param_B=self.param_B,
            param_C=self.param_C,
            sigma_epsilon=self.param_sigma_epsilon,
            target_x=self.param_target_x,
            u_range=self.u_range,
            current_step=self.current_step,
            current_x=self.current_x,
            previous_x=prev_x,
            current_u=self.current_u
            # Остальные поля (KPI, history_text) заполнит агент
        )

    def apply_policy_change(self, policy_change: Optional[Policy]) -> None:
        """Обновляет активную политику управления u_k."""
        if policy_change is None:
            # Агент решил не менять политику или не смог предложить валидную.
            # print(f"  - Активная политика управления ('set_control_input') не изменена.")
            return

        if policy_change.policy_type == "set_control_input":
            found = False
            for i, p in enumerate(self.active_policies):
                if p.policy_type == "set_control_input":
                    # Заменяем существующую политику новой
                    print(f"  - Обновление политики 'set_control_input' -> {policy_change}")
                    self.active_policies[i] = policy_change
                    found = True
                    break
            if not found:
                # Если политики такого типа нет, добавляем новую
                print(f"  - Добавление новой политики: {policy_change}")
                self.active_policies.append(policy_change)
        else:
             print(f"Предупреждение: LinearStochasticSystem получила политику неизвестного типа '{policy_change.policy_type}'. Игнорируется.")


    def step(self) -> None:
        """Выполняет один шаг симуляции: вычисление u_k и обновление x_k."""

        # --- 1. Вычисление управляющего воздействия u_k ---
        calculated_u = self.current_u # Используем предыдущее значение по умолчанию

        # Ищем активную политику
        active_policy: Optional[Policy] = None
        for p in self.active_policies:
             if p.policy_type == "set_control_input":
                 active_policy = p
                 break

        if active_policy and active_policy._compiled_safe_code:
            # Создаем контекст для выполнения выражения
            simulation_context = self.get_current_metrics()

            # Безопасно вычисляем значение политики
            policy_value = evaluate_safe_policy_code(
                active_policy._compiled_safe_code,
                simulation_context
            )

            if policy_value is not None and isinstance(policy_value, (int, float)):
                # Применяем ограничения диапазона ПОСЛЕ вычисления
                min_u, max_u = self.u_range
                calculated_u = float(max(min_u, min(max_u, policy_value)))
                # print(f"  - Политика '{active_policy.policy_type}' вычислила u_k = {calculated_u:.4f} (сырое значение: {policy_value})")
            elif policy_value is None:
                 print(f"Ошибка при вычислении значения для политики {active_policy.id}. Используется предыдущее значение u_k={self.current_u:.4f}.")
                 calculated_u = self.current_u # Остается старое значение
            else:
                 print(f"Предупреждение: Политика {active_policy.id} вернула некорректный тип {type(policy_value)}. Используется предыдущее значение u_k={self.current_u:.4f}.")
                 calculated_u = self.current_u # Остается старое значение

        # Обновляем текущее значение u_k для использования в динамике и логирования
        self.current_u = calculated_u

        # --- 2. Обновление состояния системы x_k ---
        # Генерируем стохастический шок
        shock = random.gauss(0, self.param_sigma_epsilon)

        # Рассчитываем новое состояние x_{k+1}
        next_x = (self.param_A * self.current_x +
                  self.param_B * self.current_u +
                  self.param_C +
                  shock)

        # Обновляем состояние
        self.current_x = next_x

        # --- 3. Завершение шага ---
        self.current_step += 1
        self.state = self._update_state() # Обновляем полное состояние (включая метрики и лог политик)
        self.history.append(self.state)

        # Вывод информации о шаге
        # print(f"Шаг {self.current_step}: x={self.current_x:.3f}, u={self.current_u:.3f} (Шок={shock:.3f})")

    def emulate_policy(self, policy: Policy, duration: int, agents_subset: Optional[List[AgentId]] = None) -> Dict[str, Any]:
        """Заглушка для абстрактного метода эмуляции."""
        print("Предупреждение: emulate_policy вызван, но не реализован для LinearStochasticSystem.")
        # В реальной реализации здесь нужно было бы создать копию модели,
        # применить политику и прогнать 'duration' шагов, затем вернуть результат.
        raise NotImplementedError("Метод emulate_policy не реализован для LinearStochasticSystem.")

