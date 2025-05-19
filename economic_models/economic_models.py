# economic_models.py
import random
from typing import Dict, Any, List, Optional, Set, Tuple

# Импортируем базовый класс, интерфейс Policy и утилиту выполнения
from interfaces import BaseEconomicSystem, Policy, AgentId, PolicyDescriptor
from policy_utils import evaluate_safe_policy_code


class SimpleGrowthModel(BaseEconomicSystem):
    """
    Простейшая макроэкономическая модель для демонстрации и тестирования.
    Описывает динамику одного агрегата (ВВП) под влиянием базового роста
    и налоговой ставки, устанавливаемой политикой правительства.
    """
    def __init__(self, params: Dict[str, Any]):
        """
        Инициализация модели.
        Args:
            params: Словарь параметров из config.py (например, initial_gdp, base_growth_rate).
        """
        super().__init__(params) # Инициализация базового класса (шаг, состояние, политики, история)
        # Параметры модели
        self.gdp: float = params.get("initial_gdp", 1000.0)
        self.base_growth_rate: float = params.get("base_growth_rate", 0.02)
        # Внутренние переменные состояния, на которые влияют политики
        self.current_tax_rate: float = params.get("init_tax_rate", 0.1) # Начальная налоговая ставка

        # Инициализация начального состояния и истории
        self.state = self._update_state()
        self.history.append(self.state)

    def get_policy_descriptors(self) -> List[PolicyDescriptor]:
        """Возвращает список поддерживаемых дескрипторов политик."""
        tax_policy_descriptor = PolicyDescriptor(
            policy_type_id="set_tax_rate",
            description="Устанавливает общую налоговую ставку в экономике.",
            target_variable_name="current_tax_rate", # Явно указываем переменную
            value_type=float, # Ожидаем число с плавающей точкой
            value_range=(0.0, 0.5), # Допустимый диапазон
            available_context_vars=["step", "gdp", "tax_rate"], # Что можно использовать в выражении
            constraints={ # Пример ограничений
                'min_change_interval': 3, # Менять налог не чаще чем раз в 3 шага
                'change_cost': 5.0        # Изменение стоит 5 единиц (нужен бюджет!)
            }
        )

        return [tax_policy_descriptor] # Пока возвращаем только один дескриптор

    def _update_state(self) -> Dict[str, Any]:
        """
        Собирает и возвращает полное текущее состояние модели в виде словаря.
        Включает шаг, метрики и активные политики (в сериализуемом виде).
        """
        # Преобразуем активные политики в словари для безопасной сериализации
        serializable_policies = [p.to_dict() for p in self.active_policies]

        return {
            "step": self.current_step,
            "metrics": self.get_current_metrics(), # Получаем текущие метрики
            # ВАЖНО: Сохраняем политики как словари, чтобы избежать проблем с JSON
            "active_policies_log": serializable_policies
        }

    def get_current_metrics(self) -> Dict[str, float]:
        """Возвращает основные метрики модели."""
        return {
            "step": float(self.current_step), # Убедимся, что тип float для JSON
            "gdp": self.gdp,
            "tax_rate": self.current_tax_rate
        }

    def get_state_for_agent(self) -> Dict[str, Any]:
        """
        Возвращает состояние, видимое агенту-правительству.
        Включает текущие метрики и *полные* объекты активных политик (включая выражения).
        """
        return {
            # Агент получает словарь текущих метрик
            "metrics": self.get_current_metrics(),
            # Агент получает список текущих объектов Policy (не словарей)
            "active_policies": self.active_policies
        }

    def apply_policy_change(self, policy_change: Optional[Policy]) -> None:
        """
        Обновляет список активных политик (`self.active_policies`).
        Заменяет существующую политику того же типа или добавляет новую.
        """
        if policy_change is None:
            # Агент решил не менять политику или не смог предложить валидную.
            print("  - Активная политика не изменена.")
            return

        # Проверяем, есть ли уже политика такого типа
        found = False
        for i, p in enumerate(self.active_policies):
            if p.policy_type == policy_change.policy_type:
                # Заменяем существующую политику новой
                print(f"  - Обновление политики типа '{p.policy_type}' -> {policy_change}")
                self.active_policies[i] = policy_change
                found = True
                break

        if not found:
            # Если политики такого типа нет, добавляем новую
             print(f"  - Добавление новой политики: {policy_change}")
             self.active_policies.append(policy_change)

    def step(self) -> None:
        """Выполняет один шаг симуляции: вычисление политик и обновление экономики."""

        # --- 1. Вычисление и применение эффектов активных политик ---
        # Создаем контекст для выполнения выражений политик.
        # В него входят текущие метрики модели.
        simulation_context = self.get_current_metrics()
        # Можно добавить и другие переменные, если они нужны выражениям
        # simulation_context['previous_gdp'] = self.history[-1]['metrics']['gdp'] if self.history else self.gdp

        executed_policy_values = {} # Сохраним вычисленные значения для отладки

        for policy in self.active_policies:
            # Проверяем, что политика была успешно скомпилирована агентом
            if policy._compiled_safe_code is None:
                print(f"Предупреждение: Политика {policy.id} ('{policy.policy_type}') не имеет скомпилированного кода, пропускается.")
                continue

            # Безопасно вычисляем значение политики с текущим контекстом
            policy_value = evaluate_safe_policy_code(
                policy._compiled_safe_code,
                simulation_context
            )
            executed_policy_values[policy.policy_type] = policy_value # Сохраняем результат

            # Применяем эффект, если вычисление прошло успешно
            if policy_value is not None:
                if policy.policy_type == "set_tax_rate":
                    # Обновляем внутреннее состояние модели на основе вычисленного значения
                    # Применяем ограничения уже после вычисления (или в самом выражении через min/max)
                    self.current_tax_rate = float(max(0.0, min(0.5, policy_value))) # Приводим к float
                    print(f"  - Политика '{policy.policy_type}' установила tax_rate = {self.current_tax_rate:.4f} (вычислено: {policy_value})")
                # elif policy.policy_type == "set_transfer_amount":
                #     # self.current_transfer = float(policy_value) ...
                #     pass
            else:
                # Ошибка при вычислении выражения (например, деление на ноль)
                print(f"Ошибка при вычислении значения для политики {policy.id} ('{policy.policy_type}'). Эффект не применен.")
                # В этом случае переменная состояния (self.current_tax_rate) НЕ изменяется

        # --- 2. Основная логика шага экономической модели ---
        # ВВП изменяется под влиянием базового роста и *текущей* (возможно, измененной) налоговой ставки
        growth_modifier = 1.0 - 2 * self.current_tax_rate # ! Пример влияния налога
        current_growth = self.base_growth_rate * growth_modifier
        # Добавляем стохастический шум
        noise = random.uniform(-0.005, 0.005)
        self.gdp *= (1 + current_growth + noise)

        # --- 3. Завершение шага ---
        self.current_step += 1
        # Обновляем полное состояние модели (включая метрики и лог политик)
        self.state = self._update_state()
        # Добавляем текущее состояние в историю
        self.history.append(self.state)

        # Вывод информации о шаге
        print(f"Шаг {self.current_step}: ВВП={self.gdp:.2f}, Нал. ставка={self.current_tax_rate:.3f}")


    def emulate_policy(self, policy: Policy, duration: int, agents_subset: Optional[List[AgentId]] = None) -> Dict[str, Any]:

        """Заглушка для абстрактного метода эмуляции."""
        print("Предупреждение: emulate_policy вызван, но не реализован для SimpleGrowthModel.")
        # В реальной реализации здесь нужно было бы создать копию модели,
        # применить политику и прогнать 'duration' шагов, затем вернуть результат.
        raise NotImplementedError("Метод emulate_policy не реализован для SimpleGrowthModel.")


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
        return metrics

    def get_state_for_agent(self) -> Dict[str, Any]:
        """Возвращает состояние, видимое агенту-правительству, включая параметры модели."""
        state_for_agent = {
            "metrics": self.get_current_metrics(),
            "active_policies": self.active_policies, # Передаем объекты Policy
            # Добавляем параметры самой модели, чтобы агент мог их использовать
            "model_params": {
                'A': self.param_A,
                'B': self.param_B,
                'C': self.param_C,
                'sigma_epsilon': self.param_sigma_epsilon,
                'target_x': self.param_target_x,
                'u_range': self.u_range
            }
        }
        # Добавляем предыдущее значение x, если оно есть в истории
        if len(self.history) > 0:
            prev_metrics = self.history[-1].get("metrics")
            if prev_metrics and "current_x" in prev_metrics:
                state_for_agent["metrics"]["previous_x"] = prev_metrics["current_x"]
        else:
            # Если истории нет, устанавливаем какое-то дефолтное значение
             state_for_agent["metrics"]["previous_x"] = state_for_agent["metrics"]["current_x"]


        return state_for_agent

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
            simulation_context = self.get_state_for_agent()["metrics"] # Используем тот же state, что видит агент

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
        print(f"Шаг {self.current_step}: x={self.current_x:.3f}, u={self.current_u:.3f} (Шок={shock:.3f})")

    def emulate_policy(self, policy: Policy, duration: int, agents_subset: Optional[List[AgentId]] = None) -> Dict[str, Any]:
        """Заглушка для абстрактного метода эмуляции."""
        print("Предупреждение: emulate_policy вызван, но не реализован для LinearStochasticSystem.")
        # В реальной реализации здесь нужно было бы создать копию модели,
        # применить политику и прогнать 'duration' шагов, затем вернуть результат.
        raise NotImplementedError("Метод emulate_policy не реализован для LinearStochasticSystem.")

