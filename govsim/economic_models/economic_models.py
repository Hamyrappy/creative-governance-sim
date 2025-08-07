# economic_models.py
import random
from typing import Dict, Any, List, Optional, Set, Tuple

from govsim.utils.interfaces import BaseEconomicSystem, Policy, AgentId, PolicyDescriptor
from govsim.utils.policy_utils import evaluate_safe_policy_code


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
