# government_agents.py
import random
import time
from typing import Dict, Any, List, Optional, Set, Tuple
import re
import json
import numpy as np

# Импортируем интерфейсы и утилиты
from govsim.utils.interfaces import BaseGovernmentAgent, Policy, PolicyDescriptor, BaseEconomicSystem
from govsim.utils.policy_utils import validate_and_compile_policy_expression, PolicyValidationError

try:
    from govsim.utils.gemini_utils import create_agent, BaseAgent as GeminiBaseAgent
except ImportError as e:
    print(f"ПРЕДУПРЕЖДЕНИЕ из government_agents.py: Не удалось импортировать .gemini_with_demonstrations. Ошибка: {e}. IntelligentLLMAgent может не работать.")
    GeminiBaseAgent = None
    import traceback
    traceback.print_exc()

# --- RandomAgent ---
class RandomAgent(BaseGovernmentAgent):
    """
    Агент-заглушка для тестирования механизма политик с использованием PolicyDescriptor.
    1. Запрашивает у экономической системы список доступных типов политик (дескрипторов).
    2. Случайно выбирает один тип политики.
    3. Генерирует случайное выражение для этого типа, используя информацию из дескриптора
       (доступные переменные, диапазон значений).
    4. Валидирует и компилирует выражение согласно правилам дескриптора.
    5. Возвращает объект Policy.
    """
    def __init__(self, params: Dict[str, Any]):
        """
        Инициализация агента.
        Args:
            params: Словарь параметров из config.py. Ожидает:
                    'change_probability' (float): Вероятность предложения изменений на каждом шаге.
        """
        # possible_policy_types больше не нужен, т.к. получаем их от системы
        self.change_probability = params.get("change_probability", 0.3)
        self.policy_counter = 0 # Счетчик для генерации уникальных ID политик

    def _generate_random_expression(self,
                                    descriptor: PolicyDescriptor,
                                    current_context_values: Dict[str, Any]) -> str:
        """
        Генерирует строку со случайным Python-выражением для ДАННОГО типа политики.
        Использует ТОЛЬКО переменные, разрешенные в descriptor.available_context_vars.
        Пытается учесть descriptor.value_range.

        Args:
            descriptor: Дескриптор выбранного типа политики.
            current_context_values: Словарь с ТЕКУЩИМИ значениями переменных из контекста
                                    симуляции (например, {'gdp': 1050.0, 'step': 10}).

        Returns:
            Строка с Python-выражением.
        """
        policy_type = descriptor.policy_type_id
        allowed_vars = set(descriptor.available_context_vars)
        val_range = descriptor.value_range

        expression = ""
        # Безопасно получаем значения нужных переменных, если они разрешены и доступны
        # Используем .get с дефолтным значением на случай, если метрика еще не посчитана
        tax_rate = current_context_values.get('tax_rate', 0.1) if 'tax_rate' in allowed_vars else 0.1
        gdp = current_context_values.get('gdp', 1000.0) if 'gdp' in allowed_vars else 1000.0
        step = current_context_values.get('step', 0) if 'step' in allowed_vars else 0

        # Генерируем выражение в зависимости от типа политики
        if policy_type == "set_tax_rate":
            choice = random.randint(1, 5)
            base_expr = "" # Базовое выражение до применения ограничений диапазона

            if choice == 1 and 'tax_rate' in allowed_vars:
                change = random.uniform(-0.03, 0.03)
                base_expr = f"tax_rate + {change:.4f}" # Используем имя переменной
            elif choice == 2 and 'step' in allowed_vars and 'math' in self.allowed_math_funcs: # Проверяем доступность math
                base_expr = f"0.15 + math.sin(step / 20.0) * 0.05"
            elif choice == 3 and 'gdp' in allowed_vars:
                # Выражение с зависимостью от gdp
                 base_expr = f"0.1 + (gdp - 1000.0) * 0.0001"
            elif choice == 4:
                # Константное значение
                 base_expr = f"{random.uniform(0.05, 0.3):.4f}"
            elif choice == 5 and 'tax_rate' in allowed_vars:
                 # Возвращаем текущую ставку
                 base_expr = f"tax_rate"
            else:
                 # Если ни один шаблон не подходит или переменные недоступны, ставим константу
                 base_expr = f"{tax_rate:.4f}" # Возвращаем текущее значение как константу

            # Применяем ограничения диапазона, если они есть в дескрипторе
            if val_range is not None:
                 min_val, max_val = val_range
                 # Встраиваем min/max прямо в выражение
                 expression = f"min({max_val}, max({min_val}, {base_expr}))"
            else:
                 # Если диапазона нет, просто используем базовое выражение
                 expression = base_expr

        # elif policy_type == "другой_тип":
             # ... генерация для другого типа ...
        else:
            # Неизвестный тип политики - возвращаем что-то безопасное (например, 0 или None?)
            # Зависит от ожиданий value_type дескриптора
            if descriptor.value_type == float:
                 expression = "0.0"
            elif descriptor.value_type == int:
                 expression = "0"
            else:
                 expression = "None" # Или пустую строку? Нужно решить.

        return expression


    def decide_policy(self,
                      current_state_for_agent: Dict[str, Any],
                      history: List[Dict[str, Any]],
                      economic_system: BaseEconomicSystem,
                      llm_extra_context: Optional[Dict[str, Any]] = None
                      ) -> Optional[Policy]:
        """
        Основной метод принятия решений RandomAgent, использующий PolicyDescriptor.
        """
        # Запрашиваем у системы доступные дескрипторы политик
        try:
             available_descriptors: List[PolicyDescriptor] = economic_system.get_policy_descriptors()
        except Exception as e:
             print(f"Ошибка при получении дескрипторов политик от системы: {e}")
             return None # Не можем работать без дескрипторов

        if not available_descriptors:
             print("RandomAgent: Экономическая система не предоставила доступных политик.")
             return None

        # С вероятностью change_probability решаем предложить изменение
        if random.random() < self.change_probability:
            print("RandomAgent: Решено предложить изменение политики.")

            # 1. Случайно выбираем один из доступных дескрипторов
            selected_descriptor = random.choice(available_descriptors)
            policy_type_id = selected_descriptor.policy_type_id
            print(f"RandomAgent: Выбран тип политики '{policy_type_id}' (описание: {selected_descriptor.description})")

            # 2. Генерируем строку с выражением, используя информацию дескриптора
            # Передаем текущие метрики, чтобы генератор мог использовать актуальные значения
            current_metrics = current_state_for_agent.get("metrics", {})
            expression_string = self._generate_random_expression(selected_descriptor, current_metrics)

            if not expression_string:
                 print(f"RandomAgent: Не удалось сгенерировать выражение для типа '{policy_type_id}', изменений нет.")
                 return None

            print(f"RandomAgent: Сгенерировано выражение: {expression_string}")

            # 3. Валидация и компиляция сгенерированного выражения
            # Используем набор переменных, разрешенных ДЕСКРИПТОРОМ для этого типа политики
            context_vars_for_validation: Set[str] = set(selected_descriptor.available_context_vars)

            try:
                # Вызываем утилиту для валидации и компиляции
                compiled_code = validate_and_compile_policy_expression(
                    expression_string,
                    context_vars_for_validation # <<<--- Валидируем по правилам дескриптора
                )

                # 4. Если успешно, создаем объект Policy
                self.policy_counter += 1
                policy_id = f"random_policy_{self.policy_counter}"
                # Используем описание из дескриптора + доп. инфо
                description = f"Random rule {self.policy_counter} for: {selected_descriptor.description}"

                new_policy = Policy(
                    id=policy_id,
                    description=description,
                    policy_type=policy_type_id, # Используем ID из дескриптора
                    value_expression=expression_string,
                    _compiled_safe_code=compiled_code # Сохраняем результат компиляции
                    # target_selector пока оставляем 'default'
                )
                print(f"RandomAgent: Политика '{policy_id}' успешно скомпилирована.")
                return new_policy # Возвращаем готовую политику

            except PolicyValidationError as e:
                print(f"ОШИБКА ВАЛИДАЦИИ RandomAgent: {e}. Выражение: '{expression_string}'. Политика не будет изменена.")
                return None
            except Exception as e:
                print(f"Неожиданная ошибка в RandomAgent при компиляции: {e}")
                return None

        else:
            # Агент решил не генерировать новую политику на этом шаге
            print("RandomAgent: Решено не менять политику.")
            return None

    # Добавляем заглушку для свойства, которое используется в _generate_random_expression
    # В реальном LLMAgent это можно сделать по-другому
    @property
    def allowed_math_funcs(self):
         # Это нужно для проверки доступности math.sin в генераторе
         # В реальном LLM можно передать список доступных функций в промпт
         from policy_utils import ALLOWED_MATH_NAMES
         return ALLOWED_MATH_NAMES


# --- Агент, который НИЧЕГО не делает ---
class StaticPolicyAgent(BaseGovernmentAgent):
    """
    Агент-заглушка, который никогда не предлагает изменений политики.
    Используется для создания базового сценария симуляции (baseline),
    где правительство пассивно и не вмешивается в экономику
    после начальной установки политик (если они были).
    """
    def __init__(self, params: Dict[str, Any]):
        """
        Инициализация агента. Параметры не используются.
        Args:
            params: Словарь параметров из config.py (игнорируется).
        """
        super().__init__(params) # Вызов инициализатора базового класса (хотя он пустой)
        print("StaticPolicyAgent: Инициализирован. Политики изменяться не будут.")

    def decide_policy(self,
                      current_state_for_agent: Dict[str, Any],
                      history: List[Dict[str, Any]],
                      economic_system: BaseEconomicSystem,
                      llm_extra_context: Optional[Dict[str, Any]] = None
                      ) -> Optional[Policy]:
        """
        Метод принятия решений. Всегда возвращает None.
        """
        # Этот агент не анализирует состояние и не предлагает изменений.

        return None


class TestPoliciesAgent(BaseGovernmentAgent):
    """
    Агент для тестирования конкретной политики.
    Политика (тип и выражение) задается при инициализации.
    Выполняет валидацию и компиляцию как LLM-агент.
    """
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.test_policy_data = params.get("test_policy_dict")
        if not self.test_policy_data or \
           not isinstance(self.test_policy_data, dict) or \
           "policy_type_id" not in self.test_policy_data or \
           "value_expression" not in self.test_policy_data:
            raise ValueError("TestPoliciesAgent требует 'test_policy_dict' в параметрах "
                             "с ключами 'policy_type_id' и 'value_expression'.")

        self.policy_type_id_to_test = self.test_policy_data["policy_type_id"]
        self.value_expression_to_test = self.test_policy_data["value_expression"]
        self.reasoning_for_test = self.test_policy_data.get("reasoning", "Тестовая политика.") # Опционально

        self.policy_counter = 0 # Для уникальных ID политик
        print(f"TestPoliciesAgent: Инициализирован для тестирования политики типа "
              f"'{self.policy_type_id_to_test}' с выражением '{self.value_expression_to_test}'")

    def decide_policy(self,
                      current_state_for_agent: Dict[str, Any],
                      history: List[Dict[str, Any]],
                      economic_system: BaseEconomicSystem,
                      llm_extra_context: Optional[Dict[str, Any]] = None
                      ) -> Optional[Policy]:

        print('Тестовый агент вызван')

        # Получаем дескрипторы от экономической системы
        try:
             available_descriptors: List[PolicyDescriptor] = economic_system.get_policy_descriptors()
        except Exception as e:
             print(f"TestPoliciesAgent: Ошибка при получении дескрипторов политик: {e}")
             return None

        if not available_descriptors:
             print("TestPoliciesAgent: Экономическая система не предоставила доступных политик.")
             return None

        # Находим дескриптор для нашей тестовой политики
        selected_descriptor = next((d for d in available_descriptors if d.policy_type_id == self.policy_type_id_to_test), None)

        if not selected_descriptor:
            print(f"ПРЕДУПРЕЖДЕНИЕ TestPoliciesAgent: Заданный для теста policy_type_id "
                  f"'{self.policy_type_id_to_test}' не найден среди доступных дескрипторов. Отклонено.")
            return None

        # Валидация и компиляция (аналогично IntelligentLLMAgent)
        context_vars_for_validation: Set[str] = set(selected_descriptor.available_context_vars)
        try:
            compiled_code = validate_and_compile_policy_expression(
                self.value_expression_to_test,
                context_vars_for_validation
            )

            self.policy_counter += 1
            # Формируем уникальный ID, включающий шаг, чтобы видеть, когда создается новый объект Policy
            current_step_metric = current_state_for_agent.get("metrics", {}).get("step", "N/A")
            policy_id = f"test_policy_{self.policy_counter}_s{current_step_metric}"
            description = f"Тест ({self.reasoning_for_test}): {selected_descriptor.description}"

            new_policy = Policy(
                id=policy_id,
                description=description,
                policy_type=self.policy_type_id_to_test, # Тип из настроек агента
                value_expression=self.value_expression_to_test, # Выражение из настроек агента
                _compiled_safe_code=compiled_code
            )
            # Раскомментируйте для детальной отладки каждого решения
            print(f"TestPoliciesAgent: Политика '{policy_id}' (тип: {self.policy_type_id_to_test}) "
                  f"успешно скомпилирована для шага {current_step_metric}.")
            return new_policy

        except PolicyValidationError as e:
            current_step_metric = current_state_for_agent.get("metrics", {}).get("step", "N/A")
            print(f"ОШИБКА ВАЛИДАЦИИ TestPoliciesAgent (Шаг: {current_step_metric}): {e}. "
                  f"Тестируемая политика: тип='{self.policy_type_id_to_test}', "
                  f"выражение='{self.value_expression_to_test}'. Политика не будет применена.")
            return None
        except Exception as e:
            current_step_metric = current_state_for_agent.get("metrics", {}).get("step", "N/A")
            print(f"Неожиданная ошибка в TestPoliciesAgent (Шаг: {current_step_metric}) при компиляции: {e}")
            return None

