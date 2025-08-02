# government_agents.py
import random
import time
from typing import Dict, Any, List, Optional, Set, Tuple
import re
import json
import numpy as np

# Импортируем интерфейсы и утилиты
from interfaces import BaseGovernmentAgent, Policy, PolicyDescriptor, BaseEconomicSystem
from policy_utils import validate_and_compile_policy_expression, PolicyValidationError

try:
    from .gemini_with_demonstrations import create_agent, BaseAgent as GeminiBaseAgent
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

class IntelligentLLMAgent(BaseGovernmentAgent):
    """
    Агент-правительство на основе LLM, использующий gemini.py.
    """
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params) # Вызов инициализатора базового класса
        self.llm_model_name = params.get("model_name", "gemini-1.5-flash-latest")
        self.api_call_delay = params.get("api_call_delay", 4.1)
        self.prompt_template_path = params.get("prompt_template_path", "prompts/default_gov_prompt.md")
        self.temperature = params.get("temperature", 0.6) # Чуть выше для креативности
        self.max_history_steps_for_prompt = params.get("max_history_steps_for_prompt", 5)
        self.verbose_llm = params.get("verbose_llm", 1)
        self.llm_style = params.get("llm_style", "default") # 'default' или 'laconic'

        self.llm_client: Optional[GeminiBaseAgent] = None
        if GeminiBaseAgent is not None:
            try:
                self.llm_client = create_agent(
                    provider='google', # Как в вашем gemini.py
                    style=self.llm_style,
                    model_name=self.llm_model_name,
                    temperature=self.temperature,
                    verbose=self.verbose_llm,
                    retries=True, # Включаем ретраи по умолчанию
                    max_attempts=3
                )
                print(f"IntelligentLLMAgent: LLM клиент '{self.llm_model_name}' (стиль: {self.llm_style}) инициализирован.")
            except Exception as e:
                print(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось инициализировать LLM-клиент для IntelligentLLMAgent: {e}")
        else:
            print("КРИТИЧЕСКАЯ ОШИБКА: Модуль gemini.py не импортирован, IntelligentLLMAgent не будет работать.")

        self.prompt_template_content = self._load_prompt_template()
        self.policy_counter = 0

    def _load_prompt_template(self) -> str:
        try:
            with open(self.prompt_template_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            print(f"ПРЕДУПРЕЖДЕНИЕ: Файл шаблона промпта не найден: {self.prompt_template_path}. Будет использован дефолтный промпт.")
            # Возвращаем дефолтный промпт-строку здесь, если файл не найден
            # (Это базовый промпт, его нужно будет сильно улучшать)
            return """Ты — агент-правительство, управляющий экономической моделью.
Твоя цель: максимизировать social_welfare.

Доступные типы политик:
{policy_descriptors_text}

Инструкции по `value_expression`:
- Python-выражение.
- Используй переменные из `available_context_vars` для выбранного типа политики.
- Глобально доступны (но используй с умом, проверяя `available_context_vars`!): {all_available_context_vars_global}
- Функции: abs(), min(), max(), round(), pow(), math.*, np.clip().
- Для 'set_price_ceiling'/'set_price_floor' можно вернуть `None` для отключения.

Текущее состояние (шаг {current_step}):
{current_metrics_text}

История (последние {history_limit} шагов):
{history_text}

Твое решение (JSON-объект с "policy_type_id", "value_expression", "reasoning"):
```json
""" # Подсказка LLM начать с JSON

    def _format_policy_descriptors_for_prompt(self, policy_descriptors: List[PolicyDescriptor]) -> str:
        text_parts = []
        for desc in policy_descriptors:
            part = f"- policy_type_id: \"{desc.policy_type_id}\"\n"
            part += f"  description: \"{desc.description}\"\n"
            part += f"  value_type: {desc.value_type.__name__}\n"
            if desc.value_range:
                part += f"  value_range: {desc.value_range}\n"
            part += f"  available_context_vars_for_this_policy: {desc.available_context_vars}\n"
            if desc.constraints:
                part += f"  constraints: {desc.constraints}\n"
            text_parts.append(part)
        return "\n".join(text_parts)

    def _format_history_for_prompt(self, history: List[Dict[str, Any]]) -> str:
        text_parts = []
        # Берем последние N шагов из истории
        start_index = max(0, len(history) - self.max_history_steps_for_prompt)
        for i, hist_entry in enumerate(history[start_index:]):
            step_num = hist_entry.get("step", "N/A")
            metrics_str = ", ".join([f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k,v in hist_entry.get("metrics", {}).items()])
            
            policies_active_log = hist_entry.get("active_policies_log", [])
            active_policies_str_parts = []
            if policies_active_log: # active_policies_log это список словарей
                for pol_dict in policies_active_log:
                    active_policies_str_parts.append(f"'{pol_dict.get('policy_type')}' (expr: '{pol_dict.get('value_expression')}')")
            active_policies_str = "; ".join(active_policies_str_parts) if active_policies_str_parts else "Нет активных"

            text_parts.append(f"  Шаг {step_num}: Метрики({metrics_str}). Политики: [{active_policies_str}]")
        return "\n".join(text_parts) if text_parts else "История пуста."


    def _format_prompt(self,
                       current_state_for_agent: Dict[str, Any],
                       history: List[Dict[str, Any]],
                       policy_descriptors: List[PolicyDescriptor]
                       ) -> str:
        current_metrics = current_state_for_agent.get("metrics", {})
        current_step = current_metrics.get("step", "N/A")
        
        # Собираем все уникальные имена переменных, которые В ПРИНЦИПЕ могут быть доступны
        all_vars_global = set()
        for desc in policy_descriptors:
            all_vars_global.update(desc.available_context_vars)
        # Добавляем также ключи из текущих метрик (на всякий случай, если в дескрипторах что-то упущено)
        all_vars_global.update(current_metrics.keys())


        prompt_data = {
            "goal_description": "Максимизировать 'social_welfare', поддерживая стабильность других ключевых показателей.",
            "policy_descriptors_text": self._format_policy_descriptors_for_prompt(policy_descriptors),
            "all_available_context_vars_global": sorted(list(all_vars_global)),
            "current_step": current_step,
            "current_metrics_text": "\n".join([f"  - {k}: {v:.3f}" if isinstance(v, float) else f"  - {k}: {v}" for k,v in current_metrics.items()]),
            "history_limit": self.max_history_steps_for_prompt,
            "history_text": self._format_history_for_prompt(history)
        }
        
        return self.prompt_template_content.format(**prompt_data)

    def _parse_llm_response(self, response_text: str) -> Optional[Dict[str, str]]:
        # Пытаемся извлечь JSON блок из ответа LLM
        # print(f"DEBUG LLM Raw Response:\n---\n{response_text}\n---")
        match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        json_str = None
        if match:
            json_str = match.group(1)
        else:
            # Если нет ```json```, пытаемся найти JSON объект просто в тексте
            # (это менее надежно)
            match_plain_json = re.search(r'(\{.*?\})', response_text, re.DOTALL)
            if match_plain_json:
                json_str = match_plain_json.group(1)

        if not json_str:
            print("ПРЕДУПРЕЖДЕНИЕ LLM Agent: Не найден JSON в ответе LLM.")
            return None

        try:
            # Заменяем одинарные кавычки на двойные, если LLM их использует в JSON
            # (некоторые модели могут так делать)
            # А также обрабатываем возможные лишние запятые в конце списков/словарей
            cleaned_json_str = json_str.replace("'", '"')
            # Удаление хвостовых запятых (очень упрощенно, может сломать строки с запятыми)
            cleaned_json_str = re.sub(r',\s*([\}\]])', r'\1', cleaned_json_str)

            parsed_data = json.loads(cleaned_json_str)
            if isinstance(parsed_data, dict) and \
               "policy_type_id" in parsed_data and \
               "value_expression" in parsed_data and \
               "reasoning" in parsed_data:
                # Простая проверка типов
                if not isinstance(parsed_data["policy_type_id"], str) or \
                   not isinstance(parsed_data["value_expression"], str) or \
                   not isinstance(parsed_data["reasoning"], str):
                    print("ПРЕДУПРЕЖДЕНИЕ LLM Agent: Некорректные типы данных в JSON от LLM.")
                    return None
                return {
                    "policy_type_id": parsed_data["policy_type_id"].strip(),
                    "value_expression": parsed_data["value_expression"].strip(),
                    "reasoning": parsed_data["reasoning"].strip()
                }
            else:
                print("ПРЕДУПРЕЖДЕНИЕ LLM Agent: JSON от LLM не содержит всех необходимых ключей (policy_type_id, value_expression, reasoning).")
                return None
        except json.JSONDecodeError as e:
            print(f"ОШИБКА LLM Agent: Не удалось распарсить JSON из ответа LLM: {e}\nСтрока JSON: '{json_str}'")
            return None

    def decide_policy(self,
                      current_state_for_agent: Dict[str, Any],
                      history: List[Dict[str, Any]],
                      economic_system: BaseEconomicSystem,
                      llm_extra_context: Optional[Dict[str, Any]] = None,
                      ) -> Optional[Policy]:
        if random.random() < 0.9:
            return None
        
        if not self.llm_client:
            print("IntelligentLLMAgent: LLM клиент не инициализирован, политика не может быть предложена.")
            return None

        try:
             available_descriptors: List[PolicyDescriptor] = economic_system.get_policy_descriptors()
        except Exception as e:
             print(f"IntelligentLLMAgent: Ошибка при получении дескрипторов политик: {e}")
             return None

        if not available_descriptors:
             print("IntelligentLLMAgent: Экономическая система не предоставила доступных политик.")
             return None

        # 1. Формируем промпт
        prompt_str = self._format_prompt(current_state_for_agent, history, available_descriptors)
        # print(f"DEBUG: --- LLM Prompt for Step {current_state_for_agent.get('metrics',{}).get('step','N/A')} ---\n{prompt_str}\n--------------------")

        # 2. Вызываем LLM
        llm_response_text = self.llm_client(prompt_str) # Используем __call__ из gemini.BaseAgent
        time.sleep(self.api_call_delay)
        # print(f"DEBUG: --- LLM Raw Response ---\n{llm_response_text}\n--------------------")


        # 3. Парсим ответ
        parsed_llm_output = self._parse_llm_response(llm_response_text)

        if not parsed_llm_output:
            print("IntelligentLLMAgent: Не удалось получить корректный JSON от LLM. Изменений нет.")
            return None

        policy_type_id = parsed_llm_output["policy_type_id"]
        expression_string = parsed_llm_output["value_expression"]
        reasoning = parsed_llm_output["reasoning"]
        print(f"IntelligentLLMAgent (Шаг {current_state_for_agent.get('metrics',{}).get('step','N/A')}): "
              f"LLM предложил: тип='{policy_type_id}', выражение='{expression_string}', обоснование='{reasoning}'")


        # 4. Валидация и компиляция сгенерированного выражения
        selected_descriptor = next((d for d in available_descriptors if d.policy_type_id == policy_type_id), None)

        if not selected_descriptor:
            print(f"ПРЕДУПРЕЖДЕНИЕ IntelligentLLMAgent: LLM предложил неизвестный policy_type_id '{policy_type_id}'. Отклонено.")
            return None

        context_vars_for_validation: Set[str] = set(selected_descriptor.available_context_vars)
        try:
            compiled_code = validate_and_compile_policy_expression(
                expression_string,
                context_vars_for_validation
            )
            self.policy_counter += 1
            policy_id = f"llm_policy_{self.policy_counter}_step{current_state_for_agent.get('metrics',{}).get('step','N/A')}"
            # Используем описание из дескриптора + обоснование от LLM
            description = f"LLM ({reasoning}): {selected_descriptor.description}"

            new_policy = Policy(
                id=policy_id,
                description=description,
                policy_type=policy_type_id,
                value_expression=expression_string,
                _compiled_safe_code=compiled_code
            )
            print(f"IntelligentLLMAgent: Политика '{policy_id}' (тип: {policy_type_id}) успешно скомпилирована.")
            return new_policy

        except PolicyValidationError as e:
            print(f"ОШИБКА ВАЛИДАЦИИ IntelligentLLMAgent: {e}. "
                  f"LLM предложил: тип='{policy_type_id}', выражение='{expression_string}'. Политика не будет изменена.")
            return None
        except Exception as e:
            print(f"Неожиданная ошибка в IntelligentLLMAgent при компиляции: {e}")
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

