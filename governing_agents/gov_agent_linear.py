# gov_agent_linear.py
import random
import time
from typing import Dict, Any, List, Optional, Set, Tuple
import re
import json
import numpy as np # Для расчета MSE/MSU
import traceback

# Импортируем интерфейсы и утилиты
from interfaces import BaseGovernmentAgent, Policy, PolicyDescriptor, BaseEconomicSystem
from policy_utils import validate_and_compile_policy_expression, PolicyValidationError

try:
    from .gemini_with_demonstrations import create_agent, BaseAgent as GeminiBaseAgent
except ImportError as e:
    print(f"ПРЕДУПРЕЖДЕНИЕ из gov_agent_linear.py: Не удалось импортировать .gemini_with_demonstrations. Ошибка: {e}. IntelligentLLMAgent может не работать.")
    GeminiBaseAgent = None
    import traceback
    traceback.print_exc()

# --- Класс для безопасного форматирования ---
class DefaultMapping(dict):
    """Словарь, который возвращает '{key_name}' если ключ отсутствует."""
    def __missing__(self, key):
        # Возвращает сам ключ в фигурных скобках, чтобы было видно в промпте
        # Или можно вернуть пустую строку: return ""
        return f'{{{key}}}'
    # --- Универсальный дефолтный промпт ---

DEFAULT_PROMPT_TEMPLATE = """Ты — AI-агент, управляющий экономической моделью.
Твоя цель: Улучшить состояние системы согласно метрикам, таким как общественное благосостояние или стабильность ключевых показателей.

Доступные типы политик для управления:
{policy_descriptors_text}

Инструкции по формированию выражения для `value_expression`:
- Это должна быть одна строка валидного Python кода.
- Используй ТОЛЬКО переменные из `available_context_vars` для выбранного типа политики.
- Глобально доступны (проверяй `available_context_vars`!): {all_available_context_vars_global}
- Доступные функции: abs(), min(), max(), round(), pow(), math.*, np.clip().
- Результат должен соответствовать `value_type` и `value_range` (если указаны).
- Для 'set_price_ceiling'/'set_price_floor' можно вернуть `None` для отключения.

Текущее состояние экономики (шаг {current_step}):
{current_metrics_text}

История недавних состояний и политик (последние {history_limit} шагов):
{history_text}

Дополнительный контекст (если доступен):
{extra_context_text}

Твое решение (JSON объект):
Проанализируй ситуацию и историю. Предложи ОДНО изменение политики.
Верни JSON объект с ключами "policy_type_id", "value_expression", "reasoning".

Пример JSON ответа:
{{
  "policy_type_id": "ID_типа_политики",
  "value_expression": "выражение_на_python",
  "reasoning": "Краткое объяснение твоего выбора."
}}
"""


# --- Обновленный IntelligentLLMAgent ---
class IntelligentLLMAgent(BaseGovernmentAgent):
    """
    Агент-правительство на основе LLM, использующий gemini.py.
    Рассчитывает KPI и использует параметры модели при формировании промпта.
    """
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.llm_model_name = params.get("model_name", "gemini-1.5-flash-latest")
        self.api_call_delay = params.get("api_call_delay", 4.1)
        self.prompt_template_path = params.get("prompt_template_path", None) # Путь необязателен
        self.temperature = params.get("temperature", 0.5)
        self.max_history_steps_for_prompt = params.get("max_history_steps_for_prompt", 10)
        # Новый параметр: окно для расчета KPI
        self.performance_window = params.get("performance_window", 20)
        self.verbose_llm = params.get("verbose_llm", 1)
        self.llm_style = params.get("llm_style", "default")

        self.llm_client: Optional[GeminiBaseAgent] = None
        if GeminiBaseAgent is not None:
            try:
                self.llm_client = create_agent(
                    provider='google',
                    style=self.llm_style,
                    model_name=self.llm_model_name,
                    temperature=self.temperature,
                    verbose=self.verbose_llm,
                    retries=True,
                    max_attempts=3
                )
                print(f"IntelligentLLMAgent: LLM клиент '{self.llm_model_name}' (стиль: {self.llm_style}) инициализирован.")
            except Exception as e:
                print(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось инициализировать LLM-клиент: {e}")
        else:
            print("КРИТИЧЕСКАЯ ОШИБКА: Модуль gemini.py не импортирован, IntelligentLLMAgent не будет работать.")

        self.prompt_template_content = self._load_prompt_template()
        print(self.prompt_template_content )
        self.policy_counter = 0


    def _load_prompt_template(self) -> str:
        """Загружает шаблон промпта из файла или использует дефолтный."""
        if self.prompt_template_path:
            try:
                with open(self.prompt_template_path, 'r', encoding='utf-8') as f:
                    print(f"Загрузка шаблона промпта из: {self.prompt_template_path}")
                    return f.read()
            except FileNotFoundError:
                print(f"ПРЕДУПРЕЖДЕНИЕ: Файл шаблона промпта не найден: {self.prompt_template_path}. Будет использован дефолтный промпт.")
                return DEFAULT_PROMPT_TEMPLATE
            except Exception as e:
                 print(f"Ошибка при чтении файла промпта {self.prompt_template_path}: {e}. Будет использован дефолтный промпт.")
                 return DEFAULT_PROMPT_TEMPLATE
        else:
             print("Путь к шаблону промпта не указан. Будет использован дефолтный промпт.")
             return DEFAULT_PROMPT_TEMPLATE

    def _calculate_performance_kpis(self, history: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
        """Рассчитывает KPI (MSE, MSU) по последним N шагам истории."""
        mse = None
        msu = None
        window = self.performance_window

        if len(history) >= window:
            recent_history = history[-window:]
            target_x = history[-1].get("metrics",{}).get("target_x", 0.0) # Берем цель из последней записи
            if target_x is None: target_x = 0.0 # Если цели нет, считаем относительно нуля

            x_values = [entry.get("metrics",{}).get("current_x") for entry in recent_history]
            u_values = [entry.get("metrics",{}).get("current_u") for entry in recent_history]

            # Фильтруем None значения
            valid_x = [x for x in x_values if x is not None]
            valid_u = [u for u in u_values if u is not None]

            if valid_x:
                errors_sq = [(x - target_x)**2 for x in valid_x]
                mse = np.mean(errors_sq) if errors_sq else 0.0

            if valid_u:
                u_sq = [u**2 for u in valid_u]
                msu = np.mean(u_sq) if u_sq else 0.0

        return {"current_mse": mse, "current_msu": msu}

    # Методы _format_policy_descriptors_for_prompt и _format_history_for_prompt остаются без изменений

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
            metrics = hist_entry.get("metrics", {})
            metrics_str = ", ".join([f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k,v in metrics.items()])

            policies_active_log = hist_entry.get("active_policies_log", [])
            active_policies_str_parts = []
            if policies_active_log: # active_policies_log это список словарей
                for pol_dict in policies_active_log:
                    active_policies_str_parts.append(f"'{pol_dict.get('policy_type')}' (expr: '{pol_dict.get('value_expression')}')")
            active_policies_str = "; ".join(active_policies_str_parts) if active_policies_str_parts else "Нет активных"

            # Добавляем информацию о решении агента, если она была в логе этого шага
            policy_dec = hist_entry.get('policy_decision')
            decision_str = ""
            if policy_dec:
                decision_str = f" Решение: [{policy_dec.get('policy_type')}: '{policy_dec.get('value_expression')}']"


            text_parts.append(f"  Шаг {step_num}: Метрики({metrics_str}). Политики: [{active_policies_str}].{decision_str}")
        return "\n".join(text_parts) if text_parts else "История пуста."


    def _format_prompt(self,
                   current_state_for_agent: Dict[str, Any],
                   history: List[Dict[str, Any]],
                   policy_descriptors: List[PolicyDescriptor]
                   ) -> str:
        """Формирует промпт, безопасно обрабатывая форматирование."""

        current_metrics = current_state_for_agent.get("metrics", {})
        model_params = current_state_for_agent.get("model_params", {}) # Параметры модели (могут отсутствовать)
        current_step = current_metrics.get("step", "N/A")

        # --- Подготовка данных для форматирования ---

        # 1. Текст дескрипторов политик
        policy_descriptors_text = self._format_policy_descriptors_for_prompt(policy_descriptors)

        # 2. Глобально доступные переменные
        all_vars_global = set(current_metrics.keys())
        for desc in policy_descriptors:
            all_vars_global.update(desc.available_context_vars)
        all_available_context_vars_global = sorted(list(all_vars_global))

        # 3. Текст текущих метрик (безопасное форматирование чисел)
        current_metrics_lines = []
        for k, v in current_metrics.items():
            if isinstance(v, (int, float)):
                # Пытаемся форматировать с 3 знаками после запятой
                try:
                    current_metrics_lines.append(f"  - {k}: {v:.3f}")
                except (TypeError, ValueError): # Если v это None или что-то неформатируемое
                    current_metrics_lines.append(f"  - {k}: {v}") # Выводим как есть
            else:
                current_metrics_lines.append(f"  - {k}: {v}") # Выводим как есть
        current_metrics_text = "\n".join(current_metrics_lines)

        # 4. Текст истории (используем существующий метод)
        history_text = self._format_history_for_prompt(history)

        # 5. Дополнительный контекст (KPI, параметры модели) - форматируем безопасно
        extra_context_lines = []
        # KPI
        performance_kpis = self._calculate_performance_kpis(history)
        mse = performance_kpis.get("current_mse")
        msu = performance_kpis.get("current_msu")
        if mse is not None:
            try:
                extra_context_lines.append(f"  - Производительность (MSE за {self.performance_window} шагов): {mse:.4f}")
            except (TypeError, ValueError):
                extra_context_lines.append(f"  - Производительность (MSE за {self.performance_window} шагов): {mse}")
        if msu is not None:
            try:
                extra_context_lines.append(f"  - Стоимость управления (MSU за {self.performance_window} шагов): {msu:.4f}")
            except (TypeError, ValueError):
                extra_context_lines.append(f"  - Стоимость управления (MSU за {self.performance_window} шагов): {msu}")

        # Параметры модели (если они есть в state_for_agent)
        if model_params:
            extra_context_lines.append("  - Параметры модели:")
            for k, v in model_params.items():
                if isinstance(v, (int, float)):
                    try:
                        extra_context_lines.append(f"    * {k}: {v:.3f}")
                    except (TypeError, ValueError):
                        extra_context_lines.append(f"    * {k}: {v}")
                else:
                    extra_context_lines.append(f"    * {k}: {v}") # Выводим как есть (например, u_range)

        extra_context_text = "\n".join(extra_context_lines) if extra_context_lines else "N/A"


        # --- Создание словаря для format_map ---
        # Используем только те ключи, что реально есть в ШАБЛОНЕ промпта
        def safe_format_float(value, default_val=0.0, precision=3):
            if isinstance(value, (int, float)):
                return f"{value:.{precision}f}"
            return str(value) # Возвращаем как строку, если не число (или None)

        def get_safe(data_dict, key, default_val=None):
            return data_dict.get(key, default_val)

        # Создаем словарь для форматирования промпта
        prompt_data = {
            "policy_descriptors_text": self._format_policy_descriptors_for_prompt(policy_descriptors),
            "all_available_context_vars_global": str(sorted(list(all_vars_global))), # Преобразуем список в строку
            "current_step": str(current_step), # Явно в строку
            "current_metrics_text": "\n".join([f"  - {k}: {safe_format_float(v, precision=3) if isinstance(v, (int,float)) else v}"
                                            for k,v in current_metrics.items()]),
            "history_limit": str(self.max_history_steps_for_prompt),
            "history_text": self._format_history_for_prompt(history),

            # Параметры модели с безопасным форматированием
            "param_A": safe_format_float(get_safe(model_params, 'A'), precision=3),
            "param_B": safe_format_float(get_safe(model_params, 'B'), precision=3),
            "param_C": safe_format_float(get_safe(model_params, 'C'), precision=3),
            "sigma_epsilon": safe_format_float(get_safe(model_params, 'sigma_epsilon'), precision=3),
            "target_x": safe_format_float(get_safe(model_params, 'target_x'), precision=3), # target_x тоже должен быть числом для :.3f
            "u_range": str(get_safe(model_params, 'u_range', "N/A")), # u_range это кортеж, делаем строкой

            # KPI с безопасным форматированием
            "perf_window": str(self.performance_window),
            "current_mse": safe_format_float(performance_kpis.get("current_mse"), precision=4),
            "current_msu": safe_format_float(performance_kpis.get("current_msu"), precision=4),

            # Переменные из current_metrics напрямую (для шаблона, если он их ожидает)
            # Их тоже нужно безопасно форматировать, если шаблон предполагает числовой формат
            "current_x": safe_format_float(current_metrics.get("current_x"), precision=4),
            "previous_x": safe_format_float(current_metrics.get("previous_x"), precision=4),
            "current_u": safe_format_float(current_metrics.get("current_u"), precision=4),
            # Если есть другие метрики, которые шаблон использует с числовым форматированием, их тоже нужно обработать
        }

        # Добавляем остальные метрики, которые могут быть просто строками или неформатируемыми числами
        for k, v in current_metrics.items():
            if k not in prompt_data: # Добавляем, только если еще не добавлено с форматированием
                prompt_data[k] = str(v)


        # Используем DefaultMapping для обработки только тех КЛЮЧЕЙ, которые могут отсутствовать в prompt_data,
        # но значения которых УЖЕ подготовлены и безопасны для простого str.format() без спецификаторов типа
        try:
            safe_prompt_data = DefaultMapping(prompt_data)
            formatted_prompt = self.prompt_template_content.format_map(safe_prompt_data)
            # Заменяем "{ключ}", если DefaultMapping сработал, на что-то вроде "N/A"
            formatted_prompt = re.sub(r'\{[a-zA-Z0-9_]+\}', 'N/A', formatted_prompt)
            return formatted_prompt
        except Exception as e:
            print(f"КРИТИЧЕСКАЯ ОШИБКА при финальном форматировании промпта: {e}. Возвращен неформатированный шаблон.")
            print(f"Данные для форматирования: {prompt_data}")
            print(traceback.format_exc())
            return self.prompt_template_content.format(**DefaultMapping(prompt_data)) # Попытка отдать хоть что-то


    # Метод _parse_llm_response остается без изменений

    def _parse_llm_response(self, response_text: str) -> Optional[Dict[str, str]]:
        # Пытаемся извлечь JSON блок из ответа LLM
        # print(f"DEBUG LLM Raw Response:\n---\n{response_text}\n---")
        match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        json_str = None
        if match:
            json_str = match.group(1)
        else:
            # Если нет ```json```, пытаемся найти JSON объект просто в тексте
            match_plain_json = re.search(r'(\{.*?\})', response_text, re.DOTALL)
            if match_plain_json:
                json_str = match_plain_json.group(1)
            # Если и простого JSON нет, попробуем найти его без начальной { и конечной }
            # Это нужно из-за моделей типа Лаконичного Агента
            else:
                match_inner_json = re.search(r'"policy_type_id":.*"reasoning":\s*".*?"', response_text, re.DOTALL)
                if match_inner_json:
                    json_str = "{" + match_inner_json.group(0) + "}"


        if not json_str:
            print("ПРЕДУПРЕЖДЕНИЕ LLM Agent: Не найден JSON в ответе LLM.")
            # Попробуем найти хотя бы value_expression, если стиль laconic
            if self.llm_style == 'laconic':
                # Ищем просто строку в кавычках или без кавычек после ключевого слова
                 expr_match = re.search(r'(?:value_expression["\']?\s*[:=]?\s*["\']?)(.*?)(?:["\']?\s*,?\s*reasoning|$)', response_text, re.IGNORECASE | re.DOTALL)
                 if expr_match:
                     expr = expr_match.group(1).strip().replace('`','').replace('\n',' ')
                     # Пытаемся угадать policy_type_id (например, единственный доступный)
                     # Это очень ненадежно, нужно указывать в промпте для Laconic явно!
                     # Здесь просто заглушка:
                     guessed_policy_type = "set_control_input" # Или взять из дескрипторов, если он один
                     print(f"ПРЕДУПРЕЖДЕНИЕ: JSON не найден, но в Laconic стиле извлечено выражение: '{expr}'. Угаданный тип: '{guessed_policy_type}'")
                     return {
                         "policy_type_id": guessed_policy_type,
                         "value_expression": expr,
                         "reasoning": "(не извлечено из laconic ответа)"
                     }

            return None # Если не laconic или не нашли выражение

        try:
            # Очистка JSON строки (замена кавычек, удаление комментариев, хвостовых запятых)
            cleaned_json_str = json_str.replace("'", '"')
            # Удаляем однострочные комментарии // и /* ... */ (если вдруг LLM их добавит)
            cleaned_json_str = re.sub(r"//.*?\n", "\n", cleaned_json_str)
            cleaned_json_str = re.sub(r"/\*.*?\*/", "", cleaned_json_str, flags=re.DOTALL)
            # Удаление хвостовых запятых перед } или ]
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
                # Дополнительно чистим value_expression от возможных артефактов
                value_expr_clean = parsed_data["value_expression"].strip().replace('`','').replace('\n',' ')

                return {
                    "policy_type_id": parsed_data["policy_type_id"].strip(),
                    "value_expression": value_expr_clean,
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
                      economic_system: BaseEconomicSystem # Принимаем economic_system
                      ) -> Optional[Policy]:
        """Основной метод принятия решений, использующий KPI и параметры модели."""
        if not self.llm_client:
            print("IntelligentLLMAgent: LLM клиент не инициализирован, политика не может быть предложена.")
            return None

        # Получаем дескрипторы от системы
        try:
             available_descriptors: List[PolicyDescriptor] = economic_system.get_policy_descriptors()
        except Exception as e:
             print(f"IntelligentLLMAgent: Ошибка при получении дескрипторов политик: {e}")
             return None
        if not available_descriptors:
             print("IntelligentLLMAgent: Экономическая система не предоставила доступных политик.")
             return None

        # 1. Формируем промпт с использованием актуальных данных и KPI
        # _format_prompt теперь сам рассчитывает KPI и извлекает параметры
        prompt_str = self._format_prompt(current_state_for_agent, history, available_descriptors)
        # print(f"\nDEBUG: --- LLM Prompt for Step {current_state_for_agent.get('metrics',{}).get('step','N/A')} ---\n{prompt_str}\n--------------------\n")

        # 2. Вызываем LLM
        try:
            llm_response_text = self.llm_client(prompt_str) # Используем __call__

            #print('ЗАПРОС:\n', prompt_str)
            #print('ОТВЕТ:\n',llm_response_text)
            time.sleep(self.api_call_delay) # Задержка после вызова API
        except Exception as e:
            print(f"Критическая ошибка при вызове LLM API: {e}")
            return None # Не можем продолжать без ответа LLM

        # print(f"DEBUG: --- LLM Raw Response ---\n{llm_response_text}\n--------------------")

        # 3. Парсим ответ
        parsed_llm_output = self._parse_llm_response(llm_response_text)
        if not parsed_llm_output:
            print("IntelligentLLMAgent: Не удалось получить корректный JSON от LLM. Изменений нет.")
            return None

        policy_type_id = parsed_llm_output["policy_type_id"]
        expression_string = parsed_llm_output["value_expression"]
        reasoning = parsed_llm_output["reasoning"]
        step_info = current_state_for_agent.get('metrics',{}).get('step','N/A')
        print(f"IntelligentLLMAgent (Шаг {step_info}): LLM предложил: тип='{policy_type_id}', выражение='{expression_string}', обоснование='{reasoning}'")

        # 4. Валидация и компиляция
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
            policy_id = f"llm_policy_{self.policy_counter}_step{step_info}"
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
            print(f"ОШИБКА ВАЛИДАЦИИ IntelligentLLMAgent (Шаг {step_info}): {e}. "
                  f"LLM предложил: тип='{policy_type_id}', выражение='{expression_string}'. Политика не будет изменена.")
            # Здесь можно будет в будущем сохранить 'e' для last_failure_feedback
            return None
        except Exception as e:
            print(f"Неожиданная ошибка в IntelligentLLMAgent (Шаг {step_info}) при компиляции: {e}")
            return None
