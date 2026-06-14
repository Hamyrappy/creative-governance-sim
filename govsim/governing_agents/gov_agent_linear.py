# gov_agent_linear.py
import random
import time
from typing import Dict, Any, List, Optional, Set, Tuple
import re
import json
import numpy as np # Для расчета MSE/MSU
import traceback

# Импортируем интерфейсы и утилиты
from govsim.utils.interfaces import BaseGovernmentAgent, Policy, PolicyDescriptor, BaseEconomicSystem
from govsim.economic_models.linear_stochastic_system import LinearSystemAgentContext
from govsim.utils.policy_utils import validate_and_compile_policy_expression, PolicyValidationError
from govsim.utils.prompts_utils import DefaultMapping, DEFAULT_PROMPT_TEMPLATE, load_prompt_template, format_policy_descriptors_for_prompt

# ! С импортом модуля Gemini бывают проблемы при отсутствии соединения с API GoogleAI, например при отсутствии API ключа или блокировке
try:
    # Legacy Gemini path has been removed (the project migrated to the OpenAI-compatible client
    # in govsim.core.llm). This import is kept tolerant so the old simulation path still imports;
    # IntelligentLLMAgent is being replaced by an LLMRegent on the new core (agents/09-grand-plan.md).
    from govsim.utils.gemini_utils import create_agent, BaseAgent as GeminiBaseAgent  # type: ignore
except ModuleNotFoundError:
    create_agent = None  # type: ignore
    GeminiBaseAgent = None  # type: ignore 
    

# --- IntelligentLLMAgent, предназначен для линейной стохастической модели мира ---
class IntelligentLLMAgent(BaseGovernmentAgent):
    """
    Агент-правительство на основе LLM, использующий gemini.py.
    Рассчитывает KPI и использует параметры модели при формировании промпта.
    """
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)
        self.llm_model_name = params.get("model_name", "gemini-2.5-flash-lite")
        self.api_call_delay = params.get("api_call_delay", 4.1)
        self.prompt_template_path = params.get("prompt_template_path") # Путь обязателен
        self.temperature = params.get("temperature", 0.9)
        self.max_history_steps_for_prompt = params.get("max_history_steps_for_prompt", 10)
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

        self.prompt_template_content = load_prompt_template(self.prompt_template_path)
        print(self.prompt_template_content)
        self.policy_counter = 0

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

    # TODO Улучшить эту функцию, она слишком примитивна. Малоэффективно выводить историю в таком виде
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


    def _format_prompt(self, context: LinearSystemAgentContext) -> str:
        """
        Сердце агента. Принимает один 
        типизированный объект контекста.
        """
        prompt_data = context.model_dump()

        # Используем DefaultMapping на случай, если в шаблоне есть лишние ключи
        safe_prompt_data = DefaultMapping(prompt_data)
        formatted_prompt = self.prompt_template_content.format_map(safe_prompt_data)
        return formatted_prompt


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
            cleaned_json_str = json_str
            #cleaned_json_str = json_str.replace("'", '"')
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

        # Проверяем, что нам пришел ожидаемый объект контекста
        if not isinstance(current_state_for_agent, LinearSystemAgentContext):
            print(f"ОШИБКА: IntelligentLLMAgent ожидал контекст типа LinearSystemAgentContext, "
                  f"но получил {type(current_state_for_agent)}. Агент пропускает ход.")
            return None

        # Теперь мы можем безопасно работать с current_state_for_agent как с context
        context: LinearSystemAgentContext = current_state_for_agent

        # 0. Заполняем контекст
        # 0.1 Рассчитываем KPI
        kpi_data = self._calculate_performance_kpis(history)
        context.current_mse = kpi_data.get("current_mse")
        context.current_msu = kpi_data.get("current_msu")
        context.perf_window = self.performance_window

        # 0.2 Форматируем историю и дескрипторы
        descriptors = economic_system.get_policy_descriptors()
        context.history_text = self._format_history_for_prompt(history)
        context.policy_descriptors_text = self._format_policy_descriptors_for_prompt(descriptors)

        # 0.3
        all_vars_global = set(context.model_dump(exclude_none=True).keys())
        for desc in descriptors:
            all_vars_global.update(desc.available_context_vars)
        context.all_available_context_vars_global = sorted(list(all_vars_global))
        
        # 1 Формируем промпт, передавая единый объект контекста
        prompt_str = self._format_prompt(context)

        # 2. Вызываем LLM
        try:
            llm_response_text = self.llm_client(prompt_str) # Используем __call__

            # print('ЗАПРОС:\n', prompt_str)
            # print('ОТВЕТ:\n',llm_response_text)
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
        step_info = context.current_step
        print(f"IntelligentLLMAgent (Шаг {step_info}): LLM предложил: тип='{policy_type_id}', выражение='{expression_string}', обоснование='{reasoning}'")

        # 4. Валидация и компиляция
        available_descriptors = economic_system.get_policy_descriptors()

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
