import os
import json
import warnings
from typing import Any, Dict, Type, Union
import time # Для возможной задержки при rate limiting

from dotenv import load_dotenv

# LangChain core imports
from langchain.base_language import BaseLanguageModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables.retry import RunnableRetry
from langchain_core.output_parsers.json import JsonOutputParser
from pydantic import BaseModel, Field # Используем Pydantic v2+

# Google GenAI SDK
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
# Импортируем нужные исключения
from google.api_core.exceptions import InvalidArgument, ServiceUnavailable, ResourceExhausted, GoogleAPIError


# --- ENV и константы ---
DEFAULT_FAST_MODEL = "models/gemini-2.0-flash-exp"
DEFAULT_STRUCTURED_MODEL = "models/gemini-1.5-flash-latest"
DEFAULT_MAX_ATTEMPTS = 3
# Уровни детализации вывода
# 0: Только финальные результаты демо
# 1: Информация о создании агентов, основные ошибки API/выполнения
# 2: Raw ответы Laconic, JSON от Checker, предупреждения о маркерах
DEFAULT_VERBOSE_LEVEL = 1 # Уровень по умолчанию для агентов

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY не найден в .env")

genai.configure(api_key=API_KEY)

LOW_SAFETY = {
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

# --- Выбор модели --- 
def select_model(structured_output_required: bool = False, verbose: int = 0) -> str:
    """Выбирает подходящую модель Gemini."""
    prefs_fast = [
         'gemini-2.0-flash-thinking-exp', 'gemini-2.0-flash-exp',
         'gemini-2.0-pro-exp', 'gemini-1.5-pro', 'gemini-1.5-flash',
    ]
    prefs_structured = [
        'gemini-1.5-pro', 'gemini-1.5-flash', 'gemini-2.0-flash',
    ]
    preferred_list = prefs_structured if structured_output_required else prefs_fast
    default_fallback = DEFAULT_STRUCTURED_MODEL if structured_output_required else DEFAULT_FAST_MODEL

    try:
        available_models = {m.name.split('/')[-1]: m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods}
        prefs_with_latest = []
        for p in preferred_list:
            prefs_with_latest.append(p + "-latest")
            prefs_with_latest.append(p)

        for pref in prefs_with_latest:
            matching_models = {s: f for s, f in available_models.items() if s.startswith(pref)}
            if matching_models:
                 best_match_short = sorted(matching_models.keys(), key=len, reverse=True)[0]
                 return available_models[best_match_short]

        fallback_prefs_category = ['gemini-1.5-pro', 'gemini-1.5-flash'] if structured_output_required else ['gemini-1.5-flash']
        for fallback_pref in fallback_prefs_category:
             for short_name, full_name in available_models.items():
                  if short_name.startswith(fallback_pref):
                       return full_name
        return default_fallback

    except Exception as e:
        if verbose >= 1:
            warnings.warn(f"Ошибка при выборе модели: {e}. Используется fallback: {default_fallback}")
        return default_fallback

# --- ОБЕРТКА ДЛЯ ПОВТОРНЫХ ПОПЫТОК ---
def _wrap_with_retry(llm: BaseLanguageModel, enable_retry: bool, max_attempts: int = DEFAULT_MAX_ATTEMPTS) -> BaseLanguageModel:
    """Оборачивает LLM в логику повторных попыток."""
    if not enable_retry:
        return llm
    retryable_exceptions = (ServiceUnavailable, ResourceExhausted)
    return RunnableRetry(
        bound=llm, max_attempt_number=max_attempts, wait_exponential_jitter=True,
        initial_wait_s=10, max_wait_s=45, retry_exception_types=retryable_exceptions,
    )

# --- БАЗОВЫЙ АГЕНТ ---
class BaseAgent:
    """Базовый агент: отправляет промпт, возвращает текст."""
    def __init__(self, llm: BaseLanguageModel, retries: bool = True,
                 max_attempts: int = DEFAULT_MAX_ATTEMPTS, verbose: int = DEFAULT_VERBOSE_LEVEL):
        self._llm = llm
        self.retries_enabled = retries
        self.max_attempts = max_attempts
        self.verbose = verbose # Уровень детализации для экземпляра
        self.model_name = getattr(llm, 'model', 'unknown')
        self.llm = _wrap_with_retry(llm, self.retries_enabled, self.max_attempts)

    def __call__(self, prompt: str, **kwargs) -> str:
        """Синхронный вызов."""
        try:
            if self.verbose >= 3: # Очень подробный уровень для отладки промптов
                print(f"--- Prompt ({self.model_name}) ---\n{prompt}\n---")
            res = self.llm.invoke(prompt, **kwargs)
            return self._extract(res)
        except GoogleAPIError as e:
            if self.verbose >= 1:
                print(f"\n!!! Ошибка API Google ({self.model_name}): {type(e).__name__} - {e}")
            if isinstance(e, InvalidArgument): raise
            return f"Ошибка API: {e}" # Возвращаем ошибку текстом
        except Exception as e:
            if self.verbose >= 1:
                print(f"\n!!! Ошибка выполнения ({self.model_name}): {type(e).__name__} - {e}")
            return f"Ошибка выполнения: {e}" # Возвращаем ошибку текстом

    async def acall(self, prompt: str, **kwargs) -> str:
        """Асинхронный вызов."""
        try:
            if self.verbose >= 3:
                 print(f"--- Prompt (async, {self.model_name}) ---\n{prompt}\n---")
            res = await self.llm.ainvoke(prompt, **kwargs)
            return self._extract(res)
        except GoogleAPIError as e:
            if self.verbose >= 1:
                print(f"\n!!! Ошибка API Google (async, {self.model_name}): {type(e).__name__} - {e}")
            if isinstance(e, InvalidArgument): raise
            return f"Ошибка API: {e}"
        except Exception as e:
            if self.verbose >= 1:
                print(f"\n!!! Ошибка выполнения (async, {self.model_name}): {type(e).__name__} - {e}")
            return f"Ошибка выполнения: {e}"

    def structured(self, prompt: str, schema: Union[Type[BaseModel], Dict[str, Any]]) -> Any:
        """Возвращает структурированный вывод (JSON/Pydantic)."""
        if not hasattr(self._llm, 'with_structured_output'):
            raise AttributeError(f"Модель {self.model_name} не поддерживает structured output.")
        try:
            if self.verbose >= 3:
                 print(f"--- Structured Prompt ({self.model_name}) ---\n{prompt}\n---")
            structured_llm = self._llm.with_structured_output(schema, include_raw=False)
            final_runnable = _wrap_with_retry(structured_llm, self.retries_enabled, self.max_attempts) \
                             if self.retries_enabled else structured_llm
            return final_runnable.invoke(prompt)
        except InvalidArgument as e:
             if "Function calling is not enabled" in str(e) or "tool" in str(e).lower():
                  error_msg = f"Модель {self.model_name} не поддерживает structured output (tools)."
                  if self.verbose >= 1: print(f"\n!!! КРИТИЧЕСКАЯ ОШИБКА: {error_msg}")
                  raise NotImplementedError(error_msg) from e
             else:
                  if self.verbose >= 1: print(f"\n!!! Ошибка InvalidArgument ({self.model_name}): {e}")
                  raise
        except GoogleAPIError as e:
             if self.verbose >= 1: print(f"\n!!! Ошибка API Google (structured, {self.model_name}): {type(e).__name__} - {e}")
             return {"error": f"API Error: {e}"} # Возвращаем ошибку
        except Exception as e:
             if self.verbose >= 1: print(f"\n!!! Ошибка выполнения (structured, {self.model_name}): {type(e).__name__} - {e}")
             return {"error": f"Execution Error: {e}"} # Возвращаем ошибку

    @staticmethod
    def _extract(res: Any) -> str:
        """Извлекает текстовое содержимое из ответа LangChain."""
        if isinstance(res, str): return res
        if hasattr(res, 'content'): return str(res.content)
        if hasattr(res, 'text'): return str(res.text)
        return str(res)

# --- ЛАКОНИЧНЫЙ АГЕНТ ---
class LaconicAgent(BaseAgent):
    """Агент, возвращающий только финальный ответ между @@@."""
    TEMPLATE = '''
###INSTRUCTIONS###

You MUST follow the instructions for answering:

- ALWAYS answer in the LANGUAGE of the TASK.
- Main part of the message (TASK) is located between triple backticks.
- You ALWAYS will be PENALIZED for wrong and low-effort answers.
- ALWAYS follow "Answering rules."
- ALWAYS highlight the final answer with triple @ symbol on both sides.
- NEVER use bold or italic in the final answer
- Your answer is CRITICAL for my career.
- Answer the question in a NATURAL, human-like manner.

###Answering Rules###

Follow in the strict order:

1. In the FINAL ANSWER use the LANGUAGE of my TASK.
2. **ONCE PER CHAT** assign a real-world expert role to yourself before answering, e.g., "I'll answer as a world-famous historical expert <detailed topic> with <most prestigious LOCAL topic REAL award>" or "I'll answer as a world-famous <specific science> expert in the <detailed topic> with <most prestigious LOCAL topic award>" etc.
3. Read the task thoroughly.
4. REPHRASE the task to show that you understand it
5. Set for yourself the CRITERIA for the quality of this task
6. Work out your solution, while documenting your thought process one by one on text. You MUST combine your deep knowledge of the topic and clear thinking to accurately decipher the answer step-by-step with CONCRETE details.
7. Come up with clear response to your task.
8. After deciding on the final answer you must distinguish it from other parts of your answer. HIGHLIGHT it by adding triple @ symbols at the beginning and at the end of the text block of the final answer. AVOID bold and italic.
9. Double check that the final answer is correct, written in the LANGUAGE of the question or the task, HIGHLIGHTED correctly. Check there are no bold or italic in final answer.

###Task###

```
{task}
```

###Use this answering example###

I'll answer as the world-famous <specific field> scientists with <most prestigious LOCAL award>'
<Repeat the question in your own words>
<Suggest quality criteria>
<Work out the solution with knowledge, clear thinking and reasoning>
<Document your task solving process step by step>
<Formulate the final response>

@@@
<final answer in the language of the task in text format>
@@@
'''

    def __init__(self, llm: BaseLanguageModel, retries: bool = True,
                 max_attempts: int = DEFAULT_MAX_ATTEMPTS, verbose: int = DEFAULT_VERBOSE_LEVEL):
        super().__init__(llm, retries, max_attempts, verbose)
        self.prompt = PromptTemplate(input_variables=['task'], template=self.TEMPLATE)

    def __call__(self, task: str) -> str:
        """Генерирует ответ и извлекает содержимое между ПОСЛЕДНЕЙ парой @@@."""
        inp = self.prompt.format(task=task)
        resp = super().__call__(inp) # Используем __call__ родителя

        if self.verbose >= 2: # Вывод raw ответа только на уровне 2+
            print(f"--- Laconic Agent Raw Response ({self.model_name}) ---")
            print(repr(resp))
            print("---")

        if resp.startswith("Ошибка"):
             return resp

        start_marker = "@@@"
        end_marker = "@@@"
        last_end = resp.rfind(end_marker)
        if last_end != -1:
            last_start = resp.rfind(start_marker, 0, last_end)
            if last_start != -1 and last_start < last_end:
                extracted = resp[last_start + len(start_marker):last_end].strip()
                if extracted:
                    return extracted
                else:
                    if self.verbose >= 2: # Предупреждение только на уровне 2+
                        warnings.warn(f"Laconic Agent ({self.model_name}): Найдена пара @@@, но между ними пусто.")
                    return "" # Возвращаем пустую строку
        # Если маркеры не найдены, молча возвращаем весь ответ
        return resp.strip()

# --- Pydantic модель для Checker ---
class CheckerOutput(BaseModel): # ! Возможно излишнее, и можно обойтись встроенной работой с JSON в gemini-моделях
    """Структура для оценки Checker Agent."""
    reasoning: str = Field(..., description="Пошаговое объяснение вердикта.")
    verdict: str = Field(..., description="Финальный вердикт: 'correct' или 'incorrect'.")

# --- АГЕНТ-ПРОВЕРЩИК (CHECKER) ---
class CheckerAgent(BaseAgent):
    """Агент, оценивающий ответ другого агента через structured output."""
    PROMPT = (
        "Ты беспристрастный оценщик. Оцени 'Ответ' на основе 'Задачи'.\n\n"
        "Задача: {task}\n"
        "Ответ: {answer}\n\n"
        "Выведи оценку СТРОГО в формате JSON со структурой 'CheckerOutput' (используй tool/function).\n"
        "Сфокусируйся на том, точно ли ответ выполняет требования задачи.\n"
        "Оценка:"
    )

    def __init__(self, llm: BaseLanguageModel, retries: bool = True,
                 max_attempts: int = DEFAULT_MAX_ATTEMPTS, verbose: int = DEFAULT_VERBOSE_LEVEL):
        if not hasattr(llm, 'with_structured_output'):
             raise ValueError("CheckerAgent требует LLM с поддержкой structured output.")
        super().__init__(llm, retries, max_attempts, verbose)

    def __call__(self, task: str, answer: str) -> bool: # Убрали verbose параметр
        """Оценивает ответ, используя structured output."""
        prompt_str = self.PROMPT.format(task=task, answer=answer)
        out = self.structured(prompt_str, CheckerOutput) # structured унаследован от BaseAgent

        if isinstance(out, dict) and "error" in out:
             # Ошибки уже печатаются внутри self.structured при verbose >= 1
             return False

        if self.verbose >= 2: # Вывод JSON только на уровне 2+
            print(f"--- Checker Agent Structured Output ({self.model_name}) ---")
            try:
                output_dict = out.model_dump() if hasattr(out, 'model_dump') else out.dict()
                print(json.dumps(output_dict, indent=2))
            except Exception as json_e:
                 print(f"Не удалось сериализовать Pydantic объект: {json_e}\nОбъект: {out}")
            print("---")

        try:
             return out.verdict.strip().lower() == 'correct'
        except AttributeError:
             if self.verbose >= 1:
                  print(f"!!! Ошибка CheckerAgent: Не удалось получить 'verdict' из ответа: {out}")
             return False

# --- ФАБРИКА АГЕНТОВ ---
def create_agent(
    provider: str = 'google',
    style: str = 'default',
    model_name: str = None,
    temperature: float = 0.0,
    retries: bool = True,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    verbose: int = DEFAULT_VERBOSE_LEVEL
) -> BaseAgent:
    """Создает агента нужного стиля ('default', 'laconic', 'checker')."""
    if provider.lower() != 'google':
        raise ValueError('Поддерживается только провайдер google')

    agent_class: Type[BaseAgent]
    requires_structured = False

    if style == 'default': agent_class = BaseAgent
    elif style == 'laconic': agent_class = LaconicAgent
    elif style == 'checker':
        agent_class = CheckerAgent
        requires_structured = True
    else: raise ValueError(f'Неизвестный стиль агента: {style}')

    if model_name:
        name = model_name
        if requires_structured and verbose >= 1 and not ("1.5" in name or ("pro" in name and "1.0" in name)):
             warnings.warn(f"Модель '{name}' для CheckerAgent может не поддерживать structured output.")
    else:
        # Передаем verbose в select_model
        name = select_model(structured_output_required=requires_structured, verbose=verbose)

    if verbose >= 1: # Печатаем только если verbose >= 1
        print(f"Создание '{style}' агента с моделью: {name}")

    try:
        llm = ChatGoogleGenerativeAI(
            model=name, temperature=temperature, safety_settings=LOW_SAFETY,
            )
    except Exception as e:
         # Критическая ошибка всегда выводится
         print(f"!!! КРИТИЧЕСКАЯ ОШИБКА при инициализации ChatGoogleGenerativeAI '{name}': {e}")
         raise

    # Передаем verbose в конструктор агента
    return agent_class(llm=llm, retries=retries, max_attempts=max_attempts, verbose=verbose)

# --- ДЕМОНСТРАЦИЯ ---
if __name__ == '__main__':
    DEMO_VERBOSE_LEVEL = 1 # 0, 1, 2 или 3+

    if DEMO_VERBOSE_LEVEL >= 1: print("--- Демонстрация работы агентов ---")
    demo_max_attempts = 3

    try:
        if DEMO_VERBOSE_LEVEL >= 1: print("\n>>> Создание агентов...")
        # Передаем уровень детализации в фабрику
        raw_agent = create_agent(style='default', max_attempts=demo_max_attempts, verbose=DEMO_VERBOSE_LEVEL)
        laconic_agent = create_agent(style='laconic', max_attempts=demo_max_attempts, verbose=DEMO_VERBOSE_LEVEL)
        checker_agent = create_agent(style='checker', max_attempts=demo_max_attempts, verbose=DEMO_VERBOSE_LEVEL)

        if DEMO_VERBOSE_LEVEL >= 1: print("\n--- Базовый агент ---")
        raw_output = raw_agent('Сколько дней в високосном году?')
        print(f'Ответ: {raw_output}') # Финальный результат показываем всегда

        if DEMO_VERBOSE_LEVEL >= 1: print("\n--- Лаконичный агент ---")
        task_laconic = "Напиши химическую формулу воды."
        # Вызываем без verbose аргумента, агент использует свой self.verbose
        laconic_output_extracted = laconic_agent(task_laconic)
        # Задачу и результат показываем всегда
        print(f'Задача: "{task_laconic}"')
        print(f'Извлеченный ответ: "{laconic_output_extracted}"')

        # --- Проверки Агентом-проверщиком ---
        if DEMO_VERBOSE_LEVEL >= 1: print("\n--- Агент-проверщик (корректный ответ) ---")
        is_correct_1 = checker_agent(task=task_laconic, answer=laconic_output_extracted)
        # Финальный результат показываем всегда
        print(f'Проверка: "{laconic_output_extracted}" == "{task_laconic}"? --> {is_correct_1}')

        if DEMO_VERBOSE_LEVEL >= 1: print("\n--- Агент-проверщик (некорректный ответ) ---")
        task_checker_2 = "Напиши химическую формулу серной кислоты."
        incorrect_answer = "NaCl"
        is_correct_2 = checker_agent(task=task_checker_2, answer=incorrect_answer)
        print(f'Проверка: "{incorrect_answer}" == "{task_checker_2}"? --> {is_correct_2}')

        if DEMO_VERBOSE_LEVEL >= 1: print("\n--- Агент-проверщик (ответ не по форме) ---")
        task_checker_3 = "Напиши ТОЛЬКО химическую формулу воды."
        borderline_answer = "Формула воды: H2O"
        is_correct_3 = checker_agent(task=task_checker_3, answer=borderline_answer)
        print(f'Проверка: "{borderline_answer}" == "{task_checker_3}"? --> {is_correct_3}')

        if DEMO_VERBOSE_LEVEL >= 1: print("\n--- Дополнительный вызов ---")
        time.sleep(1)
        raw_output_2 = raw_agent('Столица Франции?')
        print(f'Ответ: {raw_output_2}') # Финальный результат показываем всегда


    except (ValueError, NotImplementedError, GoogleAPIError, Exception) as e:
        # Выводим ошибку всегда, если она дошла до этого уровня
        print(f"\n!!! Произошла ошибка во время демонстрации !!!")
        import traceback
        traceback.print_exc()

    if DEMO_VERBOSE_LEVEL >= 1: print("\n--- Демонстрация завершена ---")