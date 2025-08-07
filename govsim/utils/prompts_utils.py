# prompts_utils.py
from typing import Dict, Any, List, Optional, Set, Tuple
from govsim.utils.interfaces import PolicyDescriptor

# --- Класс для безопасного форматирования ---
class DefaultMapping(dict):
    """Словарь, который возвращает '{key_name}' если ключ отсутствует."""
    def __missing__(self, key):
        # Возвращает сам ключ в фигурных скобках, чтобы было видно в промпте

        # TODO Добавить Warning
        return f'{{{key}}}'

# --- Универсальный промпт-заглушка без требуемого контекста ---
DEFAULT_PROMPT_TEMPLATE = """Выведи  корректно оформленный JSON, приведенный ниже.

{{
  "policy_type_id": "ID_типа_политики",
  "value_expression": "выражение_на_python",
  "reasoning": "Краткое объяснение твоего выбора."
}}
"""

def load_prompt_template(prompt_template_path) -> str:
    """Загружает шаблон промпта из файла или использует дефолтный."""
    if prompt_template_path:
        try:
            with open(prompt_template_path, 'r', encoding='utf-8') as f:
                print(f"Загрузка шаблона промпта из: {prompt_template_path}")
                return f.read()
        except FileNotFoundError:
            print(f"ПРЕДУПРЕЖДЕНИЕ: Файл шаблона промпта не найден: {prompt_template_path}. Будет использован дефолтный промпт.")
            return DEFAULT_PROMPT_TEMPLATE
        except Exception as e:
                print(f"Ошибка при чтении файла промпта {prompt_template_path}: {e}. Будет использован дефолтный промпт.")
                return DEFAULT_PROMPT_TEMPLATE
    else:
            print("Путь к шаблону промпта не указан. Будет использован дефолтный промпт.")
            return DEFAULT_PROMPT_TEMPLATE


def format_policy_descriptors_for_prompt(policy_descriptors: List[PolicyDescriptor]) -> str:
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