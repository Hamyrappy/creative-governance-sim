import json
import re
import os

def extract_unique_policies_rationales_and_formulas_from_file(file_path: str) -> dict:
    """
    Извлекает все уникальные политики, их обоснования и формулы из JSON-файла.

    Args:
        file_path: Путь к файлу, содержащему данные в формате JSON.

    Returns:
        Словарь, где ключи - это ID уникальных политик,
        а значения - это словари с ключами "rationale" и "formula".
        Возвращает пустой словарь в случае ошибки.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Ошибка: Файл не найден по пути: {file_path}")
        return {}
    except json.JSONDecodeError as e:
        print(f"Ошибка декодирования JSON из файла {file_path}: {e}")
        return {}
    except Exception as e:
        print(f"Произошла непредвиденная ошибка при чтении файла {file_path}: {e}")
        return {}

    unique_policies_data = {}

    if 'simulation_log' not in data or not isinstance(data['simulation_log'], list):
        print("Предупреждение: Ключ 'simulation_log' не найден или не является списком в JSON.")
        return {}

    for log_entry in data['simulation_log']:
        if 'active_policies_log' in log_entry and log_entry['active_policies_log']:
            for policy_item in log_entry['active_policies_log']:
                policy_id = policy_item.get('id')
                description = policy_item.get('description')
                value_expression = policy_item.get('value_expression')

                # Политика должна иметь ID и формулу для добавления.
                # Описание (и, следовательно, обоснование) опционально.
                if not policy_id or value_expression is None: # value_expression может быть пустой строкой, но не None
                    continue

                if policy_id not in unique_policies_data:
                    rationale = None # По умолчанию нет обоснования
                    if description: # Извлекаем обоснование только если есть описание
                        match = re.match(r"LLM \((.*)\): (.*)", description, re.DOTALL)
                        if match:
                            rationale = match.group(1).strip()
                    
                    unique_policies_data[policy_id] = {
                        "rationale": rationale,
                        "formula": value_expression
                    }
    return unique_policies_data

# --- Пример использования ---

# 1. Создадим временный JSON-файл для демонстрации, включая поле value_expression
example_json_content_with_formulas = """
{
    "config": {
        "total_steps": 1000
    },
    "simulation_log": [
        {
            "step": 1,
            "active_policies_log": []
        },
        {
            "step": 200,
            "active_policies_log": [
                {
                    "id": "llm_policy_1_step200.0",
                    "description": "LLM (1. Обоснование для политики 1. Это важная причина. 2. Еще одна причина.): Действие политики 1.",
                    "policy_type": "set_control_input",
                    "value_expression": "-1.0 * (current_x - target_x)"
                }
            ]
        },
        {
            "step": 201,
            "active_policies_log": [
                {
                    "id": "llm_policy_1_step200.0",
                    "description": "LLM (1. Обоснование для политики 1. Это важная причина. 2. Еще одна причина.): Действие политики 1.",
                    "policy_type": "set_control_input",
                    "value_expression": "-1.0 * (current_x - target_x)" 
                }
            ]
        },
        {
            "step": 400,
            "active_policies_log": [
                {
                    "id": "llm_policy_2_step400.0",
                    "description": "LLM (Это другое обоснование для другой политики с ID llm_policy_2_step400.0.): Другое действие политики.",
                    "policy_type": "set_control_input",
                    "value_expression": "-0.5 * current_x + 0.1 * integral_error"
                }
            ]
        },
        {
            "step": 500,
            "active_policies_log": [
                {
                    "id": "llm_policy_3_step500.0",
                    "description": "No Rationale Format: Это описание не соответствует ожидаемому формату.",
                    "policy_type": "set_control_input",
                    "value_expression": "constant_value * 0.9"
                }
            ]
        },
        {
            "step": 600,
            "active_policies_log": [
                {
                    "id": "llm_policy_4_step600.0",
                    "description": "LLM (Обоснование для политики 4, но формула null.): Действие политики 4.",
                    "policy_type": "set_control_input",
                    "value_expression": null 
                }
            ]
        },
        {
            "step": 700,
            "active_policies_log": [
                {
                    "id": "llm_policy_5_step700.0",
                    "description": "LLM (Обоснование для политики 5, но нет value_expression ключа.)",
                    "policy_type": "set_control_input"
                }
            ]
        },
        {
            "step": 800,
            "active_policies_log": [
                {
                    "id": "llm_policy_6_step800.0",
                    "description": null, 
                    "policy_type": "set_control_input",
                    "value_expression": "0.111"
                }
            ]
        }
    ]
}
"""
temp_json_filename_formulas = "temp_data_formulas.json"
with open(temp_json_filename_formulas, 'w', encoding='utf-8') as f_temp:
    f_temp.write(example_json_content_with_formulas)

# 2. Укажите имя вашего JSON-файла
json_file_name_formulas = 'logs/simulation_results_exp4.json'# temp_json_filename_formulas # Используем созданный временный файл

# 3. Вызов функции с именем файла
extracted_full_data = extract_unique_policies_rationales_and_formulas_from_file(json_file_name_formulas)

# 4. Вывод результатов
if extracted_full_data:
    print(f"\nИзвлеченные уникальные политики, их обоснования и формулы из файла '{json_file_name_formulas}':")
    for policy_id, data_item in extracted_full_data.items():
        print(f"\nPolicy ID: {policy_id}")
        rationale_text = data_item['rationale'] if data_item['rationale'] is not None else "N/A (отсутствует или не удалось извлечь)"
        formula_text = data_item['formula'] if data_item['formula'] is not None else "N/A (отсутствует)"
        print(f"  Rationale: {rationale_text}")
        print(f"  Formula:   {formula_text}")
else:
    print(f"\nНе найдено политик с данными в файле '{json_file_name_formulas}' или произошла ошибка при обработке.")

# 5. (Опционально) Удаляем временный файл
if os.path.exists(temp_json_filename_formulas):
     os.remove(temp_json_filename_formulas)
     print(f"\nВременный файл '{temp_json_filename_formulas}' удален.")