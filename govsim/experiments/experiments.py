# experiments.py
import time
import json
from pathlib import Path
from typing import Type, Optional, Dict, Any, List
import traceback
import copy # Для глубокого копирования конфигов
import numpy as np # Для сериализатора

from interfaces import BaseEconomicSystem, BaseGovernmentAgent, Policy
from economic_models.economic_models import SimpleGrowthModel, LinearStochasticSystem
from governing_agents.government_agents import StaticPolicyAgent, RandomAgent, TestPoliciesAgent
from governing_agents.gov_agent_linear import IntelligentLLMAgent

# Импорт базового конфига
import config

# --- Фабрики ---
def create_economic_model(model_type: str, params: Dict[str, Any]) -> BaseEconomicSystem:
    """Создает экземпляр экономической модели по типу и параметрам."""
    model_classes: Dict[str, Type[BaseEconomicSystem]] = {
        "LinearStochasticSystem": LinearStochasticSystem,
    }
    model_class = model_classes.get(model_type)
    if model_class:
        return model_class(params)
    else:
        raise ValueError(f"Неизвестный или неподдерживаемый тип экономической модели для экспериментов: {model_type}")

def create_government_agent(agent_type: str, params: Dict[str, Any]) -> BaseGovernmentAgent:
    """Создает экземпляр агента-правительства по типу и параметрам."""
    agent_classes: Dict[str, Type[BaseGovernmentAgent]] = {
        "StaticPolicyAgent": StaticPolicyAgent,
        "RandomAgent": RandomAgent,
        "IntelligentLLMAgent": IntelligentLLMAgent,
        "TestPoliciesAgent": TestPoliciesAgent,
    }
    agent_class = agent_classes.get(agent_type)
    if agent_class:
        return agent_class(params)
    else:
        raise ValueError(f"Неизвестный тип агента-правительства: {agent_type}")

# --- Функция для одного эксперимента ---
def run_simulation_experiment(exp_id: int, custom_config_updates: Dict[str, Any]):
    """Запускает один эксперимент с заданными модификациями конфига."""
    print(f"--- Запуск эксперимента {exp_id} ---")

    # 1. Загрузка и обновление конфигурации
    # Глубокое копирование базовых конфигов
    base_sim_config = copy.deepcopy(config.SIMULATION_CONFIG)
    base_model_params_all = copy.deepcopy(config.ECONOMIC_MODEL_PARAMS)
    base_agent_params_all = copy.deepcopy(config.GOVERNMENT_AGENT_PARAMS)

    # Применяем обновления к копии sim_config
    current_sim_config = base_sim_config
    current_sim_config.update(custom_config_updates.get("SIMULATION_CONFIG", {}))

    # Определяем типы модели и агента ПОСЛЕ обновления sim_config
    model_type = current_sim_config.get("economic_model_type")
    agent_type = current_sim_config.get("government_agent_type")

    if not model_type or not agent_type:
        print(f"Ошибка: Не удалось определить model_type или agent_type для эксперимента {exp_id}")
        return

    # Получаем и обновляем параметры для КОНКРЕТНОЙ модели
    current_model_params = base_model_params_all.get(model_type, {})
    current_model_params.update(custom_config_updates.get("ECONOMIC_MODEL_PARAMS", {}).get(model_type, {}))

    # Получаем и обновляем параметры для КОНКРЕТНОГО агента
    current_agent_params = base_agent_params_all.get(agent_type, {})
    current_agent_params.update(custom_config_updates.get("GOVERNMENT_AGENT_PARAMS", {}).get(agent_type, {}))

    # Извлечение параметров симуляции
    total_steps = current_sim_config.get("total_steps", 1000)
    agent_decision_frequency = current_sim_config.get("agent_decision_frequency", 100)
    shock_step = custom_config_updates.get("shock_step", total_steps // 2) # По умолчанию середина
    parameter_to_change = custom_config_updates.get("parameter_to_change", None)
    new_parameter_value = custom_config_updates.get("new_parameter_value", None)

    print(f"Эксперимент {exp_id}: Модель={model_type}, Агент={agent_type}, Шаги={total_steps}, Частота решений={agent_decision_frequency}")
    if parameter_to_change:
        print(f"Шок на шаге {shock_step}: Параметр '{parameter_to_change}' -> {new_parameter_value}")

    # 2. Создание экземпляров модели и агента
    try:
        # Передаем обновленные параметры конкретной модели/агента
        economic_system = create_economic_model(model_type, current_model_params)
        government_agent = create_government_agent(agent_type, current_agent_params)
    except ValueError as e:
        print(f"Ошибка инициализации эксперимента {exp_id}: {e}")
        return
    except Exception as e:
        print(f"Неожиданная ошибка при инициализации эксперимента {exp_id}: {e}")
        print(traceback.format_exc())
        return

    # 3. Основной цикл симуляции
    simulation_log = []
    start_time = time.time()
    last_policy_decision_info = None

    for current_step in range(total_steps):
        step_start_time = time.time()

        # Применение шока параметров
        if current_step + 1 == shock_step and parameter_to_change is not None:
            try:
                if isinstance(economic_system, LinearStochasticSystem):
                    if hasattr(economic_system, parameter_to_change):
                        old_value = getattr(economic_system, parameter_to_change)
                        setattr(economic_system, parameter_to_change, new_parameter_value)
                        print(f"\n!!! Эксперимент {exp_id}, Шаг {current_step + 1}: Шок применен. '{parameter_to_change}' изменен с {old_value} на {new_parameter_value} !!!\n")
                        # Обновляем состояние модели, чтобы агент увидел измененный параметр (например, target_x)
                        economic_system.state = economic_system._update_state() # Вызываем внутренний метод обновления состояния
                    else:
                        print(f"Предупреждение: Попытка изменить несуществующий параметр '{parameter_to_change}' в модели.")
                else:
                     print(f"Предупреждение: Шок параметров '{parameter_to_change}' реализован только для LinearStochasticSystem.")
            except Exception as e:
                print(f"Ошибка при применении шока параметров на шаге {current_step + 1} эксперимента {exp_id}: {e}")
                print(traceback.format_exc())

        policy_decision_this_step: Optional[Policy] = None

        # Решение агента (с заданной частотой)
        # Агент принимает решение *перед* шагом N, если N делится на частоту
        if (current_step) % agent_decision_frequency == 0 and current_step < total_steps and current_step > 0:
             # Решение по политике принимается на шагах 100, 200...
             # И применяется ПЕРЕД этими шагами.
            print(f"Эксперимент {exp_id}: Агент {agent_type} принимает решение на шаге {current_step}...")
            try:
                current_state_for_agent = economic_system.get_state_for_agent()
                history_for_agent = list(economic_system.history)

                policy_decision_this_step = government_agent.decide_policy(
                    current_state_for_agent,
                    history_for_agent,
                    economic_system
                )
                if policy_decision_this_step:
                    print(f"  -> Агент предложил политику: ID={policy_decision_this_step.id}, Тип={policy_decision_this_step.policy_type}")
                    last_policy_decision_info = {
                        'policy_type': policy_decision_this_step.policy_type,
                        'value_expression': policy_decision_this_step.value_expression,
                        'id': policy_decision_this_step.id,
                        'description': policy_decision_this_step.description
                    }
                else:
                     print("  -> Агент решил не менять политику.")
                     last_policy_decision_info = None

            except NotImplementedError:
                 print(f"Предупреждение: Метод decide_policy не реализован для агента {agent_type}.")
                 last_policy_decision_info = None
            except Exception as e:
                 print(f"Ошибка в decide_policy агента {agent_type} на шаге {current_step} эксперимента {exp_id}: {e}")
                 print(traceback.format_exc())
                 last_policy_decision_info = None

            # Применяем политику сразу после решения агента, *перед* шагом модели
            try:
                economic_system.apply_policy_change(policy_decision_this_step)
            except Exception as e:
                print(f"Ошибка в apply_policy_change модели {model_type} на шаге {current_step} эксперимента {exp_id}: {e}")
                print(traceback.format_exc())
        else:
             # Если агент не принимал решение перед этим шагом
             last_policy_decision_info = None # Сбрасываем инфо о решении

        # Шаг экономической модели
        try:
            # Выполняем шаг N
            economic_system.step() # Теперь current_step внутри модели станет N+1
        except Exception as e:
            print(f"КРИТИЧЕСКАЯ ОШИБКА на шаге {current_step + 1} в economic_system.step() эксперимента {exp_id}: {e}")
            print(traceback.format_exc())
            break

        # Логирование состояния *после* шага N
        current_log_entry = economic_system.state.copy() # Состояние после выполнения шага N (т.е. на момент N+1)
        # В лог записываем решение, которое было принято *перед* этим шагом N
        current_log_entry['policy_decision'] = last_policy_decision_info # Переименовано для совместимости
        simulation_log.append(current_log_entry)

        step_end_time = time.time()

    # 4. Завершение и сохранение результатов
    end_time = time.time()
    print(f"--- Эксперимент {exp_id} завершен за {end_time - start_time:.2f} секунд ---")

    # --- Формирование словаря результатов в требуемом формате ---
    results_data = {
        "config": current_sim_config, # Общий конфиг симуляции
        "model_params": current_model_params, # Параметры использованной модели
        "agent_params": current_agent_params, # Параметры использованного агента
        "simulation_log": simulation_log # Лог симуляции
    }
    # results_data["shock_info"] = { ... }

    results_filename = f"simulation_results_exp{exp_id}.json"
    script_dir = Path(__file__).resolve().parent
    results_filepath = script_dir / "logs" / results_filename
    try:
        def default_serializer(o):
            if isinstance(o, (np.int_, np.intc, np.intp, np.int8,
                              np.int16, np.int32, np.int64, np.uint8,
                              np.uint16, np.uint32, np.uint64)):
                return int(o)
            elif isinstance(o, (np.float_, np.float16, np.float32,
                                np.float64)):
                if np.isnan(o): return None
                return float(o)
            elif isinstance(o, (np.ndarray,)):
                return np.where(np.isnan(o), None, o).tolist()
            elif isinstance(o, (np.bool_)):
                return bool(o)
            elif isinstance(o, (np.void)):
                return None
            elif isinstance(o, Policy):
                 return o.to_dict()
            print(f"Предупреждение: Не удалось сериализовать объект типа {type(o)}. Заменен на строку.")
            return str(o)

        with open(results_filepath, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=4, default=default_serializer)
        print(f"Результаты эксперимента {exp_id} сохранены в: {results_filepath}")
    except Exception as e:
        print(f"Ошибка при сохранении результатов эксперимента {exp_id}: {e}")
        print(traceback.format_exc())


# --- Запуск экспериментов ---
if __name__ == "__main__":

    shock_apply_step = 500 # Шок применяется ПЕРЕД НАЧАЛОМ шага 500

    # --- Эксперимент 3: Шок параметра B ---
    exp3_updates = {
        "SIMULATION_CONFIG": {
            "total_steps": 1000,
            "economic_model_type": "LinearStochasticSystem",
            "government_agent_type": "IntelligentLLMAgent",
            "agent_decision_frequency": 200,
        },
        # Параметры модели и агента берутся из config.py по умолчанию
        # Мы только указываем информацию для шока
        "shock_step": shock_apply_step,
        "parameter_to_change": "param_B",
        "new_parameter_value": 0.2
    }
    # run_simulation_experiment(exp_id=3, custom_config_updates=exp3_updates)

    # --- Эксперимент 4: Изменение цели target_x ---
    exp4_updates = {
        "SIMULATION_CONFIG": {
            "total_steps": 1000,
            "economic_model_type": "LinearStochasticSystem",
            "government_agent_type": "IntelligentLLMAgent",
            "agent_decision_frequency": 200,
        },
        "shock_step": shock_apply_step,
        "parameter_to_change": "param_target_x",
        "new_parameter_value": 1.0
    }
    run_simulation_experiment(exp_id=4, custom_config_updates=exp4_updates)

    print("\n--- Все эксперименты завершены ---")