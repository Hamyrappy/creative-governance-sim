# simulation.py
import time
import json
from pathlib import Path
from typing import Type, Optional, Dict, Any
import traceback

# Импортируем базовые классы и фабрики
from interfaces import BaseEconomicSystem, BaseGovernmentAgent
from economic_models.economic_models import SimpleGrowthModel, LinearStochasticSystem
from economic_models.single_market_model import SingleMarketModel
from governing_agents.government_agents import StaticPolicyAgent, RandomAgent, TestPoliciesAgent
from governing_agents.gov_agent_linear import IntelligentLLMAgent

# Импортируем конфиг
import config

# --- Фабрики (остаются без изменений) ---
def create_economic_model(model_type: str, params: Dict[str, Any]) -> BaseEconomicSystem:
    """Создает экземпляр экономической модели по типу и параметрам."""
    model_classes: Dict[str, Type[BaseEconomicSystem]] = {
        "SimpleGrowthModel": SimpleGrowthModel,
        "SingleMarketModel": SingleMarketModel,
        "LinearStochasticSystem": LinearStochasticSystem,
    }
    model_class = model_classes.get(model_type)
    if model_class:
        return model_class(params)
    else:
        raise ValueError(f"Неизвестный тип экономической модели: {model_type}")

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

# --- Основная функция симуляции ---
def run_simulation():
    """Запускает симуляцию на основе настроек в config.py."""
    print("--- Запуск симуляции ---")

    # Загрузка конфигурации
    sim_config = config.SIMULATION_CONFIG
    total_steps = sim_config.get("total_steps", 100)
    model_type = sim_config.get("economic_model_type")
    agent_type = sim_config.get("government_agent_type")

    agent_decision_frequency = sim_config.get("agent_decision_frequency", 1) # По умолчанию 1
    if not isinstance(agent_decision_frequency, int) or agent_decision_frequency <= 0:
        print(f"Предупреждение: Некорректное значение agent_decision_frequency ({agent_decision_frequency}). Установлено значение 1.")
        agent_decision_frequency = 1

    print(f"Модель: {model_type}, Агент: {agent_type}, Шаги: {total_steps}, Частота решений агента: {agent_decision_frequency}")

    # Получаем параметры для конкретных моделей и агентов
    model_params = config.ECONOMIC_MODEL_PARAMS.get(model_type, {})
    agent_params = config.GOVERNMENT_AGENT_PARAMS.get(agent_type, {})

    # Создание экземпляров модели и агента
    try:
        economic_system = create_economic_model(model_type, model_params)
        government_agent = create_government_agent(agent_type, agent_params)
    except ValueError as e:
        print(f"Ошибка инициализации: {e}")
        return
    except Exception as e:
        print(f"Неожиданная ошибка при инициализации: {e}")
        print(traceback.format_exc()) # Печатаем traceback
        return

    # Лог симуляции
    simulation_log = []

    # Основной цикл симуляции
    start_time = time.time()
    last_policy_decision_info = {} # Для логирования последнего решения

    for current_step in range(total_steps):
        step_start_time = time.time()
        print(f"\n--- Шаг {current_step + 1}/{total_steps} ---")

        policy_decision_this_step = None # Сбрасываем решение на этом шаге

        # 1. Агент принимает решение (только с заданной частотой)
        # --- Проверяем частоту вызова ---
        if (current_step + 1) % agent_decision_frequency == 0 and current_step+1 >= 1 and current_step+1 < total_steps:
            print(f"Агент {agent_type} принимает решение...")
            try:
                current_state_for_agent = economic_system.get_state_for_agent()
                # Передаем КОПИЮ истории, чтобы агент не мог ее случайно изменить
                history_for_agent = list(economic_system.history)

                policy_decision_this_step = government_agent.decide_policy(
                    current_state_for_agent,
                    history_for_agent,
                    economic_system
                )
                if policy_decision_this_step:
                     print(f"Агент предложил политику: ID={policy_decision_this_step.id}, Тип={policy_decision_this_step.policy_type}")
                     # Запоминаем инфо о решении для лога
                     last_policy_decision_info = {
                         'policy_type': policy_decision_this_step.policy_type,
                         'value_expression': policy_decision_this_step.value_expression,
                         'id': policy_decision_this_step.id
                     }
                else:
                     print("Агент решил не менять политику.")
                     last_policy_decision_info = None # Сбрасываем, если решения не было

            except NotImplementedError:
                 print(f"Предупреждение: Метод decide_policy не реализован для агента {agent_type}.")
            except Exception as e:
                 print(f"Ошибка при вызове decide_policy агента {agent_type} на шаге {current_step + 1}: {e}")
                 print(traceback.format_exc()) # Печатаем traceback
                 # Не меняем политику в случае ошибки агента

            # 2. Модель применяет предложенное изменение политики
            # Важно: применяем РЕЗУЛЬТАТ вызова decide_policy на этом шаге
            try:
                economic_system.apply_policy_change(policy_decision_this_step)
            except Exception as e:
                print(f"Ошибка при применении политики в модели {model_type} на шаге {current_step + 1}: {e}")
                print(traceback.format_exc())
        else:
             #print("Агент пропускает принятие решения на этом шаге (согласно частоте).")
             last_policy_decision_info = None # Сбрасываем, если решения не было

        # 3. Экономическая модель выполняет шаг
        try:
            economic_system.step()
        except Exception as e:
            print(f"КРИТИЧЕСКАЯ ОШИБКА на шаге {current_step + 1} в economic_system.step(): {e}")
            print(traceback.format_exc())
            break # Прерываем цикл

        # 4. Логирование текущего состояния
        current_log_entry = economic_system.state.copy()
        # --- Добавляем информацию о решении агента в лог ЭТОГО шага ---
        # Сохраняем last_policy_decision_info (которое было установлено, если агент принимал решение)
        current_log_entry['policy_decision'] = last_policy_decision_info
        # -------------------------------------------------------------
        simulation_log.append(current_log_entry)

        step_end_time = time.time()
        print(f"Шаг {current_step + 1} выполнен за {step_end_time - step_start_time:.3f} сек.")


    end_time = time.time()
    print(f"\n--- Симуляция завершена за {end_time - start_time:.2f} секунд ---")

    # Сохранение результатов
    results_data = {
        "config": sim_config,
        "model_params": model_params,
        "agent_params": agent_params,
        "simulation_log": simulation_log
    }

    results_filename = "simulation_results.json"
    script_dir = Path(__file__).resolve().parent
    results_filepath = script_dir / "logs" / results_filename
    try:
        # Используем обработчик для несериализуемых объектов, если вдруг они появятся
        def default_serializer(o):
            if isinstance(o, (np.int_, np.intc, np.intp, np.int8,
                              np.int16, np.int32, np.int64, np.uint8,
                              np.uint16, np.uint32, np.uint64)):
                return int(o)
            elif isinstance(o, (np.float_, np.float16, np.float32,
                                np.float64)):
                return float(o)
            elif isinstance(o, (np.ndarray,)): # Конвертируем массивы в списки
                return o.tolist()
            elif isinstance(o, (np.bool_)):
                return bool(o)
            elif isinstance(o, (np.void)): # Обработка void (если встретится)
                return None
            # TODO Можно добавить обработку datetime и т.д.
            print(f"Предупреждение: Не удалось сериализовать объект типа {type(o)}. Заменен на None.")
            return None # Заменяем несериализуемое на None

        with open(results_filepath, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, ensure_ascii=False, indent=4, default=default_serializer)
        print(f"Результаты сохранены в: {results_filepath}")
    except Exception as e:
        print(f"Ошибка при сохранении результатов: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    run_simulation()