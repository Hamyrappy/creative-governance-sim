# simulation.py
import time
import json
from typing import Dict, Type
from pathlib import Path
import traceback # Для вывода стека ошибок при сохранении

# Импортируем конфиг и интерфейсы
import config
from interfaces import BaseEconomicSystem, BaseGovernmentAgent, Policy

# Импортируем конкретные реализации динамически на основе конфига
from economic_models.economic_models import SimpleGrowthModel, LinearStochasticSystem
from economic_models.single_market_model import SingleMarketModel
from governing_agents.government_agents import RandomAgent, StaticPolicyAgent, TestPoliciesAgent, IntelligentLLMAgent

# --- Карты для динамического выбора классов ---
ECONOMIC_MODEL_MAP: Dict[str, Type[BaseEconomicSystem]] = {
    "SimpleGrowthModel": SimpleGrowthModel,
    "SingleMarketModel": SingleMarketModel,
    "LinearStochasticSystem": LinearStochasticSystem
    
}

GOVERNMENT_AGENT_MAP: Dict[str, Type[BaseGovernmentAgent]] = {
    "RandomAgent": RandomAgent,
    "StaticPolicyAgent": StaticPolicyAgent,
    "TestPoliciesAgent": TestPoliciesAgent,
    "IntelligentLLMAgent": IntelligentLLMAgent,
}
# -----------------------------------------------------

class Simulation:
    """
    Основной класс, управляющий процессом симуляции.
    Отвечает за инициализацию, запуск цикла шагов, взаимодействие
    между моделью экономики и агентом-правительством, а также за логирование
    и сохранение результатов.
    """
    def __init__(self, sim_config: Dict, model_params: Dict, agent_params: Dict):
        """
        Инициализация симуляции.
        Args:
            sim_config: Словарь с общей конфигурацией симуляции.
            model_params: Словарь с параметрами для выбранной экономической модели.
            agent_params: Словарь с параметрами для выбранного агента-правительства.
        """
        self.config = sim_config
        self.total_steps = sim_config.get("total_steps", 100)

        # --- Динамическая инициализация модели и агента ---
        model_type = sim_config.get("economic_model_type")
        agent_type = sim_config.get("government_agent_type")

        # Проверка наличия классов в картах
        if model_type not in ECONOMIC_MODEL_MAP:
            raise ValueError(f"Неизвестный economic_model_type в конфиге: {model_type}")
        if agent_type not in GOVERNMENT_AGENT_MAP:
            raise ValueError(f"Неизвестный government_agent_type в конфиге: {agent_type}")

        # Получаем классы из карт
        EconomicModelClass = ECONOMIC_MODEL_MAP[model_type]
        GovernmentAgentClass = GOVERNMENT_AGENT_MAP[agent_type]

        # Создаем экземпляры, передавая им соответствующие параметры
        print(f"Инициализация модели: {model_type} с параметрами {model_params.get(model_type, {})}")
        self.economic_system = EconomicModelClass(model_params.get(model_type, {}))

        print(f"Инициализация агента: {agent_type} с параметрами {agent_params.get(agent_type, {})}")
        self.government_agent = GovernmentAgentClass(agent_params.get(agent_type, {}))
        # ---------------------------------------------------------

        # Хранилище для логов симуляции (агрегированные данные по шагам)
        self.simulation_log: List[Dict[str, Any]] = []
        # Архив (потенциально для RAG) - сейчас это просто история состояний модели
        # self.archive = [] # Можно использовать self.economic_system.history

    def run(self):
        """Запускает основной цикл симуляции."""
        print("-" * 50)
        print(f"Запуск симуляции: {self.config['economic_model_type']} с {self.config['government_agent_type']}")
        print(f"Всего шагов: {self.total_steps}")
        print("-" * 50)

        start_time = time.time()

        for step_num in range(self.total_steps):
            current_step_display = step_num + 1
            print(f"\n--- Шаг {current_step_display}/{self.total_steps} ---")

            try:
                # 1. Получаем состояние, видимое агенту, из экономической системы
                current_state_for_agent = self.economic_system.get_state_for_agent()

                # 2. Агент принимает решение об изменении политики
                # Передаем текущее состояние и полную историю для анализа агентом
                system_history = self.economic_system.history
                policy_change: Optional[Policy] = self.government_agent.decide_policy(
                    current_state_for_agent,
                    system_history,
                    self.economic_system # Передаем ссылку на модель экономики
                )
                
                # 3. Применяем предложенное изменение политики к экономической системе
                # (Метод внутри модели обновит self.economic_system.active_policies)
                self.economic_system.apply_policy_change(policy_change)

                # 4. Выполняем шаг экономической модели
                # (Внутри этого метода произойдет вычисление эффектов активных политик
                # и обновление состояния экономики)
                self.economic_system.step()

                # 5. Логируем результаты этого шага
                # Получаем актуальные метрики *после* выполнения шага
                current_metrics = self.economic_system.get_current_metrics()
                # Получаем активные политики в виде словарей для лога
                active_policies_log = [p.to_dict() for p in self.economic_system.active_policies]

                step_log = {
                    "step": self.economic_system.current_step,
                    "metrics": current_metrics,
                    # Логируем решение агента (если оно было) как словарь
                    "policy_decision": policy_change.to_dict() if policy_change else None,
                    # Логируем список активных политик (как словари) на конец шага
                    "active_policies": active_policies_log,
                }
                self.simulation_log.append(step_log)

                # Вывод текущих метрик в консоль
                print(f"  Метрики в конце шага: {current_metrics}")

            except Exception as e:
                print(f"\n!!! ОШИБКА НА ШАГЕ {current_step_display} !!!")
                print(f"Тип ошибки: {type(e).__name__}")
                print(f"Сообщение: {e}")
                print("Трассировка стека:")
                traceback.print_exc() # Печатаем полный стек вызовов
                print("!!! Симуляция прервана !!!")
                break # Прерываем цикл симуляции при ошибке

            # Небольшая пауза для читаемости вывода (опционально)
            # time.sleep(0.05)

        # --- Завершение симуляции ---
        end_time = time.time()
        total_time = end_time - start_time
        print("-" * 50)
        print(f"--- Симуляция Завершена за {total_time:.2f} сек ---")
        print("-" * 50)

        # Сохраняем результаты в файл
        self.save_results()


    def save_results(self, filename_base="simulation_results.json"):
         """
         Сохраняет результаты симуляции (конфиг, лог шагов, историю модели)
         в JSON файл рядом со скриптом simulation.py.
         """
         # Определяем путь к файлу результатов относительно текущего скрипта
         script_dir = Path(__file__).parent
         results_path = script_dir / filename_base

         # Формируем данные для сохранения
         # ВАЖНО: Убедимся, что все данные здесь сериализуемы в JSON
         # self.economic_system.history уже содержит состояния, где политики
         # преобразованы в словари методом _update_state -> to_dict()
         results_data = {
            "config": self.config,
            "simulation_log": self.simulation_log, # Лог шагов (уже с to_dict())
            "full_system_history": self.economic_system.history # Полная история состояний
         }

         try:
            # Открываем файл для записи с кодировкой utf-8
            with open(results_path, 'w', encoding='utf-8') as f:
                # Записываем данные в JSON с отступами и поддержкой не-ASCII символов
                json.dump(results_data, f, indent=4, ensure_ascii=False)
            print(f"Результаты симуляции успешно сохранены в: {results_path}")
         except TypeError as e:
             print(f"!!! Ошибка ТИПА при сохранении результатов в JSON: {e}")
             print("Вероятно, вы пытаетесь сохранить несериализуемый объект.")
             print("Проверьте содержимое `results_data`, особенно словари политик.")
             print("Полные данные для отладки:")
             # Попытка вывести проблемные данные (может быть очень много)
             # print(results_data)
         except Exception as e:
            # Ловим другие возможные ошибки при записи файла
            print(f"!!! Ошибка при сохранении результатов в {results_path}: {e}")
            traceback.print_exc()


# --- Точка входа при запуске скрипта ---
if __name__ == "__main__":
    # Загружаем конфигурацию из модуля config
    sim_cfg = config.SIMULATION_CONFIG
    model_prms = config.ECONOMIC_MODEL_PARAMS
    agent_prms = config.GOVERNMENT_AGENT_PARAMS

    # Создаем и запускаем объект симуляции
    simulation = Simulation(sim_cfg, model_prms, agent_prms)
    simulation.run()

