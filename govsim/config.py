# config.py
from pathlib import Path

# --- Основные параметры симуляции ---
SIMULATION_CONFIG = {
    "total_steps": 200,
    "economic_model_type": "CoupledLinearStochasticSystem", #"LinearStochasticSystem",
    "government_agent_type":  "IntelligentLLMAgent", # TestPoliciesAgent, IntelligentLLMAgent, StaticPolicyAgent
    "agent_decision_frequency": 50,
    "log_level": "INFO",
    "decision_schedule_method": "INTERVAL", # Можно использовать для ясности, что частота задана
    "logs_filename": "simulation_results.json"
}

# --- Основные пути для поиска файлов ---
# менять при изменении структуры проекта
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
LOGS_DIR = PROJECT_ROOT / "logs"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
PROMPTS_DIR = PACKAGE_ROOT / "prompts"

# --- Параметры для экономических моделей ---
ECONOMIC_MODEL_PARAMS = {
    "SimpleGrowthModel": {
        "initial_gdp": 1000.0,
        "base_growth_rate": 0.005,
        "init_tax_rate": 0.2
    },
    "SingleMarketModel": {
        "num_consumers": 20,
        "num_firms": 5,
        "initial_market_price": 15.0,
        "consumer_config": {"base_a": 20.0, "b_sensitivity": 0.8, "initial_income": 100.0, "adaptation_rate": 0.01, "memory_factor": 0.3},
        "firm_config": {"fixed_costs": 10.0, "base_variable_cost_per_unit": 5.0, "initial_production_capacity": 35.0, "adaptation_rate": 0.01}
    },
    "LinearStochasticSystem": {
        "initial_x": 0.0,
        "param_A": 0.95,
        "param_B": 0.5,
        "param_C": 0.0,
        "sigma_epsilon": 0.1,
        "target_x": 0.0, # Цель стабилизации x=0
        "u_range": (-2.0, 2.0) # Ограничение на управление
    },
    "CoupledLinearStochasticSystem": {
        "initial_x": 0.0,
        "param_A": 0.96,
        "param_B": 0.4,
        "param_C": 0.0,
        "sigma_epsilon": 0.12,
        "target_x": 0.0,
        "u_range": (-2.0, 2.0),

        "a12": 0.20, "a13": -0.10,
        "a21": 0.15, "a31": -0.10,
        "gamma1": 0.92, "gamma2": 0.88,

        "d1": 0.20, "d2": -0.15,
        "sigma_aux1": 0.05, "sigma_aux2": 0.05,

        "u_smoothing_rho": 0.70,
        "param_B_drift_sigma": 0.01,
        "param_C_drift_sigma": 0.002,
        "target_drift_sigma": 0.0,

        "shock_period": 150,
        "shock_magnitude_aux1": 0.8,
        "shock_magnitude_aux2": -0.6,

        "param_B_bounds": (-1.5, 1.5),
        "param_C_bounds": (-1.0, 1.0),

        "initial_x_aux1": 0.0,
        "initial_x_aux2": 0.0
    }

}

# --- Параметры для агентов-правительств ---
GOVERNMENT_AGENT_PARAMS = {
    "StaticPolicyAgent": {},
    "RandomAgent": {
        "change_probability": 0.05
    },
    "IntelligentLLMAgent": {
        "model_name": "models/gemini-2.5-flash", #"models/gemini-2.5-flash-lite", 
        "api_call_delay": 6.1, #4.1,
        "prompt_template_path": PROMPTS_DIR / 'coupled_system_prompt.md', #"linear_system_prompt_obfuscated.md",
        "temperature": 0.5,
        "max_history_steps_for_prompt": 10,
        "performance_window": 30, # Окно для расчета KPI
        "verbose_llm": 1,
        "llm_style": "default"
    },
    "TestPoliciesAgent": {
        "test_policy_dict": {
            "policy_type_id": "set_control_input",
            "value_expression": "-0.9 * current_x", 
            "reasoning": "Тест: Пропорциональное управление."
        }
    },
}

