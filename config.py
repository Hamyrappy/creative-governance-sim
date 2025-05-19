# config.py

# --- Основные параметры симуляции ---
SIMULATION_CONFIG = {
    "total_steps": 100,
    "economic_model_type": "LinearStochasticSystem",
    "government_agent_type":  "IntelligentLLMAgent", # TestPoliciesAgent, IntelligentLLMAgent, StaticPolicyAgent
    "agent_decision_frequency": 50,
    "log_level": "INFO",
    "decision_schedule_method": "INTERVAL" # Можно использовать для ясности, что частота задана
}

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
    }
}

# --- Параметры для агентов-правительств ---
GOVERNMENT_AGENT_PARAMS = {
    "StaticPolicyAgent": {},
    "RandomAgent": {
        "change_probability": 0.05
    },
    "IntelligentLLMAgent": {
        "model_name": "models/gemini-2.5-flash-preview-04-17-thinking", #"gemini-1.5-flash-latest", 
        "api_call_delay": 10.0, #4.1,
        "prompt_template_path": "prompts/linear_system_prompt.md",
        "temperature": 0.5,
        "max_history_steps_for_prompt": 10,
        "performance_window": 10, # Окно для расчета KPI
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