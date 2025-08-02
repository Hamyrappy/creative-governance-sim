# single_market_model.py
import random
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from interfaces import BaseEconomicSystem, Policy, AgentId, PolicyDescriptor
from policy_utils import evaluate_safe_policy_code


class SMM_Consumer:
    """Простой агент-потребитель для модели одного рынка."""
    def __init__(self, id: int, base_a: float, b_sensitivity: float, initial_income: float, adaptation_rate: float = 0.05, memory_factor: float = 0.5):
        self.id = id
        self.base_a = base_a
        self.current_a = base_a
        self.b = b_sensitivity
        self.income = initial_income
        self.adaptation_rate = adaptation_rate
        self.memory_factor = memory_factor
        self.utility_last_step = 0.0
        self.quantity_demanded_at_price = 0.0
        self.quantity_bought_last_step = 0.0

    def decide_quantity_demanded(self, price_for_consumer: float) -> float:
        potential_demand = max(0.0, self.current_a - self.b * price_for_consumer)
        if price_for_consumer > 1e-6:
            max_affordable_quantity = self.income / price_for_consumer
        else:
            max_affordable_quantity = float('inf')
        self.quantity_demanded_at_price = min(potential_demand, max_affordable_quantity)
        return self.quantity_demanded_at_price

    def adapt(self, dissatisfaction_signal: float):
        capped_diss_signal = np.clip(dissatisfaction_signal, -1.0, 1.0)

        target_a_change_value = 0.0
        if capped_diss_signal > 0.1:
            target_a_change_value = -self.adaptation_rate * self.base_a * capped_diss_signal
        elif capped_diss_signal < -0.1:
            target_a_change_value = self.adaptation_rate * self.base_a * 0.1 * abs(capped_diss_signal)

        target_a_for_smoothing = self.current_a + target_a_change_value
        self.current_a = (1 - self.memory_factor) * self.current_a + self.memory_factor * target_a_for_smoothing
        self.current_a = max(self.base_a * 0.6, min(self.base_a * 1.5, self.current_a))


class SMM_Firm:
    """Простая агент-фирма для модели одного рынка."""
    def __init__(self, id: int, fixed_costs: float, base_variable_cost_per_unit: float, initial_production_capacity: float, adaptation_rate: float = 0.05, tech_level: float = 1.0):
        self.id = id
        self.fixed_costs = fixed_costs
        self.base_variable_cost_per_unit = base_variable_cost_per_unit
        self.current_variable_cost_per_unit = base_variable_cost_per_unit / tech_level
        self.production_capacity = initial_production_capacity
        self.planned_production_current_step = initial_production_capacity
        self.quantity_sold_last_step = 0.0
        self.profit_last_step = 0.0
        self.adaptation_rate = adaptation_rate
        self.tech_level = tech_level
        self.initial_capacity = initial_production_capacity

    def decide_supply_quantity(self, price_firm_receives: float) -> float:
        # Простая линейная функция предложения, ограниченная мощностью
        # Фирма начинает предлагать что-то, если цена выше некоторой доли VC
        vc_threshold_factor = 0.9 # Начинает предлагать, если цена покрывает 90% VC
        min_meaningful_price = self.current_variable_cost_per_unit * vc_threshold_factor
        
        if price_firm_receives > min_meaningful_price:
            # Чувствительность предложения к цене сверх порога
            # ? Можно сделать этот параметр гетерогенным или адаптивным
            supply_sensitivity = self.production_capacity * 0.5 / (self.current_variable_cost_per_unit + 1e-6) 
            
            potential_supply = supply_sensitivity * (price_firm_receives - min_meaningful_price)
            self.planned_production_current_step = max(0, min(self.production_capacity, potential_supply))
        else:
            self.planned_production_current_step = 0.0
        return self.planned_production_current_step

    def calculate_profit_for_step(self, price_received: float, quantity_actually_sold: float):
        revenue = quantity_actually_sold * price_received
        total_variable_costs_on_sold = quantity_actually_sold * self.current_variable_cost_per_unit
        self.profit_last_step = revenue - self.fixed_costs - total_variable_costs_on_sold
        if quantity_actually_sold == 0 and self.planned_production_current_step == 0:
            self.profit_last_step = -self.fixed_costs
        return self.profit_last_step

    def adapt(self):
        profit_threshold_factor = 0.05
        max_adaptation_step_ratio = 0.5 # Максимальное изменение мощности как доля от adaptation_rate

        if self.profit_last_step > self.fixed_costs * profit_threshold_factor: # Увеличиваем мощность если недовыпуск
            profit_norm_factor = self.profit_last_step / (abs(self.profit_last_step) + self.fixed_costs + 1e-6)
            profit_norm_factor = np.clip(profit_norm_factor, 0, 1.0) # От 0 до 1
            
            change_factor = self.adaptation_rate * profit_norm_factor
            change_factor = min(change_factor, self.adaptation_rate * max_adaptation_step_ratio) # Ограничиваем максимальное увеличение
            self.production_capacity *= (1 + change_factor)

        elif self.profit_last_step < -self.fixed_costs * profit_threshold_factor * 2: # Уменьшаем мощность если перевыпуск
            loss_norm_factor = abs(self.profit_last_step) / (abs(self.profit_last_step) + self.fixed_costs + 1e-6)
            loss_norm_factor = np.clip(loss_norm_factor, 0, 1.0)

            change_factor = self.adaptation_rate * loss_norm_factor * 0.75 # Коэфф. для снижения
            change_factor = min(change_factor, self.adaptation_rate * max_adaptation_step_ratio) # Ограничиваем максимальное уменьшение
            self.production_capacity *= (1 - change_factor)
        
        min_capacity_limit = max(self.initial_capacity * 0.1, 1.0)
        self.production_capacity = max(min_capacity_limit, min(self.production_capacity, 1000.0)) # Нижняя граница 1, а верхняя - 1000

class SingleMarketModel(BaseEconomicSystem):
    """
    Модель одного рынка с адаптивными потребителями и фирмами.
    Цена определяется взаимодействием спроса и предложения.
    """
    def __init__(self, params: Dict[str, Any]):
        super().__init__(params) 

        self.num_consumers = params.get("num_consumers", 20)
        self.num_firms = params.get("num_firms", 5)
        
        consumer_config = params.get("consumer_config", {"base_a": 20.0, "b_sensitivity": 0.8, "initial_income": 100.0, "adaptation_rate": 0.01, "memory_factor": 0.3}) # УМЕНЬШЕНА adaptation_rate
        firm_config = params.get("firm_config", {"fixed_costs": 10.0, "base_variable_cost_per_unit": 5.0, "initial_production_capacity": 35.0, "adaptation_rate": 0.01}) # УМЕНЬШЕНА adaptation_rate и initial_production_capacity

        self.consumers: List[SMM_Consumer] = [
            SMM_Consumer(id=i,
                     base_a=random.uniform(consumer_config["base_a"]*0.9, consumer_config["base_a"]*1.1), # Меньший разброс
                     b_sensitivity=random.uniform(consumer_config["b_sensitivity"]*0.9, consumer_config["b_sensitivity"]*1.1),
                     initial_income=random.uniform(consumer_config["initial_income"]*0.9, consumer_config["initial_income"]*1.1),
                     adaptation_rate=consumer_config["adaptation_rate"],
                     memory_factor=consumer_config["memory_factor"])
            for i in range(self.num_consumers)
        ]
        self.firms: List[SMM_Firm] = [
            SMM_Firm(id=j,
                 fixed_costs=random.uniform(firm_config["fixed_costs"]*0.9, firm_config["fixed_costs"]*1.1),
                 base_variable_cost_per_unit=random.uniform(firm_config["base_variable_cost_per_unit"]*0.9, firm_config["base_variable_cost_per_unit"]*1.1),
                 initial_production_capacity=random.uniform(firm_config["initial_production_capacity"]*0.8, firm_config["initial_production_capacity"]*1.2), # Небольшой разброс вокруг (новой??) базы
                 adaptation_rate=firm_config["adaptation_rate"],
                 tech_level=random.uniform(0.9, 1.1))
            for j in range(self.num_firms)
        ]

        self.market_price: float = params.get("initial_market_price", 15.0)
        self.actual_traded_quantity: float = 0.0
        
        self.government_net_revenue: float = 0.0 
        self.total_consumer_surplus: float = 0.0
        self.total_firm_profit: float = 0.0
        self.social_welfare: float = 0.0 

        self.tax_per_unit: float = 0.0
        self.subsidy_per_unit: float = 0.0
        self.price_ceiling: Optional[float] = None
        self.price_floor: Optional[float] = None
        self.government_purchase_quantity: float = 0.0 

        self.state = self._update_state() 
        self.history.append(self.state)


    def get_policy_descriptors(self) -> List[PolicyDescriptor]:
        common_context_vars = [
            "step", "market_price", "actual_traded_quantity", "government_net_revenue",
            "total_consumer_surplus", "total_firm_profit", "social_welfare",
            "avg_consumer_demand_param_a", "avg_firm_capacity", "avg_firm_vc",
            "num_consumers", "num_firms"
        ]
        
        descriptors = [
            PolicyDescriptor(
                policy_type_id="set_tax_per_unit",
                description="Устанавливает акцизный налог на единицу товара (t).",
                target_variable_name="tax_per_unit", value_type=float, value_range=(0.0, 50.0), 
                available_context_vars=common_context_vars + ["tax_per_unit"],
                constraints={'min_change_interval': 1, 'change_cost': 0.5} 
            ),
            PolicyDescriptor(
                policy_type_id="set_subsidy_per_unit",
                description="Устанавливает субсидию на единицу товара (s).",
                target_variable_name="subsidy_per_unit", value_type=float, value_range=(0.0, 30.0),
                available_context_vars=common_context_vars + ["subsidy_per_unit"],
                constraints={'min_change_interval': 1, 'change_cost': 0.5}
            ),
            PolicyDescriptor(
                policy_type_id="set_price_ceiling",
                description="Устанавливает потолок рыночной цены ($P_{max}$).",
                target_variable_name="price_ceiling", value_type=float, 
                value_range=(0.001, 100.0), 
                available_context_vars=common_context_vars + ["price_ceiling"],
                constraints={'change_cost': 0.2}
            ),
            PolicyDescriptor(
                policy_type_id="set_price_floor",
                description="Устанавливает пол рыночной цены ($P_{min}$).",
                target_variable_name="price_floor", value_type=float,
                value_range=(0.001, 80.0),
                available_context_vars=common_context_vars + ["price_floor"],
                constraints={'change_cost': 0.2}
            ),
            PolicyDescriptor(
                policy_type_id="set_government_purchase_quantity",
                description="Устанавливает объем госзакупок товара (G).",
                target_variable_name="government_purchase_quantity", value_type=float, value_range=(0.0, 200.0),
                available_context_vars=common_context_vars + ["government_purchase_quantity"],
                constraints={'change_cost': 0.1}
            )
        ]
        return descriptors

    def _update_state(self) -> Dict[str, Any]:
        serializable_policies = [p.to_dict() for p in self.active_policies]
        return {
            "step": self.current_step,
            "metrics": self.get_current_metrics(), 
            "active_policies_log": serializable_policies,
        }

    def get_current_metrics(self) -> Dict[str, float]:
        avg_consumer_a = sum(c.current_a for c in self.consumers) / self.num_consumers if self.num_consumers > 0 else 0
        avg_firm_capacity = sum(f.production_capacity for f in self.firms) / self.num_firms if self.num_firms > 0 else 0
        avg_firm_vc = sum(f.current_variable_cost_per_unit for f in self.firms) / self.num_firms if self.num_firms > 0 else 0
        
        self.social_welfare = self.total_consumer_surplus + self.total_firm_profit + self.government_net_revenue

        return {
            "step": float(self.current_step),
            "market_price": self.market_price,
            "actual_traded_quantity": self.actual_traded_quantity,
            "government_net_revenue": self.government_net_revenue,
            "total_consumer_surplus": self.total_consumer_surplus,
            "total_firm_profit": self.total_firm_profit,
            "social_welfare": self.social_welfare,
            "tax_per_unit": self.tax_per_unit, 
            "subsidy_per_unit": self.subsidy_per_unit,
            "price_ceiling": self.price_ceiling if self.price_ceiling is not None else -1.0, 
            "price_floor": self.price_floor if self.price_floor is not None else -1.0,
            "government_purchase_quantity": self.government_purchase_quantity,
            "avg_consumer_demand_param_a": avg_consumer_a,
            "avg_firm_capacity": avg_firm_capacity,
            "avg_firm_vc": avg_firm_vc,
            "num_consumers": float(self.num_consumers), 
            "num_firms": float(self.num_firms),
        }

    def get_state_for_agent(self) -> Dict[str, Any]:
        return {
            "metrics": self.get_current_metrics(),
            "active_policies": self.active_policies 
        }

    def apply_policy_change(self, policy_change: Optional[Policy]) -> None:
        if policy_change is None:
            # print(f"DEBUG: Step {self.current_step}: No policy change proposed by agent.")
            return

        eval_context = self.get_current_metrics() 
        policy_value = None
        # policy_application_info = f"Applying policy '{policy_change.policy_type}' (ID: {policy_change.id}): " # For debug print

        if policy_change._compiled_safe_code:
            policy_value = evaluate_safe_policy_code(policy_change._compiled_safe_code, eval_context)
            if policy_value is None and not policy_change.policy_type in ["set_price_ceiling", "set_price_floor"]: 
                # print(f"DEBUG: Error evaluating policy '{policy_change.policy_type}': expression result is None. Change not applied.")
                return
        elif policy_change.policy_type in ["set_price_ceiling", "set_price_floor"]:
            try:
                expr_str = policy_change.value_expression.strip().lower()
                if expr_str == 'none' or expr_str == '':
                    policy_value = None
                else:
                    policy_value = float(expr_str)
            except ValueError:
                # print(f"DEBUG: Invalid direct value for {policy_change.policy_type}: '{policy_change.value_expression}'. Not applied.")
                return
        else:
            # print(f"DEBUG: Policy '{policy_change.policy_type}' has no compiled code and is not a special type. Not applied.")
            return

        found_and_updated = False
        for i, p_active in enumerate(self.active_policies):
            if p_active.policy_type == policy_change.policy_type:
                self.active_policies[i] = policy_change 
                found_and_updated = True
                break
        if not found_and_updated:
            self.active_policies.append(policy_change) 

        descriptor = next((d for d in self.get_policy_descriptors() if d.policy_type_id == policy_change.policy_type), None)
        if descriptor and policy_value is not None: 
            if descriptor.value_range:
                min_v, max_v = descriptor.value_range
                if isinstance(policy_value, (int, float)):
                    if min_v is not None: policy_value = max(min_v, policy_value)
                    if max_v is not None: policy_value = min(max_v, policy_value)
        
        # attr_updated_log_part = "" # For debug print
        if policy_change.policy_type == "set_tax_per_unit" and isinstance(policy_value, (int,float)):
            self.tax_per_unit = float(policy_value)
            # attr_updated_log_part = f"Set tax_per_unit = {self.tax_per_unit}"
        elif policy_change.policy_type == "set_subsidy_per_unit" and isinstance(policy_value, (int,float)):
            self.subsidy_per_unit = float(policy_value)
            # attr_updated_log_part = f"Set subsidy_per_unit = {self.subsidy_per_unit}"
        elif policy_change.policy_type == "set_price_ceiling": 
            self.price_ceiling = float(policy_value) if isinstance(policy_value, (int,float)) and policy_value > 0 else None
            # attr_updated_log_part = f"Set price_ceiling = {self.price_ceiling}"
        elif policy_change.policy_type == "set_price_floor": 
            self.price_floor = float(policy_value) if isinstance(policy_value, (int,float)) and policy_value > 0 else None
            # attr_updated_log_part = f"Set price_floor = {self.price_floor}"
        elif policy_change.policy_type == "set_government_purchase_quantity" and isinstance(policy_value, (int,float)):
            self.government_purchase_quantity = float(policy_value)
            # attr_updated_log_part = f"Set government_purchase_quantity = {self.government_purchase_quantity}"
        # else:
            # attr_updated_log_part = "No matching model attribute or invalid value type for policy application."
        # print(f"DEBUG: {policy_application_info} Value from expr: {policy_value}. {attr_updated_log_part}")


    def step(self) -> None:
        # print(f"DEBUG: --- Step {self.current_step + 1} Market Simulation Start ---")

        price_for_consumers = self.market_price + self.tax_per_unit
        price_for_firms_decision = self.market_price + self.subsidy_per_unit

        total_demand_at_current_price = 0
        for consumer in self.consumers:
            total_demand_at_current_price += consumer.decide_quantity_demanded(price_for_consumers)
        
        effective_total_demand = total_demand_at_current_price + self.government_purchase_quantity

        total_supply_at_current_price = 0
        for firm in self.firms:
            total_supply_at_current_price += firm.decide_supply_quantity(price_for_firms_decision)

        new_market_price = self.market_price
        
        if effective_total_demand <= 1e-6 and total_supply_at_current_price <= 1e-6 :
            # Если нет ни спроса, ни предложения, цена может медленно падать или оставаться
            new_market_price = max(0.01, new_market_price * 0.98) # Медленное снижение, если рынок "мертв"
        else:
            # ИЗМЕНЕНО: количество итераций и adjustment_factor
            for _ in range(10): # Увеличено количество итераций
                p_cons_iter = new_market_price + self.tax_per_unit
                p_firm_iter = new_market_price + self.subsidy_per_unit

                iter_demand = sum(c.decide_quantity_demanded(p_cons_iter) for c in self.consumers) + self.government_purchase_quantity
                iter_supply = sum(f.decide_supply_quantity(p_firm_iter) for f in self.firms)

                if abs(iter_demand - iter_supply) < max(iter_demand, iter_supply) * 0.01 or abs(iter_demand - iter_supply) < 0.1: 
                    break
                
                adjustment_factor = 0.02 # ! ЗНАЧИТЕЛЬНО УМЕНЬШЕНО ДЛЯ СТАБИЛЬНОСТИ
                if iter_demand > iter_supply:
                    # Более мягкое изменение, особенно если разница большая
                    price_increase_factor = adjustment_factor * np.log1p(abs(iter_demand - iter_supply)) # Логарифм для сглаживания больших скачков
                    price_increase_factor = min(price_increase_factor, 0.1) # Ограничиваем максимальный рост за итерацию (например, 10%)
                    new_market_price *= (1 + price_increase_factor)

                else:
                    price_decrease_factor = adjustment_factor * np.log1p(abs(iter_demand - iter_supply))
                    price_decrease_factor = min(price_decrease_factor, 0.1)
                    new_market_price *= (1 - price_decrease_factor)

                new_market_price = max(0.01, new_market_price) 

        self.market_price = new_market_price

        if self.price_ceiling is not None and self.market_price > self.price_ceiling:
            self.market_price = self.price_ceiling
            # print(f"DEBUG: Price ceiling applied: {self.market_price}") 
        if self.price_floor is not None and self.market_price < self.price_floor:
            self.market_price = self.price_floor
            # print(f"DEBUG: Price floor applied: {self.market_price}") 

        final_price_consumers_pay = self.market_price + self.tax_per_unit
        final_price_firms_receive = self.market_price + self.subsidy_per_unit

        # Пересчитываем спрос потребителей при финальной цене
        final_total_demand_consumers_only = sum(c.decide_quantity_demanded(final_price_consumers_pay) for c in self.consumers)
        final_total_demand = final_total_demand_consumers_only + self.government_purchase_quantity
        
        # Пересчитываем предложение фирм при финальной цене
        final_total_supply = sum(f.decide_supply_quantity(final_price_firms_receive) for f in self.firms)
        
        self.actual_traded_quantity = min(final_total_demand, final_total_supply)
        # print(f"DEBUG: Market Cleared. Price: {self.market_price:.2f}, Traded Q: {self.actual_traded_quantity:.2f} (Demand: {final_total_demand:.2f}, Supply: {final_total_supply:.2f})")

        # Распределение фактического объема
        # 1. Госзакупки (если есть спрос и товар)
        gov_actual_purchase_final = 0.0
        if self.government_purchase_quantity > 0 and final_total_demand > 0: # Если правительство вообще хотело что-то купить
            # Доля госзакупок в общем спросе
            gov_share_in_total_final_demand = self.government_purchase_quantity / final_total_demand
            gov_actual_purchase_final = gov_share_in_total_final_demand * self.actual_traded_quantity
            gov_actual_purchase_final = min(gov_actual_purchase_final, self.government_purchase_quantity) # Не может купить больше, чем хотело

        quantity_remaining_for_consumers = self.actual_traded_quantity - gov_actual_purchase_final
        
        # 2. Потребители
        self.total_consumer_surplus = 0.0
        if final_total_demand_consumers_only > 1e-6 and quantity_remaining_for_consumers > 0:
            for c in self.consumers:
                # c.quantity_demanded_at_price уже содержит спрос при final_price_consumers_pay
                consumer_share_in_consumer_demand = c.quantity_demanded_at_price / final_total_demand_consumers_only
                
                c.quantity_bought_last_step = consumer_share_in_consumer_demand * quantity_remaining_for_consumers
                c.quantity_bought_last_step = max(0, min(c.quantity_bought_last_step, c.quantity_demanded_at_price)) # не может купить больше, чем хотел или меньше нуля
                
                if c.b > 1e-6 and c.quantity_bought_last_step > 1e-6:
                    p_intercept = c.current_a / c.b 
                    if p_intercept > final_price_consumers_pay:
                        c.utility_last_step = 0.5 * c.quantity_bought_last_step * (p_intercept - final_price_consumers_pay)
                    else: 
                        c.utility_last_step = 0
                else: 
                    c.utility_last_step = 0
                self.total_consumer_surplus += c.utility_last_step
        else: # Если не было спроса от потребителей или для них ничего не осталось
            for c in self.consumers:
                c.quantity_bought_last_step = 0
                c.utility_last_step = 0
        
        # 3. Фирмы
        self.total_firm_profit = 0.0
        if final_total_supply > 1e-6: 
            for f in self.firms:
                # f.planned_production_current_step уже содержит предложение при final_price_firms_receive
                firm_share_in_supply = f.planned_production_current_step / final_total_supply if final_total_supply > 0 else 0
                f.quantity_sold_last_step = firm_share_in_supply * self.actual_traded_quantity
                f.quantity_sold_last_step = max(0, min(f.quantity_sold_last_step, f.planned_production_current_step)) 
                
                f.calculate_profit_for_step(final_price_firms_receive, f.quantity_sold_last_step)
                self.total_firm_profit += f.profit_last_step
        else: # Если не было предложения
            for f in self.firms:
                f.quantity_sold_last_step = 0
                f.calculate_profit_for_step(final_price_firms_receive, 0) # Несут постоянные издержки
                self.total_firm_profit += f.profit_last_step


        tax_revenue = self.actual_traded_quantity * self.tax_per_unit
        subsidy_cost = self.actual_traded_quantity * self.subsidy_per_unit
        gov_purchase_spending = gov_actual_purchase_final * self.market_price # Закупает по рыночной цене (до налогов/субсидий)
        
        self.government_net_revenue = tax_revenue - subsidy_cost - gov_purchase_spending
        # print(f"DEBUG: Gov Net Revenue: {self.government_net_revenue:.2f} (Tax: {tax_revenue:.2f}, Subsidy: {subsidy_cost:.2f}, Purchases: {gov_purchase_spending:.2f} for Q:{gov_actual_purchase_final:.2f})")

        # Адаптация агентов
        for c in self.consumers:
            dissatisfaction = 0.0
            if c.quantity_demanded_at_price > 1e-6: # Спрос при финальной цене
                dissatisfaction = (c.quantity_demanded_at_price - c.quantity_bought_last_step) / c.quantity_demanded_at_price
            c.adapt(dissatisfaction)

        for f in self.firms:
            f.adapt()

        self.current_step += 1
        self.state = self._update_state()
        self.history.append(self.state)
        # print(f"DEBUG: --- Step {self.current_step} Market Simulation End. Social Welfare: {self.social_welfare:.2f} ---")


    def emulate_policy(self, policy: Policy, duration: int, agents_subset: Optional[List[AgentId]] = None) -> Dict[str, Any]:
        print("Warning: emulate_policy called but not implemented for SingleMarketModel.")
        return {"emulation_status": "not_implemented", "predicted_welfare_change": 0.0}