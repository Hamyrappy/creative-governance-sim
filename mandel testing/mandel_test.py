# abm_model.py
import random
import numpy as np
import matplotlib.pyplot as plt
import copy
import traceback # Import for error logging

class Good:
    def __init__(self, id): self.id = id

class AgentAsBuyer:
    def __init__(self, num_goods_to_buy_config):
        self.known_suppliers = {g_id: [] for g_id in range(num_goods_to_buy_config)}
        self.num_suppliers_to_evaluate_on_trade = 3
        self.max_known_suppliers_per_good = 10
        self.supplier_search_probability = 0.1
        self.num_new_suppliers_to_sample_on_search = 5

    def find_initial_suppliers(self, all_firms_list, good_id_to_find, num_goods, own_id_to_exclude=None):
        if good_id_to_find >= num_goods: return
        candidate_firms = [
            f for f in all_firms_list
            if f.sector_id == good_id_to_find and (own_id_to_exclude is None or f.id != own_id_to_exclude)
        ]
        if not candidate_firms: self.known_suppliers[good_id_to_find] = []; return

        num_to_select = min(len(candidate_firms), self.max_known_suppliers_per_good)
        if num_to_select > 0:
            self.known_suppliers[good_id_to_find] = random.sample([f.id for f in candidate_firms], k=num_to_select)
        else: self.known_suppliers[good_id_to_find] = []

    def _find_and_evaluate_new_potential_suppliers(self, good_id, all_firms_list, firms_map, own_id_to_exclude=None):
        new_candidates_info = []
        current_known_ids = set(self.known_suppliers.get(good_id, []))
        potential_new_supplier_objects = [
            f for f in all_firms_list
            if f.sector_id == good_id and
               f.id not in current_known_ids and
               (own_id_to_exclude is None or f.id != own_id_to_exclude)
        ]

        if not potential_new_supplier_objects: return []
        num_to_sample = min(len(potential_new_supplier_objects), self.num_new_suppliers_to_sample_on_search)
        if num_to_sample <= 0: return []

        sampled_new_firms = random.sample(potential_new_supplier_objects, k=num_to_sample)

        for firm in sampled_new_firms:
            if firm.price > 1e-3 :
                new_candidates_info.append({'firm_id': firm.id, 'price': firm.price, 'firm_obj': firm})

        new_candidates_info.sort(key=lambda x: x['price'])
        return [item['firm_obj'] for item in new_candidates_info]

    def _update_known_suppliers_list(self, good_id, new_best_supplier_obj, firms_map):
        current_known_ids = self.known_suppliers.get(good_id, [])
        if new_best_supplier_obj.id in current_known_ids: return

        if len(current_known_ids) < self.max_known_suppliers_per_good:
            current_known_ids.append(new_best_supplier_obj.id)
        else:
            most_expensive_known_id = -1
            max_price = -1.0
            found_expensive_to_replace = False
            for fid in current_known_ids:
                firm_obj = firms_map.get(fid)
                if firm_obj and firm_obj.price > 1e-3:
                    if firm_obj.price > max_price :
                         max_price = firm_obj.price
                         most_expensive_known_id = fid
                         found_expensive_to_replace = True

            if found_expensive_to_replace and new_best_supplier_obj.price < max_price:
                try:
                    idx_to_replace = current_known_ids.index(most_expensive_known_id)
                    current_known_ids[idx_to_replace] = new_best_supplier_obj.id
                except ValueError: # Should not happen if logic is correct
                    current_known_ids[random.randrange(len(current_known_ids))] = new_best_supplier_obj.id
            elif not found_expensive_to_replace and current_known_ids: # Replace random if no clearly expensive one or new is not cheaper
                 current_known_ids[random.randrange(len(current_known_ids))] = new_best_supplier_obj.id

        self.known_suppliers[good_id] = current_known_ids

    def attempt_to_update_supplier_network(self, good_id, all_firms_list, firms_map, own_id_to_exclude=None):
        if random.random() < self.supplier_search_probability:
            sorted_new_potential_suppliers = self._find_and_evaluate_new_potential_suppliers(
                good_id, all_firms_list, firms_map, own_id_to_exclude
            )
            if not sorted_new_potential_suppliers: return
            for new_supplier in sorted_new_potential_suppliers: # Try to add multiple new good suppliers
                self._update_known_suppliers_list(good_id, new_supplier, firms_map)

    def choose_supplier_from_known(self, good_id, firms_map, units_demanded):
        known_supplier_ids = self.known_suppliers.get(good_id, [])
        if not known_supplier_ids: return None

        if len(known_supplier_ids) > self.num_suppliers_to_evaluate_on_trade:
            ids_to_evaluate = random.sample(known_supplier_ids, k=self.num_suppliers_to_evaluate_on_trade)
        else:
            ids_to_evaluate = list(known_supplier_ids) # Evaluate all known if less than threshold

        potential_sellers_info = []
        for firm_id in ids_to_evaluate:
            firm = firms_map.get(firm_id)
            if firm and firm.sector_id == good_id and \
               (good_id < len(firm.inventory) and firm.inventory[good_id] > 1e-3) and \
               firm.price > 1e-3: # Ensure firm is active and has stock
                inventory_val = firm.inventory[good_id]
                potential_sellers_info.append({'firm_id': firm.id, 'price': firm.price, 'inventory': inventory_val, 'firm_obj': firm})

        if not potential_sellers_info: return None
        potential_sellers_info.sort(key=lambda x: x['price']) # Sort by price
        
        # Prefer supplier who can fulfill the whole order
        for seller_data in potential_sellers_info:
            if seller_data['inventory'] >= units_demanded - 1e-6 : # Allow for small float inaccuracies
                return seller_data['firm_obj']
        
        # If no one can fulfill, return the cheapest among evaluated (as per original logic)
        return potential_sellers_info[0]['firm_obj']


class Household(AgentAsBuyer):
    def __init__(self, id, num_goods):
        AgentAsBuyer.__init__(self, num_goods)
        self.id = id
        self.goods_stock = np.zeros(num_goods)
        self.consumption_technology = np.ones(num_goods) / num_goods if num_goods > 0 else np.array([])
        self.monetary_holdings = 200.0 + random.uniform(-50, 50)
        self.savings = 100.0 + random.uniform(-20, 20)
        self.savings = max(0, self.savings)
        self.wage_fallback = 0.3 + random.random() * 0.2 # MODIFIED: Lower fallback wage

        self.income = 0.0; self.net_income_current_period = 0.0
        self.expected_income = 25.0 + random.uniform(-5, 5) # Initial expected income
        self.expected_income_trend = 0.0; self.labor_supply = 1.0 # Normalized
        self.employed_by_firm_id = None; self.wage_received_this_period = 0.0

        self.saving_propensity = 0.05 # MODIFIED: Increased saving propensity slightly
        self.expected_income_update_rate = 0.1
        self.min_expected_income_threshold = 15.0 # MODIFIED: Lowered min expected income threshold

    def __repr__(self):
        return (f"Household(id={self.id}, money={self.monetary_holdings:.2f}, "
                f"savings={self.savings:.2f}, net_inc={self.net_income_current_period:.2f}, "
                f"emp_by={self.employed_by_firm_id}, wage_fall={self.wage_fallback:.2f})")

    def update_expectations(self):
        old_expected_income = self.expected_income
        self.expected_income = (self.expected_income_update_rate * self.net_income_current_period +
                                (1 - self.expected_income_update_rate) *
                                (self.expected_income + self.expected_income_trend))
        self.expected_income_trend = (self.expected_income_update_rate *
                                      (self.expected_income - old_expected_income) +
                                      (1 - self.expected_income_update_rate) *
                                      self.expected_income_trend)
        self.expected_income = max(0, self.expected_income) # Cannot be negative
        self.expected_income = max(self.expected_income, self.min_expected_income_threshold) # Floor for expectations

    def consume_goods(self): self.goods_stock = np.zeros_like(self.goods_stock) # Consume all stock

    def allocate_income_to_savings_and_consumption_budget(self, interest_rate_deposits, stage="full"):
        if stage == "interest_only" or stage == "full":
            interest_on_own_savings = self.savings * interest_rate_deposits
            self.monetary_holdings += interest_on_own_savings
            self.income += interest_on_own_savings 

        if stage == "deaton_rule" or stage == "full":
            current_cash = self.monetary_holdings
            if current_cash > self.expected_income: # Surplus cash
                surplus = current_cash - self.expected_income
                amount_to_add_to_savings = self.saving_propensity * surplus
                self.savings += amount_to_add_to_savings
                self.monetary_holdings -= amount_to_add_to_savings
            else: # Shortfall of cash vs expected income
                shortfall = self.expected_income - current_cash
                draw_propensity = max(self.saving_propensity, 0.1) 
                amount_to_draw_from_savings = draw_propensity * shortfall
                actual_draw = min(amount_to_draw_from_savings, self.savings) # Cannot draw more than available
                self.savings -= actual_draw
                self.monetary_holdings += actual_draw

        self.monetary_holdings = max(0, self.monetary_holdings)
        self.savings = max(0, self.savings)


class Firm(AgentAsBuyer):
    def __init__(self, id, sector_id, num_goods, owner_household_id, creation_period):
        AgentAsBuyer.__init__(self, num_goods)
        self.id = id; self.sector_id = sector_id; self.owner_household_id = owner_household_id
        self.num_goods = num_goods; self.creation_period = creation_period

        self.inventory = np.zeros(num_goods)
        if self.sector_id < num_goods:
             self.inventory[self.sector_id] = 15.0 + random.uniform(-3,3) 
        self.intermediary_inputs = np.ones(num_goods) * (8.0 + random.uniform(-2,2))
        self.fixed_capital = np.ones(num_goods) * (12.0 + random.uniform(-2,2))

        self.monetary_holdings = 700.0 + random.uniform(-100,100) 
        self.debt = 20.0 + random.uniform(-10,10); self.debt = max(0, self.debt)
        self.labor_employed = 0.0; self.employees = {};
        self.target_employment_history = []; self.labor_employed_history = []

        self.intermediary_input_coeffs = np.ones(num_goods) * 0.05 
        self.fixed_capital_coeffs = np.ones(num_goods) * 0.1; self.labor_coeff = 0.3 

        self.actual_production = 0.0; self.production_volume_last_period = 0.0
        self.target_production = 3.0 
        self.min_target_production_threshold = 1.0 

        self.expected_sales = 3.0 * (1.0 + random.uniform(-0.1,0.1))
        self.min_expected_sales_threshold = 10.0 # MODIFIED: Increased min expected sales
        self.expected_sales_trend = 0.0

        self.target_employment = self.target_production * self.labor_coeff
        self.sales_last_period_volume = 0.0; self.sales_volume_history = []
        self.sales_revenue = 0.0; self.profit_history = []; self.profit_current_period = 0.0
        self.costs_current_period = 0.0; self.costs_last_period = 0.0;
        self.price = 1.0 + random.uniform(-0.05, 0.05)
        self.markup = 0.10 + random.random() * 0.10
        self.reservation_wage_index = 0.9 + random.random() * 0.2

        self.max_debt_to_capital_ratio = 3.0 
        self.mutation_rate = 0.05
        self.peers_sampling_rate_genetic = 0.25; self.imitation_rate = 0.3
        self.mutation_strength = 0.15; self.fitness_alpha = 0.5 # For markup evolution (if used)
        self.min_labor_capacity_ratio = 0.8; self.dividend_rate = 0.3
        self.expected_sales_update_rate = 0.1; self.price_update_smoothing = 0.25
        self.inventory_to_sales_ratio_target = 0.2; self.capital_depreciation_rate = 0.02
        self.inventory_depreciation_rate = 0.01
        
        self.investment_funds_from_profit_share = 0.7
        self.investment_funds_from_cash_share = 0.2
        self.profit_window_for_imitation = 5; self.wage_index_evolution_window = 3
        
        # New parameters for investment logic
        self.capital_utilization_threshold_for_expansion = 0.85
        self.expansion_investment_target_increase_pct = 0.15


    def __repr__(self):
        avg_profit = self.get_average_profit(self.profit_window_for_imitation)
        return (f"Firm(id={self.id}, sec={self.sector_id}, pr={self.price:.2f}, mu={self.markup:.3f}, "
                f"L_coeff={self.labor_coeff:.3f}, wage_idx={self.reservation_wage_index:.2f}, "
                f"mon={self.monetary_holdings:.2f}, debt={self.debt:.2f}, "
                f"avg_P{self.profit_window_for_imitation}={avg_profit:.2f})")

    def get_average_profit(self, num_periods_for_avg):
        if not self.profit_history: return 0.0
        actual_periods = min(num_periods_for_avg, len(self.profit_history))
        if actual_periods <= 0: return 0.0
        relevant_history = self.profit_history[-actual_periods:]
        return np.mean(relevant_history) if relevant_history else 0.0

    def get_potential_production_from_capital(self):
        """Estimates production capacity based solely on current fixed capital and its coefficients."""
        cap_from_cap_list_plan = []
        for i in range(self.num_goods):
            if self.fixed_capital_coeffs[i] > 1e-6:
                cap_from_cap_list_plan.append(self.fixed_capital[i] / self.fixed_capital_coeffs[i])
            elif self.fixed_capital[i] > 1e-6: # Has capital but zero coefficient (implies infinite capacity from this type if used)
                cap_from_cap_list_plan.append(float('inf'))
        potential_prod_from_capital = min(cap_from_cap_list_plan) if cap_from_cap_list_plan else 0.0
        return potential_prod_from_capital

    def produce(self, labor_productivity_sector):
        effective_labor = self.labor_employed * labor_productivity_sector
        
        capacity_from_capital = self.get_potential_production_from_capital()

        if self.labor_coeff > 1e-6:
            capacity_from_labor = effective_labor / self.labor_coeff if effective_labor > 1e-6 else 0.0
        else: capacity_from_labor = float('inf')

        capacity_from_inputs_list = []
        for i in range(len(self.intermediary_inputs)):
            if self.intermediary_input_coeffs[i] > 1e-6:
                capacity_from_inputs_list.append(self.intermediary_inputs[i] / self.intermediary_input_coeffs[i])
            elif self.intermediary_inputs[i] > 1e-6 and self.intermediary_input_coeffs[i] < 1e-6: # Has input, zero coeff
                capacity_from_inputs_list.append(float('inf'))
        capacity_from_inputs = min(capacity_from_inputs_list) if capacity_from_inputs_list else 0.0 # 0 if no inputs/coeffs

        max_producible_quantity = min(capacity_from_capital, capacity_from_labor, capacity_from_inputs)
        self.actual_production = min(max_producible_quantity, self.target_production)
        self.actual_production = max(0, self.actual_production)
        
        if self.actual_production > 1e-6:
            for i in range(len(self.intermediary_inputs)):
                consumed_input = self.actual_production * self.intermediary_input_coeffs[i]
                self.intermediary_inputs[i] -= consumed_input
            
            depreciation_amount_vector = self.fixed_capital * self.capital_depreciation_rate
            self.fixed_capital -= depreciation_amount_vector
            
            if self.sector_id < len(self.inventory): self.inventory[self.sector_id] += self.actual_production

        self.inventory *= (1 - self.inventory_depreciation_rate) # Inventory depreciation
        self.inventory = np.maximum(0, self.inventory)
        self.fixed_capital = np.maximum(0, self.fixed_capital)
        self.intermediary_inputs = np.maximum(0, self.intermediary_inputs)
        return self.actual_production

    def update_price(self):
        target_price = self.price # Default to current price if no better info
        # Use costs_last_period and sales/production_volume_last_period for unit cost calculation
        if self.sales_last_period_volume > 1e-3 and self.costs_last_period > 1e-6:
            unit_cost_last_period = self.costs_last_period / self.sales_last_period_volume
            target_price = (1 + self.markup) * unit_cost_last_period
        elif self.production_volume_last_period > 1e-3 and self.costs_last_period > 1e-6: # Fallback to production volume
            unit_cost_last_period = self.costs_last_period / self.production_volume_last_period
            target_price = (1 + self.markup) * unit_cost_last_period

        self.price = ((1 - self.price_update_smoothing) * self.price + self.price_update_smoothing * target_price)
        self.price = max(0.01, self.price) # Price floor

    def update_expectations_and_target_production(self):
        old_expected_sales_value = self.expected_sales
        # sales_revenue is from the current period's trading step
        self.expected_sales = (self.expected_sales_update_rate * self.sales_revenue + 
                               (1 - self.expected_sales_update_rate) * (self.expected_sales + self.expected_sales_trend))
        self.expected_sales_trend = (self.expected_sales_update_rate * (self.expected_sales - old_expected_sales_value) +
                                     (1 - self.expected_sales_update_rate) * self.expected_sales_trend)
        self.expected_sales = max(0, self.expected_sales)
        self.expected_sales = max(self.expected_sales, self.min_expected_sales_threshold)

        current_price_for_target = self.price if self.price > 1e-3 else 1.0
        target_production_units_from_sales_expectation = self.expected_sales / current_price_for_target

        # Inventory adjustment based on target sales
        if self.sector_id < len(self.inventory):
            current_inventory_own_good = self.inventory[self.sector_id]
            target_inventory_units = self.inventory_to_sales_ratio_target * target_production_units_from_sales_expectation
            inventory_adjustment_units = target_inventory_units - current_inventory_own_good
            self.target_production = target_production_units_from_sales_expectation + inventory_adjustment_units
        else: # Should not happen for firms producing goods
            self.target_production = target_production_units_from_sales_expectation

        self.target_production = max(0, self.target_production)
        self.target_production = max(self.target_production, self.min_target_production_threshold)

        self.target_employment = self.target_production * self.labor_coeff
        self.target_employment = max(0, self.target_employment)

        self.target_employment_history.append(self.target_employment)
        if len(self.target_employment_history) > self.wage_index_evolution_window + 2:
            self.target_employment_history.pop(0)

    def accounting(self, households_map_ignored, interest_rate_debt, market_prices_for_costs):
        # Wage costs (self.costs_current_period already includes them from main loop)
        # Interest on debt payment
        interest_on_debt_payment = self.debt * interest_rate_debt
        self.monetary_holdings -= interest_on_debt_payment
        self.costs_current_period += interest_on_debt_payment

        # Depreciation cost of fixed capital
        depreciation_cost_value = 0
        # self.fixed_capital is K_t (after depreciation). We need K_{t-1} to calculate (K_{t-1} * rate * price)
        for good_id in range(self.num_goods):
            if (1 - self.capital_depreciation_rate) > 1e-6 : # Avoid division by zero
                 # Estimate capital at start of period before depreciation
                 estimated_k_old_good_id = self.fixed_capital[good_id] / (1 - self.capital_depreciation_rate)
                 depreciated_amount_units = estimated_k_old_good_id * self.capital_depreciation_rate
                 price_of_capital_good = market_prices_for_costs.get(good_id, 1.0)
                 depreciation_cost_value += depreciated_amount_units * price_of_capital_good
            # If rate is 100%, K_old was self.fixed_capital[good_id] / 0 -> problematic.
            # If rate is 100%, self.fixed_capital[good_id] should be 0 (or very small due to float).
            # Assume if rate is 1.0, K_old was small and fully depreciated.
            # The current logic might slightly miscalculate if rate is exactly 1.0 or 0.0 without K_old storage.
            # For typical small depreciation rates, this estimation is acceptable.

        self.costs_current_period += depreciation_cost_value

        if self.monetary_holdings < 0: # Cover deficit with debt
            amount_to_borrow = -self.monetary_holdings
            self.debt += amount_to_borrow
            self.monetary_holdings = 0.0

        if self.monetary_holdings > 0 and self.debt > 0: # Repay debt if possible
            repayment_amount = min(self.monetary_holdings * 0.5, self.debt) 
            self.debt -= repayment_amount
            self.monetary_holdings -= repayment_amount
            self.debt = max(0, self.debt)

        self.profit_current_period = self.sales_revenue - self.costs_current_period
        self.profit_history.append(self.profit_current_period)
        
        self.monetary_holdings = max(0, self.monetary_holdings)

    def plan_investment(self, financial_system):
        desired_investment_units = np.zeros(self.num_goods)
        
        # --- Standard investment to meet target production and cover depreciation ---
        for good_id in range(self.num_goods):
            if self.fixed_capital_coeffs[good_id] > 1e-6: # If this good is used as capital
                required_capital_good_for_target = self.target_production * self.fixed_capital_coeffs[good_id]
                investment_to_cover_depreciation = self.fixed_capital[good_id] * self.capital_depreciation_rate
                # Ensure at least depreciation is planned, plus a small buffer
                min_investment_target_units = investment_to_cover_depreciation * 1.05 

                capital_good_shortfall = required_capital_good_for_target - self.fixed_capital[good_id]
                
                planned_units_std = max(capital_good_shortfall, 0) # For growth
                planned_units_std = max(planned_units_std, min_investment_target_units) # For depreciation
                desired_investment_units[good_id] = planned_units_std
        
        # --- Expansionary investment if capital utilization is high ---
        potential_prod_from_capital = self.get_potential_production_from_capital()
        capital_utilization = 0
        if potential_prod_from_capital > 1e-6:
            # Use production_volume_last_period as proxy for current utilization driver
            capital_utilization = self.production_volume_last_period / potential_prod_from_capital
        
        if capital_utilization > self.capital_utilization_threshold_for_expansion:
            for good_id_exp in range(self.num_goods):
                if self.fixed_capital_coeffs[good_id_exp] > 1e-6: # Only expand capital types used
                    expansion_k_units = self.fixed_capital[good_id_exp] * self.expansion_investment_target_increase_pct
                    desired_investment_units[good_id_exp] += expansion_k_units

        # --- Financial viability & Loan requests for total desired investment ---
        current_capital_value = sum(self.fixed_capital[i] * financial_system.get_market_price_of_good(i) for i in range(self.num_goods))
        recent_avg_profit = self.get_average_profit(self.profit_window_for_imitation)
        target_profit_rate_for_scaling = financial_system.interest_rate_debt + financial_system.target_inflation_rate
        if target_profit_rate_for_scaling < 1e-6: target_profit_rate_for_scaling = 1e-6
        
        # Profitability scaling factor (applies to the total desired investment)
        # Add a small base to allow some investment even if currently unprofitable but expecting growth
        profit_scaling_factor = max(0.05, min(1.0, (recent_avg_profit / (current_capital_value + 1e-6) + 0.01) / target_profit_rate_for_scaling))

        total_cost_of_desired_investment = 0
        for gd_id_cost_calc in range(self.num_goods):
            desired_investment_units[gd_id_cost_calc] *= profit_scaling_factor # Scale down if not very profitable
            desired_investment_units[gd_id_cost_calc] = max(0, desired_investment_units[gd_id_cost_calc]) # Ensure non-negative
            price_of_cap_good = financial_system.get_market_price_of_good(gd_id_cost_calc)
            if price_of_cap_good is None or price_of_cap_good < 1e-3: price_of_cap_good = 1.0 # Fallback price
            total_cost_of_desired_investment += desired_investment_units[gd_id_cost_calc] * price_of_cap_good
            
        # Own funds available for investment
        # profit_current_period is from previous accounting cycle.
        retained_profit_for_investment = max(0, self.profit_current_period * (1 - self.dividend_rate)) * \
                                         self.investment_funds_from_profit_share
        cash_for_investment_before_loan = self.monetary_holdings * self.investment_funds_from_cash_share
        own_funds_available = max(0, retained_profit_for_investment + cash_for_investment_before_loan)
        
        loan_needed = total_cost_of_desired_investment - own_funds_available
        if loan_needed > 1e-3:
             financial_system.provide_loan_to_firm(self, loan_needed, current_capital_value)
             # Note: desired_investment_units is what the firm *wants*. Actual purchase will be limited by supplier inventory.
             # The loan directly increases monetary_holdings for the purchase phase.

        return desired_investment_units


    def calculate_unit_cost(self, benchmark_wage, market_prices_snapshot):
        cost = self.labor_coeff * benchmark_wage # Labor cost per unit
        for i in range(self.num_goods): # Intermediary inputs cost per unit
            cost += self.intermediary_input_coeffs[i] * market_prices_snapshot.get(i, 1.0)
        for i in range(self.num_goods): # Capital depreciation cost per unit
            cost += self.fixed_capital_coeffs[i] * market_prices_snapshot.get(i, 1.0) * self.capital_depreciation_rate
        return max(0.01, cost) # Cost floor

    def evolve_technology(self, peers, benchmark_wage, market_prices_snapshot):
        if peers and random.random() < self.imitation_rate:
            num_peers_to_sample = max(1, int(len(peers) * self.peers_sampling_rate_genetic))
            sampled_peers = random.sample(peers, k=min(len(peers), num_peers_to_sample))
            best_peer = None; min_peer_unit_cost = float('inf')
            for peer in sampled_peers:
                peer_unit_cost = peer.calculate_unit_cost(benchmark_wage, market_prices_snapshot)
                if peer_unit_cost < min_peer_unit_cost: min_peer_unit_cost = peer_unit_cost; best_peer = peer
            
            current_unit_cost = self.calculate_unit_cost(benchmark_wage, market_prices_snapshot)
            if best_peer and min_peer_unit_cost < current_unit_cost:
                self.intermediary_input_coeffs = copy.deepcopy(best_peer.intermediary_input_coeffs)
                self.fixed_capital_coeffs = copy.deepcopy(best_peer.fixed_capital_coeffs)
                self.labor_coeff = best_peer.labor_coeff
        
        if random.random() < self.mutation_rate:
            choice = random.random()
            if choice < 0.4 and self.num_goods > 0: # Mutate one intermediary input coeff
                idx_to_mutate = random.randrange(self.num_goods)
                self.intermediary_input_coeffs[idx_to_mutate] *= (1 + (random.random() * 2 - 1) * self.mutation_strength)
                self.intermediary_input_coeffs[idx_to_mutate] = max(0.0, self.intermediary_input_coeffs[idx_to_mutate])
            elif choice < 0.8 and self.num_goods > 0: # Mutate one fixed capital coeff
                idx_to_mutate = random.randrange(self.num_goods)
                self.fixed_capital_coeffs[idx_to_mutate] *= (1 + (random.random() * 2 - 1) * self.mutation_strength)
                self.fixed_capital_coeffs[idx_to_mutate] = max(0.001, self.fixed_capital_coeffs[idx_to_mutate]) # Must be > 0
            else: # Mutate labor coeff
                self.labor_coeff *= (1 + (random.random() * 2 - 1) * self.mutation_strength)
                self.labor_coeff = max(0.01, self.labor_coeff) # Must be > 0

    def evolve_markup(self, peers):
        # Fitness: Mandel 2012, p.112, item 13: "convex combination of profit and sales growth rates."
        # Current code uses average profit. Keeping for now due to complexity of tracking growth rates.
        if peers and random.random() < self.imitation_rate:
            num_peers_to_sample = max(1, int(len(peers) * self.peers_sampling_rate_genetic))
            sampled_peers = random.sample(peers, k=min(len(peers), num_peers_to_sample))
            best_peer = None; max_peer_fitness_val = -float('inf') 
            
            for peer in sampled_peers:
                # Simple fitness: average profit. TODO: Implement composite fitness.
                peer_fitness_val = peer.get_average_profit(self.profit_window_for_imitation)
                if peer_fitness_val > max_peer_fitness_val: max_peer_fitness_val = peer_fitness_val; best_peer = peer
            
            current_fitness_val = self.get_average_profit(self.profit_window_for_imitation)
            if best_peer and max_peer_fitness_val > current_fitness_val: self.markup = best_peer.markup
        
        if random.random() < self.mutation_rate:
            self.markup *= (1 + (random.random() * 2 - 1) * self.mutation_strength)
            self.markup = max(0.01, self.markup) # Markup floor

    def _calculate_wage_index_fitness(self):
        # Fitness: Mandel 2012, p.112, item 14: "share of vacancies filled."
        if not self.target_employment_history or not self.labor_employed_history: return -1.0 # Default if no history
        
        # Use most recent history for calculation
        relevant_targets = self.target_employment_history[-self.wage_index_evolution_window:]
        relevant_employed = self.labor_employed_history[-self.wage_index_evolution_window:]
        if not relevant_targets or not relevant_employed or len(relevant_targets) != len(relevant_employed): return -1.0

        total_target_sum = sum(relevant_targets)
        total_employed_sum = sum(relevant_employed)
        
        if total_target_sum < 1e-3: return 1.0 # If no target, 100% efficient

        share_filled = total_employed_sum / total_target_sum
        fitness = min(1.0, max(0.0, share_filled)) # Bounded between 0 and 1

        if abs(fitness - 1.0) < 1e-3 : # If (almost) all vacancies filled
            # Prefer lower wage index if all vacancies are filled (secondary criteria)
            fitness += (1.5 - self.reservation_wage_index) * 0.05 # Smaller bonus
        return fitness

    def evolve_wage_index(self, peers):
        if peers and random.random() < self.imitation_rate:
            num_peers_to_sample = max(1, int(len(peers) * self.peers_sampling_rate_genetic))
            sampled_peers = random.sample(peers, k=min(len(peers), num_peers_to_sample))
            best_peer = None; max_peer_fitness = -float('inf')
            for peer in sampled_peers:
                peer_fitness = peer._calculate_wage_index_fitness()
                if peer_fitness > max_peer_fitness: max_peer_fitness = peer_fitness; best_peer = peer
            
            current_fitness = self._calculate_wage_index_fitness()
            if best_peer and max_peer_fitness > current_fitness: self.reservation_wage_index = best_peer.reservation_wage_index
        
        if random.random() < self.mutation_rate:
            self.reservation_wage_index *= (1 + (random.random() * 2 - 1) * self.mutation_strength)
            self.reservation_wage_index = max(0.5, min(2.0, self.reservation_wage_index)) # Bounds

class Sector:
    def __init__(self, id, num_goods):
        self.id = id; self.benchmark_wage = 10.0; self.labor_productivity_index = 1.0
        self.num_active_firms = 0; self.risk_premium = 0.02; 
        # self.entry_rate_firms, self.exit_rate_firms not directly used for now; entry/exit managed by Sim
        self.average_profit_rate_history = []

        # MODIFIED: Parameter for productivity growth based on K_t/K_{t-1}
        self.max_productivity_growth_from_capital_ratio = 0.005 # e.g., 0.5% per period max increase
        self.min_labor_productivity_index = 0.25; self.min_benchmark_wage = 1.0
        
        self.capital_stock_value_end_of_last_period = 0.0 # Initialize for K_{t-1} tracking

    def calculate_average_sector_profit_rate(self, firms_in_sector, profit_history_length_for_avg, market_prices_snapshot):
        if not firms_in_sector: return 0.0
        total_profit_sum = 0; total_capital_value_sum = 0; num_valid_firms = 0
        for firm in firms_in_sector:
            avg_profit_firm = firm.get_average_profit(profit_history_length_for_avg)
            firm_capital_value = sum(firm.fixed_capital[i] * market_prices_snapshot.get(i, 1.0) for i in range(firm.num_goods))
                
            if firm_capital_value > 1e-3:
                total_profit_sum += avg_profit_firm; total_capital_value_sum += firm_capital_value
                num_valid_firms +=1
        if num_valid_firms > 0 and total_capital_value_sum > 1e-3: 
            return total_profit_sum / total_capital_value_sum
        return 0.0 

    def update_labor_productivity(self, firms_in_sector, market_prices_snapshot):
        # Mandel et al. (2010), eq (2) links labor productivity growth to capital stock growth:
        # omega_g^{t+1} / omega_g^t related to K_g^t / K_g^{t-1}
        
        if not firms_in_sector: # No firms, no change or slow decay?
            # For now, no change if no firms. Could add a decay factor.
            return

        # Calculate K_g^t (capital in sector g at end of current period)
        current_total_capital_stock_value_end_of_period = 0
        for firm in firms_in_sector:
            for i in range(firm.num_goods):
                price_of_capital_good = market_prices_snapshot.get(i, 1.0)
                current_total_capital_stock_value_end_of_period += firm.fixed_capital[i] * price_of_capital_good
        
        capital_growth_ratio = 1.0 # Default to no growth
        if self.capital_stock_value_end_of_last_period > 1e-3: # Avoid division by zero if K_{t-1} was zero
            capital_growth_ratio = current_total_capital_stock_value_end_of_period / self.capital_stock_value_end_of_last_period
        elif current_total_capital_stock_value_end_of_period > 1e-3: # Grew from (near) zero
            capital_growth_ratio = 1.0 + self.max_productivity_growth_from_capital_ratio # Max growth from small base
            
        # growth_rate_capital is (K_t / K_{t-1}) - 1
        growth_rate_from_capital = capital_growth_ratio - 1.0
        
        # Apply limits to the growth rate itself
        effective_growth_rate = np.clip(growth_rate_from_capital,
                                        -self.max_productivity_growth_from_capital_ratio, # Max decrease rate
                                        self.max_productivity_growth_from_capital_ratio) # Max increase rate
        
        growth_factor = 1.0 + effective_growth_rate

        self.labor_productivity_index *= growth_factor
        self.benchmark_wage *= growth_factor # Benchmark wage indexed on productivity

        self.labor_productivity_index = max(self.labor_productivity_index, self.min_labor_productivity_index)
        self.benchmark_wage = max(self.benchmark_wage, self.min_benchmark_wage)
        
        # Update K_{t-1} for the next period's calculation
        self.capital_stock_value_end_of_last_period = current_total_capital_stock_value_end_of_period


class Government:
    def __init__(self):
        self.monetary_holdings = 0.0; self.unemployment_benefit_rate = 0.4
        self.tax_rate = 0.2; self.total_unemployment_benefits_paid_this_period = 0.0
        self.min_tax_rate = 0.05; self.max_tax_rate = 0.50; self.gov_debt_limit = -2000.0
    
    def set_tax_rate_and_collect_taxes(self, households, average_wage_economy):
        self.total_unemployment_benefits_paid_this_period = 0.0; num_unemployed = 0
        total_gross_income_households_before_benefits_and_tax = sum(hh.income for hh in households)

        for hh in households:
            if hh.employed_by_firm_id is None: num_unemployed += 1
        
        benefit_per_unemployed = self.unemployment_benefit_rate * average_wage_economy if average_wage_economy > 0 else 0
        benefit_per_unemployed = max(0, benefit_per_unemployed)
        self.total_unemployment_benefits_paid_this_period = num_unemployed * benefit_per_unemployed
        
        total_taxable_income_base = total_gross_income_households_before_benefits_and_tax + self.total_unemployment_benefits_paid_this_period
        
        if total_taxable_income_base > 1e-3: # Gov aims to balance budget
            self.tax_rate = self.total_unemployment_benefits_paid_this_period / total_taxable_income_base
            self.tax_rate = np.clip(self.tax_rate, self.min_tax_rate, self.max_tax_rate)
        else: self.tax_rate = (self.min_tax_rate + self.max_tax_rate) / 2 
        
        total_taxes_collected_this_period = 0
        for hh in households:
            gross_income_this_hh = hh.income 
            benefit_for_this_hh_value = 0
            if hh.employed_by_firm_id is None: benefit_for_this_hh_value = benefit_per_unemployed
            
            taxable_base_for_hh = gross_income_this_hh + benefit_for_this_hh_value
            tax_amount = taxable_base_for_hh * self.tax_rate
            tax_amount = max(0, tax_amount)
            
            hh.monetary_holdings -= tax_amount
            if benefit_for_this_hh_value > 0: hh.monetary_holdings += benefit_for_this_hh_value
            
            total_taxes_collected_this_period += tax_amount
            hh.net_income_current_period = gross_income_this_hh - tax_amount + benefit_for_this_hh_value
            hh.net_income_current_period = max(0, hh.net_income_current_period)

        self.monetary_holdings += total_taxes_collected_this_period - self.total_unemployment_benefits_paid_this_period
        if self.monetary_holdings < self.gov_debt_limit:
             self.monetary_holdings = self.gov_debt_limit

class FinancialSystem:
    def __init__(self):
        self.interest_rate_deposits = 0.01; self.interest_rate_debt = 0.03
        self.market_prices_snapshot = {}; self.num_goods = 0 # Set by Sim
        self.target_unemployment_rate = 0.05; self.target_inflation_rate = 0.02
        self.natural_interest_rate = 0.02; self.unemployment_coeff = 0.2; self.inflation_coeff = 1.0
        self.min_debt_rate = 0.005; self.max_debt_rate = 0.20
        self.min_deposit_rate_spread = 0.005; self.min_deposit_rate_abs = 0.001
    
    def update_interest_rate_taylor_rule(self, current_unemployment_rate, current_inflation_rate):
        base_rate = (self.natural_interest_rate + self.target_inflation_rate + 
                     self.inflation_coeff * (current_inflation_rate - self.target_inflation_rate) -
                     self.unemployment_coeff * (current_unemployment_rate - self.target_unemployment_rate))
        
        self.interest_rate_debt = np.clip(base_rate, self.min_debt_rate, self.max_debt_rate)
        
        self.interest_rate_deposits = self.interest_rate_debt - 0.02 # Fixed spread
        self.interest_rate_deposits = max(self.interest_rate_deposits, self.min_deposit_rate_abs)
        self.interest_rate_deposits = min(self.interest_rate_deposits, self.interest_rate_debt - self.min_deposit_rate_spread)
        self.interest_rate_deposits = max(self.interest_rate_deposits, self.min_deposit_rate_abs) # Re-check after spread

    def provide_loan_to_firm(self, firm, amount_requested, firm_capital_value):
        eligible_for_loan = True
        if amount_requested <= 1e-3: eligible_for_loan = False
        
        # Loan eligibility checks
        if firm_capital_value < 1e-3 and amount_requested > 0 : eligible_for_loan = False
        elif (firm.debt + amount_requested) / (firm_capital_value + 1e-6) >= firm.max_debt_to_capital_ratio:
            eligible_for_loan = False
        if amount_requested > firm_capital_value * 0.7 : eligible_for_loan = False # Loan not > 70% of current capital
        if firm_capital_value < 5.0 and amount_requested > 0: eligible_for_loan = False # Min capital for loan

        if eligible_for_loan:
            firm.monetary_holdings += amount_requested
            firm.debt += amount_requested
            return True
        return False

    def update_market_prices_snapshot(self, firms):
        self.market_prices_snapshot = {}; prices_by_good = {}
        for firm in firms: # Only active firms with valid prices
            good_id = firm.sector_id
            if good_id not in prices_by_good: prices_by_good[good_id] = []
            if firm.price > 1e-3: prices_by_good[good_id].append(firm.price)
        
        for good_id, price_list in prices_by_good.items():
            if price_list: self.market_prices_snapshot[good_id] = np.mean(price_list)
            # If no firms produce a good, its price won't be in snapshot. get_market_price_of_good handles default.
            
    def get_market_price_of_good(self, good_id):
        return self.market_prices_snapshot.get(good_id, 1.0) # Default to 1.0 if no price

class Simulation:
    def __init__(self, num_goods, num_households, num_firms_per_sector_list, num_sectors):
        self.num_goods = num_goods; self.num_sectors = num_sectors
        if num_sectors != len(num_firms_per_sector_list):
            raise ValueError("num_firms_per_sector_list length must match num_sectors")

        self.households = [Household(i, num_goods) for i in range(num_households)]
        # MODIFIED: Sector initialization for K_{t-1}
        self.sectors = [Sector(s_id, num_goods) for s_id in range(num_sectors)]
        
        self.firms = []; self.next_firm_id = 0

        for s_id in range(num_sectors):
            num_firms_in_this_sector = num_firms_per_sector_list[s_id]
            sector_initial_capital_value = 0
            for _ in range(num_firms_in_this_sector):
                owner_hh_id = random.choice(self.households).id if self.households else -1
                firm = Firm(id=self.next_firm_id, sector_id=s_id, num_goods=num_goods,
                            owner_household_id=owner_hh_id, creation_period=0)
                # Initial capital value sum for sector K_{t-1} init
                sector_initial_capital_value += sum(firm.fixed_capital[i] * 1.0 for i in range(self.num_goods)) # Assume initial price 1.0

                self.next_firm_id += 1
                self.firms.append(firm)
            
            if self.num_sectors > s_id : # Check index
                 self.sectors[s_id].num_active_firms = len(self.get_firms_in_sector(s_id))
                 self.sectors[s_id].capital_stock_value_end_of_last_period = sector_initial_capital_value


            # Default tech coefficients (can be refined with data later)
            for firm_in_sec in self.get_firms_in_sector(s_id):
                firm_in_sec.intermediary_input_coeffs = np.random.rand(num_goods) * 0.05
                if s_id < num_goods: firm_in_sec.intermediary_input_coeffs[s_id] = 0 
                firm_in_sec.fixed_capital_coeffs = np.random.rand(num_goods) * 0.1
                firm_in_sec.labor_coeff = 0.2 + np.random.rand() * 0.2
                firm_in_sec.target_production = firm_in_sec.min_target_production_threshold # Start small
                firm_in_sec.expected_sales = firm_in_sec.min_expected_sales_threshold # Start small
                firm_in_sec.target_employment = firm_in_sec.target_production * firm_in_sec.labor_coeff

        self.firms_map = {f.id: f for f in self.firms}
        if self.firms: # Initial supplier network
            for hh in self.households:
                for g_id in range(self.num_goods): hh.find_initial_suppliers(self.firms, g_id, self.num_goods)
            for firm_agent in self.firms:
                 for g_id in range(self.num_goods):
                     firm_agent.find_initial_suppliers(self.firms, g_id, self.num_goods, own_id_to_exclude=firm_agent.id)

        self.government = Government(); self.financial_system = FinancialSystem()
        self.financial_system.num_goods = num_goods; self.current_period = 0
        
        # Simulation run parameters
        self.max_periods = 600 
        self.genetic_evolution_periodicity = 5
        self.firm_management_periodicity = 7 

        # History tracking
        self.history_gdp = []; self.history_unemployment_rate = []; self.history_avg_price = []
        self.history_inflation = []; self.history_total_investment_value = []
        self.history_sector_productivity = []; self.history_num_firms = []

        # MODIFIED: Parameters for firm entry/exit/bankruptcy (made more lenient)
        self.profit_eval_window_management = 30 
        self.bankruptcy_cash_threshold = -1000.0 
        self.bankruptcy_avg_profit_threshold_abs = -500.0 
        self.min_firms_per_sector = 4
        self.max_firms_per_sector = 25 

        self.new_firm_start_money_base = 7500.0 
        self.new_firm_start_capital_base = 150.0 
        self.new_firm_start_inventory_base = 250.0 

        self.sector_exit_profit_rate_threshold_factor = 0.25 
        self.sector_entry_profit_rate_threshold_factor = 1.05 

    def get_firms_in_sector(self, sector_id): return [f for f in self.firms if f.sector_id == sector_id]
    def update_firms_map(self): self.firms_map = {f.id: f for f in self.firms}

    def manage_firm_entry_exit_bankruptcy(self):
        bankrupted_firms_ids = []
        for firm in list(self.firms): 
            avg_profit_for_bankruptcy = firm.get_average_profit(self.profit_eval_window_management)
            is_mature_enough_for_profit_check = (self.current_period - firm.creation_period) >= self.profit_eval_window_management

            if (firm.monetary_holdings < self.bankruptcy_cash_threshold or
               (is_mature_enough_for_profit_check and avg_profit_for_bankruptcy < self.bankruptcy_avg_profit_threshold_abs)):
                bankrupted_firms_ids.append(firm.id)
                if firm in self.firms: self.firms.remove(firm)
        self.update_firms_map()

        for sector_idx, sector_obj in enumerate(self.sectors):
            if sector_idx >= self.num_sectors: continue # Safety check
            firms_in_this_sector = self.get_firms_in_sector(sector_idx)
            current_num_firms = len(firms_in_this_sector)
            market_prices = self.financial_system.market_prices_snapshot 
            
            sector_avg_profit_rate = sector_obj.calculate_average_sector_profit_rate(
                firms_in_this_sector, self.profit_eval_window_management, market_prices
            )
            threshold_rate_benchmark = self.financial_system.interest_rate_debt + sector_obj.risk_premium

            # Sector Exit
            # Only exit if sector is mature and average profit rate is low
            sector_is_mature_enough_for_exit = firms_in_this_sector and \
                all((self.current_period - f.creation_period) > self.profit_eval_window_management for f in firms_in_this_sector)


            if (current_num_firms > self.min_firms_per_sector and
                sector_avg_profit_rate < self.sector_exit_profit_rate_threshold_factor * threshold_rate_benchmark and
                sector_is_mature_enough_for_exit):
                
                firms_in_this_sector_with_rate = []
                for f_exit_candidate in firms_in_this_sector:
                     firm_capital_value = sum(f_exit_candidate.fixed_capital[i] * market_prices.get(i,1.0) for i in range(f_exit_candidate.num_goods))
                     avg_profit = f_exit_candidate.get_average_profit(self.profit_eval_window_management)
                     profit_rate = avg_profit / (firm_capital_value + 1e-6) if firm_capital_value > 1e-6 else -float('inf')
                     firms_in_this_sector_with_rate.append((profit_rate, f_exit_candidate))

                firms_in_this_sector_with_rate.sort(key=lambda x: x[0]) 

                if firms_in_this_sector_with_rate: # Remove the least profitable
                    firm_to_remove = firms_in_this_sector_with_rate[0][1] 
                    if firm_to_remove.id not in bankrupted_firms_ids and firm_to_remove in self.firms:
                         self.firms.remove(firm_to_remove); current_num_firms -= 1
                         self.update_firms_map()


            # Sector Entry
            if (self.households and current_num_firms < self.max_firms_per_sector and
                sector_avg_profit_rate > self.sector_entry_profit_rate_threshold_factor * threshold_rate_benchmark):

                successful_firms_in_sector = []
                if firms_in_this_sector: 
                    for f_obj in firms_in_this_sector:
                        firm_capital_value = sum(f_obj.fixed_capital[i] * market_prices.get(i,1.0) for i in range(f_obj.num_goods))
                        avg_profit = f_obj.get_average_profit(max(1, self.profit_eval_window_management // 2)) # Shorter window for template
                        profit_rate = avg_profit / (firm_capital_value + 1e-6) if firm_capital_value > 1e-6 else -float('inf')
                        if profit_rate > threshold_rate_benchmark * 0.8: # "Successful enough"
                             successful_firms_in_sector.append(f_obj)
                    if not successful_firms_in_sector: 
                        successful_firms_in_sector = firms_in_this_sector # Fallback: copy any existing


                new_owner_id = random.choice(self.households).id if self.households else -1
                new_firm = Firm(id=self.next_firm_id, sector_id=sector_idx, num_goods=self.num_goods,
                                owner_household_id=new_owner_id, creation_period=self.current_period)
                self.next_firm_id += 1
                
                # Initialize new firm with substantial resources
                new_firm.monetary_holdings = self.new_firm_start_money_base  * (0.8 + random.random()*0.4)
                new_firm.fixed_capital = np.ones(self.num_goods) * self.new_firm_start_capital_base * (0.8 + random.random()*0.4)
                if new_firm.sector_id < self.num_goods: # If it's a goods-producing sector
                    new_firm.inventory[new_firm.sector_id] = self.new_firm_start_inventory_base * (0.8 + random.random()*0.4)
                
                all_potential_suppliers_list = [f_map_item for f_map_item in self.firms_map.values() if f_map_item.id != new_firm.id]
                if all_potential_suppliers_list: 
                    for g_id_new_firm in range(self.num_goods):
                        new_firm.find_initial_suppliers(all_potential_suppliers_list, g_id_new_firm, self.num_goods, own_id_to_exclude=new_firm.id)

                if successful_firms_in_sector: # Inherit technology and pricing from template
                    template_firm = random.choice(successful_firms_in_sector)
                    new_firm.intermediary_input_coeffs = np.maximum(0.0, template_firm.intermediary_input_coeffs.copy() * (0.8 + random.random() * 0.4))
                    new_firm.fixed_capital_coeffs = np.maximum(0.001, template_firm.fixed_capital_coeffs.copy() * (0.8 + random.random() * 0.4))
                    new_firm.labor_coeff = max(0.01, template_firm.labor_coeff * (0.8 + random.random() * 0.4))
                    new_firm.price = max(0.1, template_firm.price * (0.9 + random.random() * 0.2))
                    new_firm.markup = max(0.01, template_firm.markup * (0.8 + random.random() * 0.4))
                else: # Fallback if no template (e.g. first firm in sector) - use Firm defaults
                     pass 

                self.firms.append(new_firm); current_num_firms += 1
                self.update_firms_map() 

            if sector_idx < self.num_sectors :
                self.sectors[sector_idx].num_active_firms = current_num_firms
        self.update_firms_map()

    def run_period(self):
        # === RESTRUCTURED PERIOD ACCORDING TO MANDEL (2012) ===

        # --- 1. Preparatory step ---
        for firm in self.firms:
            firm.costs_last_period = firm.costs_current_period
            firm.sales_last_period_volume = firm.sales_revenue / firm.price if firm.price > 1e-3 and firm.sales_revenue > 0 else 0.0
            firm.production_volume_last_period = firm.actual_production # Store for firm's own use (e.g. capital util calc)
            firm.sales_volume_history.append(firm.sales_last_period_volume)
            
            firm.costs_current_period = 0.0; firm.sales_revenue = 0.0 
            firm.actual_production = 0.0; firm.profit_current_period = 0.0

            firm.labor_employed_history.append(firm.labor_employed)
            if len(firm.labor_employed_history) > firm.wage_index_evolution_window + 2:
                firm.labor_employed_history.pop(0)
        
        for hh in self.households:
            hh.income = 0.0; hh.net_income_current_period = 0.0; hh.wage_received_this_period = 0.0

        # --- 2. Trading process ---
        # --- 2.a Dynamic Supplier Network Update ---
        if self.firms: 
            for hh_buyer in self.households:
                for good_id in range(self.num_goods):
                    hh_buyer.attempt_to_update_supplier_network(good_id, self.firms, self.firms_map)
            for firm_as_buyer in self.firms:
                for good_id in range(self.num_goods):
                    firm_as_buyer.attempt_to_update_supplier_network(good_id, self.firms, self.firms_map, own_id_to_exclude=firm_as_buyer.id)

        # --- 2.b Firms buy CAPITAL GOODS ---
        current_total_investment_value_this_period = 0 
        if self.firms: 
            def get_shuffled_buyers_firms(): shuffled_list = list(self.firms); random.shuffle(shuffled_list); return shuffled_list

            for firm_buyer in get_shuffled_buyers_firms():
                desired_investment_units_vector = firm_buyer.plan_investment(self.financial_system) 

                for capital_good_id in range(self.num_goods):
                    if desired_investment_units_vector[capital_good_id] > 1e-3:
                        units_to_buy_capital_planned = desired_investment_units_vector[capital_good_id]
                        supplier_firm_capital = firm_buyer.choose_supplier_from_known(capital_good_id, self.firms_map, units_to_buy_capital_planned)
                        
                        if not supplier_firm_capital: 
                            all_f_prod_cap = [f for f in self.firms if f.sector_id == capital_good_id and (capital_good_id < len(f.inventory) and f.inventory[capital_good_id] > 1e-3) and f.id != firm_buyer.id]
                            if all_f_prod_cap:
                                sample_fb_cap = random.sample(all_f_prod_cap, k=min(len(all_f_prod_cap), firm_buyer.num_suppliers_to_evaluate_on_trade))
                                if sample_fb_cap: sample_fb_cap.sort(key=lambda f_obj:f_obj.price); supplier_firm_capital = sample_fb_cap[0]
                        if not supplier_firm_capital: continue

                        seller_price_cap = supplier_firm_capital.price
                        if seller_price_cap < 1e-3: continue

                        units_can_buy_from_supplier = supplier_firm_capital.inventory[capital_good_id] if capital_good_id < len(supplier_firm_capital.inventory) else 0
                        actual_units_to_buy_capital = min(units_to_buy_capital_planned, units_can_buy_from_supplier) # No monetary constraint for firms here
                        actual_units_to_buy_capital = max(0, actual_units_to_buy_capital)

                        if actual_units_to_buy_capital > 1e-3:
                            cost_of_capital_purchase = actual_units_to_buy_capital * seller_price_cap
                            firm_buyer.monetary_holdings -= cost_of_capital_purchase 
                            firm_buyer.fixed_capital[capital_good_id] += actual_units_to_buy_capital
                            current_total_investment_value_this_period += cost_of_capital_purchase
                            
                            supplier_firm_capital.monetary_holdings += cost_of_capital_purchase
                            if capital_good_id < len(supplier_firm_capital.inventory):
                                supplier_firm_capital.inventory[capital_good_id] -= actual_units_to_buy_capital
                            supplier_firm_capital.sales_revenue += cost_of_capital_purchase
        self.history_total_investment_value.append(current_total_investment_value_this_period)

        # --- 2.c Firms buy INTERMEDIARY GOODS ---
        if self.firms:
            for firm_buyer in get_shuffled_buyers_firms(): 
                for input_good_id in range(self.num_goods):
                    if firm_buyer.intermediary_input_coeffs[input_good_id] > 1e-6:
                        needed_input_units_for_target_prod = firm_buyer.target_production * firm_buyer.intermediary_input_coeffs[input_good_id]
                        current_input_stock = firm_buyer.intermediary_inputs[input_good_id]
                        shortfall_units_to_buy = needed_input_units_for_target_prod - current_input_stock

                        if shortfall_units_to_buy > 1e-3:
                            supplier_firm_input = firm_buyer.choose_supplier_from_known(input_good_id, self.firms_map, shortfall_units_to_buy)
                            if not supplier_firm_input: 
                                all_f_prod_in = [f for f in self.firms if f.sector_id == input_good_id and (input_good_id < len(f.inventory) and f.inventory[input_good_id] > 1e-3) and f.id != firm_buyer.id]
                                if all_f_prod_in:
                                    sample_fb_in = random.sample(all_f_prod_in, k=min(len(all_f_prod_in), firm_buyer.num_suppliers_to_evaluate_on_trade))
                                    if sample_fb_in: sample_fb_in.sort(key=lambda f_obj:f_obj.price); supplier_firm_input = sample_fb_in[0]
                            if not supplier_firm_input: continue
                            
                            seller_price_in = supplier_firm_input.price
                            if seller_price_in < 1e-3: continue

                            units_can_buy_from_supplier_input = supplier_firm_input.inventory[input_good_id] if input_good_id < len(supplier_firm_input.inventory) else 0
                            actual_units_to_buy_input = min(shortfall_units_to_buy, units_can_buy_from_supplier_input) # No monetary constraint
                            actual_units_to_buy_input = max(0, actual_units_to_buy_input)

                            if actual_units_to_buy_input > 1e-3:
                                cost_of_input = actual_units_to_buy_input * seller_price_in
                                firm_buyer.monetary_holdings -= cost_of_input 
                                firm_buyer.intermediary_inputs[input_good_id] += actual_units_to_buy_input
                                firm_buyer.costs_current_period += cost_of_input 
                                
                                supplier_firm_input.monetary_holdings += cost_of_input
                                if input_good_id < len(supplier_firm_input.inventory):
                                    supplier_firm_input.inventory[input_good_id] -= actual_units_to_buy_input
                                supplier_firm_input.sales_revenue += cost_of_input
        
        # --- 2.d Households buy CONSUMPTION GOODS ---
        if self.households and self.firms: 
            def get_shuffled_buyers_households(): shuffled_list = list(self.households); random.shuffle(shuffled_list); return shuffled_list
            
            # For HH consumption, use current firm prices (FS snapshot should be up-to-date or updated before this)
            # If trading (step 2) is first, FS snapshot needs an initial update.
            if self.current_period == 0: self.financial_system.update_market_prices_snapshot(self.firms)
            market_prices_for_hh_decision = self.financial_system.market_prices_snapshot

            for hh_buyer in get_shuffled_buyers_households():
                money_for_consumption = hh_buyer.monetary_holdings 
                money_for_consumption = max(0, money_for_consumption)
                if money_for_consumption <= 1e-3 or self.num_goods == 0: continue
                
                initial_period_budget_for_hh = money_for_consumption 
                goods_to_buy_shuffled = list(range(self.num_goods)); random.shuffle(goods_to_buy_shuffled)

                for good_id in goods_to_buy_shuffled:
                    if hh_buyer.monetary_holdings <= 1e-3: break 
                    
                    budget_share_for_this_good = hh_buyer.consumption_technology[good_id] if self.num_goods > 0 and good_id < len(hh_buyer.consumption_technology) else 0
                    budget_for_this_good_type = initial_period_budget_for_hh * budget_share_for_this_good
                    if budget_for_this_good_type <= 1e-3: continue

                    estimated_price_for_demand_calc = market_prices_for_hh_decision.get(good_id, 1.0)
                    if estimated_price_for_demand_calc < 1e-3: estimated_price_for_demand_calc = 1.0
                    
                    units_demanded_approx = budget_for_this_good_type / estimated_price_for_demand_calc
                    supplier_firm = hh_buyer.choose_supplier_from_known(good_id, self.firms_map, units_demanded_approx)
                    if not supplier_firm: 
                        all_producing_good = [f for f in self.firms if f.sector_id == good_id and (good_id < len(f.inventory) and f.inventory[good_id] > 1e-3)]
                        if all_producing_good:
                             sample_fb = random.sample(all_producing_good, k=min(len(all_producing_good), hh_buyer.num_suppliers_to_evaluate_on_trade))
                             if sample_fb: sample_fb.sort(key=lambda f_obj:f_obj.price); supplier_firm = sample_fb[0]
                    if not supplier_firm: continue
                    
                    seller_price = supplier_firm.price
                    if seller_price < 1e-3: continue

                    units_can_afford_this_good_type = budget_for_this_good_type / seller_price
                    units_can_afford_overall = hh_buyer.monetary_holdings / seller_price
                    supplier_inventory_val = supplier_firm.inventory[good_id] if good_id < len(supplier_firm.inventory) else 0
                    
                    units_to_buy = min(units_can_afford_this_good_type, units_can_afford_overall, supplier_inventory_val)
                    units_to_buy = max(0, units_to_buy)
                    actual_cost_of_purchase = units_to_buy * seller_price

                    if actual_cost_of_purchase < 1e-6 and units_to_buy > 1e-3: units_to_buy = 0; actual_cost_of_purchase = 0
                    
                    if units_to_buy > 1e-3 and hh_buyer.monetary_holdings >= actual_cost_of_purchase - 1e-6 :
                        hh_buyer.monetary_holdings -= actual_cost_of_purchase
                        hh_buyer.goods_stock[good_id] += units_to_buy
                        
                        supplier_firm.monetary_holdings += actual_cost_of_purchase
                        if good_id < len(supplier_firm.inventory): supplier_firm.inventory[good_id] -= units_to_buy
                        supplier_firm.sales_revenue += actual_cost_of_purchase
        
        # --- 3. Labor market ---
        total_labor_supply_units = len(self.households) * 1.0
        for hh_lab in self.households: hh_lab.employed_by_firm_id = None
        for firm_lab in self.firms: firm_lab.employees = {}; firm_lab.labor_employed = 0.0

        available_households_for_hire = list(self.households); random.shuffle(available_households_for_hire)
        sorted_firms_for_hiring = sorted(self.firms, key=lambda f: f.reservation_wage_index * (self.sectors[f.sector_id].benchmark_wage if f.sector_id < len(self.sectors) else 10.0), reverse=True)

        for firm_hiring in sorted_firms_for_hiring:
            needed_labor_units = firm_hiring.target_employment # From prev. period's expectation update
            current_available_iter = list(available_households_for_hire) 
            next_round_available_households = [] 

            for hh_to_consider in current_available_iter:
                if needed_labor_units - firm_hiring.labor_employed <= 1e-3: 
                    next_round_available_households.append(hh_to_consider); continue

                sector_benchmark_wage = self.sectors[firm_hiring.sector_id].benchmark_wage if firm_hiring.sector_id < len(self.sectors) else 10.0
                offered_wage_per_unit_labor = firm_hiring.reservation_wage_index * sector_benchmark_wage
                # HH fallback can be based on average economy wage, or sector benchmark. Using sector benchmark for consistency.
                household_fallback_wage = hh_to_consider.wage_fallback * sector_benchmark_wage 

                if offered_wage_per_unit_labor >= household_fallback_wage:
                    labor_to_hire_from_hh = min(needed_labor_units - firm_hiring.labor_employed, hh_to_consider.labor_supply)
                    firm_hiring.employees[hh_to_consider.id] = labor_to_hire_from_hh
                    firm_hiring.labor_employed += labor_to_hire_from_hh
                    hh_to_consider.employed_by_firm_id = firm_hiring.id
                    hh_to_consider.wage_received_this_period = offered_wage_per_unit_labor * labor_to_hire_from_hh 
                else: 
                    next_round_available_households.append(hh_to_consider)
            available_households_for_hire = next_round_available_households

        num_employed_households = sum(1 for hh_stats in self.households if hh_stats.employed_by_firm_id is not None)
        current_unemployment_rate = (total_labor_supply_units - num_employed_households) / total_labor_supply_units if total_labor_supply_units > 0 else 0.0
        self.history_unemployment_rate.append(current_unemployment_rate)

        # --- 4. Production and consumption ---
        gdp_this_period = 0
        for firm_prod in self.firms:
            prod_sector_idx = firm_prod.sector_id
            prod_sector_obj = self.sectors[prod_sector_idx] if prod_sector_idx < len(self.sectors) else None
            prod_labor_productivity = prod_sector_obj.labor_productivity_index if prod_sector_obj else 1.0
            
            production_value = firm_prod.produce(prod_labor_productivity) * firm_prod.price
            gdp_this_period += production_value
        self.history_gdp.append(gdp_this_period)
        
        for hh_cons in self.households: hh_cons.consume_goods() 

        # --- 5. Accounting ---
        households_map = {h.id: h for h in self.households} 
        
        for firm_acc in self.firms: # Pay wages
             total_wage_cost_firm = 0
             for hh_id, labor_amount in list(firm_acc.employees.items()): 
                 hh_obj = households_map.get(hh_id)
                 if hh_obj:
                     wage_payment_for_hh = hh_obj.wage_received_this_period 
                     firm_acc.monetary_holdings -= wage_payment_for_hh 
                     hh_obj.monetary_holdings += wage_payment_for_hh   
                     hh_obj.income += wage_payment_for_hh             
                     total_wage_cost_firm += wage_payment_for_hh
                 else: 
                     del firm_acc.employees[hh_id] 
             firm_acc.costs_current_period += total_wage_cost_firm

        self.financial_system.update_market_prices_snapshot(self.firms) # Update for accounting (e.g. depreciation value)
        market_prices_for_accounting = self.financial_system.market_prices_snapshot

        for firm_acc in self.firms: # Firms' internal accounting
            firm_acc.accounting(households_map, self.financial_system.interest_rate_debt, market_prices_for_accounting)

        for firm_acc in self.firms: # Pay dividends
            if firm_acc.profit_current_period > 0 and firm_acc.monetary_holdings > 0:
                available_for_dividends = min(firm_acc.profit_current_period, firm_acc.monetary_holdings)
                dividends_to_pay = available_for_dividends * firm_acc.dividend_rate
                if dividends_to_pay > 1e-6:
                     firm_acc.monetary_holdings -= dividends_to_pay
                     owner_hh = households_map.get(firm_acc.owner_household_id)
                     if owner_hh:
                         owner_hh.monetary_holdings += dividends_to_pay
                         owner_hh.income += dividends_to_pay 

        for hh_acc in self.households: # HHs receive interest on savings
            hh_acc.allocate_income_to_savings_and_consumption_budget(self.financial_system.interest_rate_deposits, stage="interest_only")
            
        total_wages_paid_to_hh_this_period = sum(hh_calc.wage_received_this_period for hh_calc in self.households)
        avg_wage_economy = total_wages_paid_to_hh_this_period / num_employed_households if num_employed_households > 0 else \
                           (self.sectors[0].benchmark_wage if self.sectors and self.sectors[0] else 10.0)
        self.government.set_tax_rate_and_collect_taxes(self.households, avg_wage_economy)
        
        # --- 6. Expectations updating ---
        for firm_exp in self.firms: # Firms update sales expectations & target production for NEXT period
            firm_exp.update_expectations_and_target_production() 
        
        for hh_exp in self.households: # HHs update income expectations & decide savings (Deaton's rule)
            hh_exp.update_expectations() 
            hh_exp.allocate_income_to_savings_and_consumption_budget(self.financial_system.interest_rate_deposits, stage="deaton_rule")

        # --- 7. Labor productivity updating ---
        if self.firms and self.sectors:
            current_sector_productivities_log = []
            for sector_idx, sector_obj in enumerate(self.sectors):
                if sector_idx >= self.num_sectors: continue
                firms_in_s = self.get_firms_in_sector(sector_idx)
                sector_obj.update_labor_productivity(firms_in_s, market_prices_for_accounting) # Uses K_t and K_{t-1} logic
                current_sector_productivities_log.append(sector_obj.labor_productivity_index)
            
            if current_sector_productivities_log : # Ensure not empty
                 self.history_sector_productivity.append(current_sector_productivities_log if self.num_sectors > 1 else current_sector_productivities_log[0])
        else: 
            no_firm_prod_log = ([s.labor_productivity_index for s in self.sectors] if self.num_sectors > 1 else (self.sectors[0].labor_productivity_index if self.sectors else 0.0))
            self.history_sector_productivity.append(no_firm_prod_log)


        # --- 8. Price updating (Firms) ---
        for firm_price_update in self.firms:
            firm_price_update.update_price() 

        # --- 9. Interest rate updating (Financial System) ---
        avg_prices_list_for_inflation = [f_price.price for f_price in self.firms if f_price.price > 1e-3]
        current_avg_market_price_for_inflation = np.mean(avg_prices_list_for_inflation) if avg_prices_list_for_inflation else \
                                                 (self.history_avg_price[-1] if self.history_avg_price else 1.0)
        self.history_avg_price.append(current_avg_market_price_for_inflation)
        
        prev_avg_market_price_for_inflation = self.history_avg_price[-2] if len(self.history_avg_price) > 1 else current_avg_market_price_for_inflation
        inflation_rate_current = (current_avg_market_price_for_inflation / prev_avg_market_price_for_inflation - 1.0) if prev_avg_market_price_for_inflation > 1e-6 else 0.0
        self.history_inflation.append(inflation_rate_current)
        
        self.financial_system.update_interest_rate_taylor_rule(current_unemployment_rate, inflation_rate_current)

        # --- 10. Firms entry, exit and bankruptcy (Periodic) ---
        if self.current_period > 0 and self.current_period % self.firm_management_periodicity == 0:
            self.manage_firm_entry_exit_bankruptcy() 
        self.history_num_firms.append(len(self.firms))


        # --- 11-14. Genetic evolutions (Periodic) ---
        if self.current_period > 0 and self.current_period % self.genetic_evolution_periodicity == 0:
             prices_for_genetic_calc = self.financial_system.market_prices_snapshot # Use latest snapshot
             for sector_idx_gen, sector_obj_gen in enumerate(self.sectors):
                 if sector_idx_gen >= self.num_sectors: continue
                 firms_in_sector_gen = self.get_firms_in_sector(sector_idx_gen)
                 if len(firms_in_sector_gen) > 1: 
                     peers_map = {firm_g.id: [p for p in firms_in_sector_gen if p.id != firm_g.id] for firm_g in firms_in_sector_gen}
                     for firm_g_evolve in firms_in_sector_gen:
                         peers_for_firm = peers_map.get(firm_g_evolve.id, [])
                         firm_g_evolve.evolve_technology(peers_for_firm, sector_obj_gen.benchmark_wage, prices_for_genetic_calc)
                         firm_g_evolve.evolve_markup(peers_for_firm)
                         firm_g_evolve.evolve_wage_index(peers_for_firm)
        
        # --- Logging ---
        if self.current_period % 20 == 0:
            num_firms_log = len(self.firms)
            gdp_log = gdp_this_period
            unemployment_log = current_unemployment_rate
            avg_price_log = current_avg_market_price_for_inflation 
            inflation_log = inflation_rate_current
            investment_log = current_total_investment_value_this_period
            print(f"P{self.current_period}: GDP={gdp_log:.1f}, U={unemployment_log:.1%}, AvgP={avg_price_log:.2f}, Inf={inflation_log:.1%}, Firms={num_firms_log}, Inv={investment_log:.1f}")
        
        self.current_period += 1

    def run_simulation(self, plot_results=True):
        print(f"Starting simulation for {self.max_periods} periods...")
        # Initialize K_{t-1} for sectors based on initial capital of firms
        if self.current_period == 0: # Only on the very first call
            for s_idx, sector_obj_init_k in enumerate(self.sectors):
                if s_idx >= self.num_sectors: continue
                firms_in_s_init_k = self.get_firms_in_sector(s_idx)
                initial_k_value_sector = 0
                for f_init_k in firms_in_s_init_k:
                    initial_k_value_sector += sum(f_init_k.fixed_capital[i] * 1.0 for i in range(self.num_goods)) # Assume price 1.0 at t=0
                sector_obj_init_k.capital_stock_value_end_of_last_period = initial_k_value_sector
                # print(f"Sector {s_idx} initial K_val for K_t-1: {initial_k_value_sector}")


        for i in range(self.max_periods):
            if not self.firms:
                print(f"Simulation stopped at period {self.current_period}: no active firms.");
                # Fill remaining history 
                remaining_periods = self.max_periods - self.current_period
                if remaining_periods > 0:
                    self.history_gdp.extend([self.history_gdp[-1] if self.history_gdp else 0] * remaining_periods)
                    self.history_total_investment_value.extend([self.history_total_investment_value[-1] if self.history_total_investment_value else 0] * remaining_periods)
                    self.history_unemployment_rate.extend([1.0] * remaining_periods) # Assume 100% U if no firms
                    avg_price_to_fill = self.history_avg_price[-1] if self.history_avg_price else 1.0
                    self.history_avg_price.extend([avg_price_to_fill] * remaining_periods)
                    self.history_inflation.extend([self.history_inflation[-1] if self.history_inflation else 0] * remaining_periods)
                    
                    prod_to_fill_hist = []
                    if self.history_sector_productivity:
                        last_prod_val = self.history_sector_productivity[-1]
                        prod_to_fill_hist = last_prod_val # Already in correct list/float format
                    else: 
                        prod_to_fill_hist = ([0.0] * self.num_sectors if self.num_sectors > 1 else 0.0)
                    self.history_sector_productivity.extend([prod_to_fill_hist] * remaining_periods)
                    
                    self.history_num_firms.extend([0] * remaining_periods)
                    self.current_period += remaining_periods # Ensure current_period reaches max_periods for plotting
                break
            try: self.run_period()
            except Exception as e:
                print(f"Error during period {self.current_period}: {e}"); traceback.print_exc();
                # Similar fill logic on error
                remaining_periods = self.max_periods - self.current_period
                if remaining_periods > 0:
                    self.history_gdp.extend([self.history_gdp[-1] if self.history_gdp else 0] * remaining_periods)
                    self.history_total_investment_value.extend([self.history_total_investment_value[-1] if self.history_total_investment_value else 0] * remaining_periods)
                    self.history_unemployment_rate.extend([self.history_unemployment_rate[-1] if self.history_unemployment_rate else 1.0] * remaining_periods)
                    self.history_avg_price.extend([self.history_avg_price[-1] if self.history_avg_price else 1.0] * remaining_periods)
                    self.history_inflation.extend([self.history_inflation[-1] if self.history_inflation else 0] * remaining_periods)
                    
                    prod_to_fill_err_hist = []
                    if self.history_sector_productivity:
                        last_prod_val_err = self.history_sector_productivity[-1]
                        prod_to_fill_err_hist = last_prod_val_err
                    else: 
                        prod_to_fill_err_hist = ([0.0] * self.num_sectors if self.num_sectors > 1 else 0.0)
                    self.history_sector_productivity.extend([prod_to_fill_err_hist] * remaining_periods)
                    
                    self.history_num_firms.extend([self.history_num_firms[-1] if self.history_num_firms else 0] * remaining_periods)
                    self.current_period += remaining_periods
                break

        actual_periods_for_plot = self.current_period 
        history_attributes = ['history_gdp', 'history_unemployment_rate', 'history_avg_price', 
                              'history_inflation', 'history_total_investment_value', 
                              'history_sector_productivity', 'history_num_firms']
        for attr_name in history_attributes: # Ensure all history lists are of same length for plotting
            hist_list = getattr(self, attr_name)
            if len(hist_list) < actual_periods_for_plot : 
                pad_len = actual_periods_for_plot - len(hist_list)
                if pad_len > 0:
                    last_val_pad = hist_list[-1] if hist_list else 0 
                    if attr_name == 'history_unemployment_rate': last_val_pad = hist_list[-1] if hist_list else 1.0
                    elif attr_name == 'history_avg_price': last_val_pad = hist_list[-1] if hist_list else 1.0
                    elif attr_name == 'history_sector_productivity':
                        if hist_list: last_val_pad = hist_list[-1]
                        else: last_val_pad = ([0.0] * self.num_sectors if self.num_sectors > 1 else 0.0)
                    hist_list.extend([last_val_pad] * pad_len)
            setattr(self, attr_name, hist_list[:actual_periods_for_plot]) # Trim if somehow longer

        print(f"\n--- Simulation Ended at Period {self.current_period} ---")
        if plot_results: self.plot_results()

    def plot_results(self): # Plotting logic remains the same as provided by you
        actual_periods_run = self.current_period
        if actual_periods_run == 0: print("No periods were run, cannot plot results."); return
        periods = range(actual_periods_run); num_plots = 7 
        fig, axs = plt.subplots(num_plots, 1, figsize=(12, num_plots * 3.5), sharex=True); plot_idx = 0
        
        if self.history_gdp and len(self.history_gdp) == actual_periods_run:
            axs[plot_idx].plot(periods, self.history_gdp, marker='.', linestyle='-', markersize=3)
            axs[plot_idx].set_title('GDP Over Time'); axs[plot_idx].set_ylabel('GDP')
            positive_gdp = [v for v in self.history_gdp if v is not None and v > 1e-9] 
            if positive_gdp and max(positive_gdp, default=0) > 10 and min(positive_gdp, default=1e-9) > 0:
                try: axs[plot_idx].set_yscale('log')
                except ValueError: axs[plot_idx].set_yscale('linear') 
            else: axs[plot_idx].set_yscale('linear')
            axs[plot_idx].grid(True)
        plot_idx += 1

        if self.history_total_investment_value and len(self.history_total_investment_value) == actual_periods_run:
            axs[plot_idx].plot(periods, self.history_total_investment_value, marker='.', linestyle='-', markersize=2, alpha=0.7, label="Investment Value")
            window_size = 10
            valid_investment_data = [val for val in self.history_total_investment_value if val is not None and isinstance(val, (int, float))]
            if len(valid_investment_data) >= window_size:
                invest_ma = np.convolve(valid_investment_data, np.ones(window_size)/window_size, mode='valid')
                ma_periods = periods[window_size - 1 : len(valid_investment_data)] 
                axs[plot_idx].plot(ma_periods[:len(invest_ma)], invest_ma, linestyle='-', label=f"MA({window_size})", color='red')

            axs[plot_idx].set_title('Total Investment Value Over Time'); axs[plot_idx].set_ylabel('Investment Value')
            min_inv_val = min(valid_investment_data, default=0) if valid_investment_data else 0
            max_inv_val = max(valid_investment_data, default=0) if valid_investment_data else 0
            if max_inv_val > 10 and min_inv_val >= 0 : 
                 try: axs[plot_idx].set_yscale('log')
                 except ValueError: axs[plot_idx].set_yscale('linear')
            else:
                axs[plot_idx].set_yscale('linear')
                if min_inv_val < 0: axs[plot_idx].axhline(0, color='black', linestyle='--', linewidth=0.7)
            axs[plot_idx].grid(True); axs[plot_idx].legend()
        plot_idx += 1

        if self.history_unemployment_rate and len(self.history_unemployment_rate) == actual_periods_run:
            axs[plot_idx].plot(periods, [r * 100 for r in self.history_unemployment_rate], marker='.', linestyle='-', markersize=3)
            axs[plot_idx].set_title('Unemployment Rate Over Time'); axs[plot_idx].set_ylabel('Unemployment (%)')
            axs[plot_idx].grid(True); axs[plot_idx].set_ylim(0, 100)
        plot_idx += 1
        
        if self.history_avg_price and len(self.history_avg_price) == actual_periods_run:
            axs[plot_idx].plot(periods, self.history_avg_price, marker='.', linestyle='-', markersize=3, color='blue', label='Avg Price')
            axs[plot_idx].set_title('Avg Price & Inflation Rate'); axs[plot_idx].set_ylabel('Avg Price', color='blue')
            positive_prices = [v for v in self.history_avg_price if v is not None and v > 1e-9]
            if positive_prices and max(positive_prices, default=0) > 10 and min(positive_prices, default=1e-9) > 0:
                try: axs[plot_idx].set_yscale('log')
                except ValueError: axs[plot_idx].set_yscale('linear')
            else: axs[plot_idx].set_yscale('linear')
            axs[plot_idx].tick_params(axis='y', labelcolor='blue'); axs[plot_idx].grid(True, axis='y', linestyle='--', alpha=0.7)
            
            ax_twin = axs[plot_idx].twinx()
            if self.history_inflation and len(self.history_inflation) == actual_periods_run:
                ax_twin.plot(periods, [r * 100 for r in self.history_inflation], marker='.', linestyle='-', markersize=2, color='red', alpha=0.6, label='Inflation (%)')
                ax_twin.set_ylabel('Inflation (%)', color='red'); ax_twin.tick_params(axis='y', labelcolor='red')
                ax_twin.hlines(0, 0, actual_periods_run, colors='red', linestyles=':', lw=1)
            lines, labels = axs[plot_idx].get_legend_handles_labels(); lines2, labels2 = ax_twin.get_legend_handles_labels()
            if lines or lines2: ax_twin.legend(lines + lines2, labels + labels2, loc='best')
        plot_idx += 1

        if self.history_sector_productivity and len(self.history_sector_productivity) == actual_periods_run:
            if self.num_sectors == 1:
                prod_data_single_sector = []
                for item in self.history_sector_productivity:
                    if isinstance(item, list) and item is not None and len(item) > 0: prod_data_single_sector.append(item[0])
                    elif isinstance(item, (int,float)) and item is not None : prod_data_single_sector.append(item)
                    else: prod_data_single_sector.append(None) 
                
                axs[plot_idx].plot(periods, prod_data_single_sector, marker='.', linestyle='-', markersize=3, label=f'Sector 0 Prod.')
            else: 
                for s_idx in range(self.num_sectors):
                    prod_data_sector = []
                    for item_period in self.history_sector_productivity:
                        if isinstance(item_period, list) and len(item_period) > s_idx and item_period[s_idx] is not None:
                            prod_data_sector.append(item_period[s_idx])
                        else: prod_data_sector.append(None) 
                    
                    valid_periods = [p for i, p in enumerate(periods) if prod_data_sector[i] is not None]
                    valid_prod_data = [d for d in prod_data_sector if d is not None]
                    if valid_prod_data:
                        axs[plot_idx].plot(valid_periods, valid_prod_data, marker='.', linestyle='-', markersize=2, alpha=0.8, label=f'Sector {s_idx} Prod.')
            
            axs[plot_idx].set_title('Labor Productivity Index'); axs[plot_idx].set_ylabel('Productivity Idx')
            axs[plot_idx].grid(True)
            if any(axs[plot_idx].get_lines()): axs[plot_idx].legend() 
        plot_idx += 1
        
        if self.history_num_firms and len(self.history_num_firms) == actual_periods_run:
            axs[plot_idx].plot(periods, self.history_num_firms, marker='.', linestyle='-', markersize=3)
            axs[plot_idx].set_title('Number of Active Firms'); axs[plot_idx].set_ylabel('Num Firms')
            axs[plot_idx].grid(True); axs[plot_idx].set_ylim(bottom=0) 
        plot_idx += 1
        
        if num_plots > 6 and plot_idx < num_plots: # Placeholder 7th plot
             axs[plot_idx].set_title('Financial System Interest Rates')
             # Example: Requires storing interest rate history in Simulation class
             # axs[plot_idx].plot(periods, self.history_debt_rate, label='Debt Rate')
             # axs[plot_idx].plot(periods, self.history_deposit_rate, label='Deposit Rate')
             axs[plot_idx].grid(True)
             if any(axs[plot_idx].get_lines()): axs[plot_idx].legend()

        plt.xlabel('Period'); plt.tight_layout(); plt.show()

if __name__ == '__main__':
    # Пример запуска для тестирования (односекторный)
    sim_single = Simulation(num_goods=1, 
                            num_households=10, # Было 200, уменьшил для скорости теста
                            num_firms_per_sector_list=[10], # Было 20
                            num_sectors=1)
    sim_single.max_periods = 300 # Было 600-1000
    sim_single.run_simulation(plot_results=True)
