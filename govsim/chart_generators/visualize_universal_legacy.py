# visualize.py
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List

DEFAULT_RESULTS_FILENAME = "simulation_results.json"
HIGH_ITERATION_THRESHOLD = 100

def load_results(filepath: Path) -> Optional[Dict[str, Any]]:
    """Загружает результаты симуляции из JSON файла."""
    print(f"Попытка загрузить результаты из: {filepath}")
    if not filepath.is_file():
        print(f"Ошибка: Файл результатов '{filepath}' не найден.")
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            results_data = json.load(f)
        print("Результаты успешно загружены.")
        return results_data
    except json.JSONDecodeError as e:
        print(f"Ошибка декодирования JSON: {e}")
        return None
    except Exception as e:
        print(f"Неожиданная ошибка при загрузке: {e}")
        return None

def get_metric(entry: Dict[str, Any], metric_name: str, default: Any = None) -> Any:
    """Безопасно извлекает метрику из записи лога."""
    return entry.get('metrics', {}).get(metric_name, default)

def downsample_data(*args, factor: int):
    """Уменьшает плотность данных, беря каждую factor-ую точку."""
    if factor <= 1:
        return args
    return tuple(arg[::factor] if arg is not None and isinstance(arg, list) and len(arg) >= factor else arg for arg in args)


def moving_average(data: List[Optional[float]], window_size: int) -> List[Optional[float]]:
    """Расчет скользящего среднего. Обрабатывает None значения."""
    if window_size <= 1 or not data:
        return data
    # Преобразуем в numpy массив, заменяя None на NaN
    np_data = np.array(data, dtype=float)
    if len(np_data) < window_size:
        return [None] * len(data)

    # Используем np.nanmean для игнорирования NaN (бывших None)
    # Это более сложная реализация скользящего среднего с окном, которое двигается
    # Но для простоты можно использовать convolve с фильтрацией NaN
    # Let's stick to a simple convolve for now, assuming None's are handled by filtering before MA
    # Or, a simple loop:
    result = []
    for i in range(len(data)):
        if i < window_size - 1:
            result.append(None)
        else:
            window = [x for x in data[i - window_size + 1 : i + 1] if x is not None]
            if window:
                result.append(np.mean(window))
            else:
                result.append(None)
    return result


def plot_linear_system_results(results_data: Dict[str, Any]):
    """Визуализирует результаты для модели LinearStochasticSystem, включая KPI."""
    if not results_data:
        print("Ошибка: Нет данных для визуализации LinearStochasticSystem.")
        return
    log = results_data.get('simulation_log')
    if not log:
        print("Ошибка: 'simulation_log' отсутствует или пуст для LinearStochasticSystem.")
        return

    config = results_data.get('config', {})
    agent_params = results_data.get('agent_params', {})
    num_steps_total = len(log)
    print(f"Данные лога LinearStochasticSystem получены ({num_steps_total} шагов). Начинаем построение...")

    # --- Параметры визуализации ---
    is_high_iteration_mode = num_steps_total > HIGH_ITERATION_THRESHOLD
    plot_kwargs = {'marker': None, 'linewidth': 1.2, 'alpha': 0.9} if is_high_iteration_mode else {'marker': '.', 'markersize': 4, 'linewidth': 1.5, 'alpha': 0.8}
    axis_label_fontsize = 9
    title_fontsize = 11
    legend_fontsize = 8

    downsample_factor = 1
    if num_steps_total > 1000: # Увеличил порог для downsampling x, u
        downsample_factor = 5
    if num_steps_total > 5000:
         downsample_factor = 10 # Еще больше для очень длинных симуляций

    # Окно для расчета KPI из agent_params
    # Если отсутствует, используем 20 по умолчанию, как в агенте
    performance_window = agent_params.get('performance_window', 20)
    if performance_window <= 0: performance_window = 1
    if performance_window > num_steps_total: performance_window = num_steps_total # Окно не может быть больше истории

    # Окно для скользящего среднего на графиках (может отличаться от окна KPI)
    plotting_ma_window = 1
    if num_steps_total > 200: # Применяем сглаживание для более длинных графиков
        plotting_ma_window = max(1, num_steps_total // 200) # Пример: окно = 0.5% от общей длины
    if plotting_ma_window > 20: plotting_ma_window = 20 # Ограничиваем максимальный размер окна для графика

    # --- Извлечение данных из лога ---
    steps_orig = [get_metric(entry, 'step') for entry in log]
    current_x_orig = [get_metric(entry, 'current_x') for entry in log]
    current_u_orig = [get_metric(entry, 'current_u') for entry in log]
    target_x_val = get_metric(log[0] if log else {}, 'target_x', None) # Цель

    # Фильтрация невалидных шагов (на всякий случай)
    valid_indices = [i for i, s in enumerate(steps_orig) if s is not None]
    if not valid_indices:
        print("Ошибка: Не найдено валидных шагов в логе LinearStochasticSystem.")
        return
    steps_full = [steps_orig[i] for i in valid_indices]
    current_x_full = [current_x_orig[i] for i in valid_indices]
    current_u_full = [current_u_orig[i] for i in valid_indices]

    if not steps_full:
         print("Ошибка: После фильтрации не осталось данных для шагов LinearStochasticSystem.")
         return

    # --- Расчет KPI (MSE, MSU) по полным данным ---
    mse_values_full = []
    msu_values_full = []
    kpi_steps_full = [] # Шаги, на которых рассчитаны KPI

    # Приводим x_full к numpy массиву для расчета MSE
    np_x_full = np.array(current_x_full, dtype=float)
    np_u_full = np.array(current_u_full, dtype=float)
    np_target_x = np.array(target_x_val, dtype=float) if target_x_val is not None else 0.0 # Цель как numpy float

    # Расчет скользящего KPI
    for k in range(performance_window - 1, num_steps_total):
        window_x = np_x_full[k - performance_window + 1 : k + 1]
        window_u = np_u_full[k - performance_window + 1 : k + 1]

        # Используем np.nanmean для расчета средних, игнорируя NaN (если вдруг появятся)
        mse_k = np.nanmean((window_x - np_target_x)**2) if len(window_x) > 0 else np.nan # MSE
        msu_k = np.nanmean(window_u**2) if len(window_u) > 0 else np.nan # MSU

        mse_values_full.append(mse_k)
        msu_values_full.append(msu_k)
        kpi_steps_full.append(steps_full[k]) # Шаг соответствует концу окна


    # --- Downsampling и MA для графиков ---
    # Downsampling для x и u (применяем к полным данным)
    (steps_plot_x_u, current_x_plot, current_u_plot) = downsample_data(
        steps_full, current_x_full, current_u_full, factor=downsample_factor
    )

    # Downsampling для KPI (применяем к рассчитанным KPI)
    # Шаги для KPI начинаются позже, чем для x и u
    (kpi_steps_plot, mse_values_plot, msu_values_plot) = downsample_data(
        kpi_steps_full, mse_values_full, msu_values_full, factor=downsample_factor
    )

    # Скользящее среднее для графиков
    current_x_plot_smooth = moving_average(current_x_plot if current_x_plot else [], plotting_ma_window)
    current_u_plot_smooth = moving_average(current_u_plot if current_u_plot else [], plotting_ma_window)
    mse_values_plot_smooth = moving_average(mse_values_plot if mse_values_plot else [], plotting_ma_window)
    msu_values_plot_smooth = moving_average(msu_values_plot if msu_values_plot else [], plotting_ma_window)


    # Шаги, когда политика была обновлена
    # Ищем в оригинальных шагах лога, но отмечаем на downsampled графике
    policy_decision_steps_log = [entry.get('step') for entry in log if entry.get('policy_decision') is not None]
    # Фильтруем эти шаги, чтобы они были в диапазоне downsampled X оси
    policy_decision_steps_plot = [s for s in policy_decision_steps_log if steps_plot_x_u and min(steps_plot_x_u) <= s <= max(steps_plot_x_u)]


    # --- Построение графиков ---
    # Создаем 2x1 сетку графиков
    fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f'Симуляция: {config.get("economic_model_type", "N/A")} / {config.get("government_agent_type", "N/A")} ({num_steps_total} шагов, окно KPI={performance_window})', fontsize=14)

    # --- Верхний график: Состояние (x_k) и Управление (u_k) ---
    ax1 = axs[0]
    color_x = 'tab:blue'
    ax1.set_ylabel('Состояние системы (x_k)', color=color_x, fontsize=axis_label_fontsize)
    line_x, = ax1.plot(steps_plot_x_u, current_x_plot_smooth, color=color_x, label='Состояние x_k (MA)', **plot_kwargs)
    ax1.tick_params(axis='y', labelcolor=color_x, labelsize=axis_label_fontsize-1)
    ax1.grid(True, linestyle=':', which='both')

    # График цели x*
    target_line = None
    if target_x_val is not None:
        target_line = ax1.axhline(y=target_x_val, color='grey', linestyle='--', linewidth=1.2, label=f'Цель x* = {target_x_val}')

    # График управления u_k (на второй оси Y для верхнего графика)
    ax1_twin = ax1.twinx()
    color_u = 'tab:red'
    ax1_twin.set_ylabel('Управление (u_k)', color=color_u, fontsize=axis_label_fontsize)
    # Обычно управление u_k не сглаживают, но можно добавить опцию
    line_u, = ax1_twin.plot(steps_plot_x_u, current_u_plot, color=color_u, linestyle='--', label='Управление u_k', **plot_kwargs) # Используем unsmoothed u_k
    ax1_twin.tick_params(axis='y', labelcolor=color_u, labelsize=axis_label_fontsize-1)

    # Вертикальные линии для решений агента на верхнем графике
    v_lines = []
    for s_dec in policy_decision_steps_plot:
         vline = ax1.axvline(x=s_dec, color='green', linestyle=':', alpha=0.5, linewidth=0.8)
         if not v_lines: v_lines.append(vline) # Добавляем только одну для легенды

    # Легенда для верхнего графика
    handles1 = [line_x, line_u]
    labels1 = [h.get_label() for h in handles1]
    if target_line:
        handles1.insert(0, target_line) # Добавляем цель в начало
        labels1.insert(0, target_line.get_label())
    if v_lines:
         handles1.append(v_lines[0])
         labels1.append('Обновление политики')

    ax1.legend(handles1, labels1, loc='best', fontsize=legend_fontsize)


    # --- Нижний график: KPI (MSE, MSU) ---
    ax2 = axs[1]
    ax2.set_xlabel('Шаг симуляции', fontsize=axis_label_fontsize)
    ax2.set_ylabel('MSE', color='tab:purple', fontsize=axis_label_fontsize)
    line_mse, = ax2.plot(kpi_steps_plot, mse_values_plot_smooth, color='tab:purple', label=f'MSE (окно={performance_window}, MA={plotting_ma_window})', **plot_kwargs)
    ax2.tick_params(axis='y', labelcolor='tab:purple', labelsize=axis_label_fontsize-1)
    ax2.grid(True, linestyle=':', which='both')

    # График MSU (на второй оси Y для нижнего графика)
    ax2_twin = ax2.twinx()
    color_msu = 'tab:orange'
    ax2_twin.set_ylabel('MSU', color=color_msu, fontsize=axis_label_fontsize)
    line_msu, = ax2_twin.plot(kpi_steps_plot, msu_values_plot_smooth, color=color_msu, linestyle='--', label=f'MSU (окно={performance_window}, MA={plotting_ma_window})', **plot_kwargs)
    ax2_twin.tick_params(axis='y', labelcolor=color_msu, labelsize=axis_label_fontsize-1)
    ax2_twin.set_ylim(bottom=0) # MSU >= 0

    # Вертикальные линии для решений агента на нижнем графике
    v_lines2 = []
    for s_dec in policy_decision_steps_plot:
         # Проверяем, попадает ли линия в диапазон X для KPI графика (начинается позже)
         if kpi_steps_plot and min(kpi_steps_plot) <= s_dec <= max(kpi_steps_plot):
              vline2 = ax2.axvline(x=s_dec, color='green', linestyle=':', alpha=0.5, linewidth=0.8)
              # Не добавляем в легенду, так как уже есть в верхнем графике

    # Легенда для нижнего графика
    handles2 = [line_mse, line_msu]
    labels2 = [h.get_label() for h in handles2]
    ax2.legend(handles2, labels2, loc='best', fontsize=legend_fontsize)


    # Общие настройки макета
    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Подгоняем макет, оставляя место для общего заголовка и легенды
    plt.show()


def plot_simple_growth_results(results_data: Dict[str, Any]):
    """Визуализирует результаты для модели SimpleGrowthModel."""
    if not results_data:
        print("Ошибка: Нет данных для визуализации SimpleGrowthModel.")
        return
    log = results_data.get('simulation_log')
    if not log:
        print("Ошибка: 'simulation_log' отсутствует или пуст для SimpleGrowthModel.")
        return

    config = results_data.get('config', {})
    num_steps_total = len(log)
    print(f"Данные лога SimpleGrowthModel получены ({num_steps_total} шагов). Начинаем построение...")

    is_high_iteration_mode = num_steps_total > HIGH_ITERATION_THRESHOLD
    plot_kwargs = {'marker': None, 'linewidth': 1.2, 'alpha': 0.9} if is_high_iteration_mode else {'marker': '.', 'markersize': 4, 'linewidth': 1.5, 'alpha': 0.8}
    axis_label_fontsize = 9
    title_fontsize = 11
    legend_fontsize = 8

    downsample_factor = 1
    if num_steps_total > 500:
        downsample_factor = 5

    # --- Извлечение данных ---
    steps_orig = [get_metric(entry, 'step') for entry in log]
    gdp_orig = [get_metric(entry, 'gdp') for entry in log]
    tax_rate_orig = [get_metric(entry, 'tax_rate') for entry in log]

    valid_indices = [i for i, s in enumerate(steps_orig) if s is not None]
    if not valid_indices:
        print("Ошибка: Не найдено валидных шагов в логе SimpleGrowthModel.")
        return

    def filter_by_valid_indices(*data_lists):
        return tuple([data_list[i] for i in valid_indices] if data_list and len(data_list) > max(valid_indices, default=-1) else [] for data_list in data_lists)

    (steps, gdp, tax_rate) = filter_by_valid_indices(
        steps_orig, gdp_orig, tax_rate_orig
    )

    if not steps:
        print("Ошибка: После фильтрации не осталось данных для шагов SimpleGrowthModel.")
        return

    (steps_plot, gdp_plot, tax_rate_plot) = downsample_data(
        steps, gdp, tax_rate, factor=downsample_factor
    )

    policy_decision_steps = [entry.get('step') for entry in log if entry.get('policy_decision') is not None]
    policy_decision_steps_filtered = [s for s in policy_decision_steps if s in steps] # Ищем в оригинальных 'steps'

    # --- Построение графика ---
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))
    fig.suptitle(f'Результаты симуляции: {config.get("economic_model_type", "N/A")} / {config.get("government_agent_type", "N/A")} ({num_steps_total} шагов)', fontsize=14)

    # График ВВП
    color_gdp = 'tab:blue'
    ax1.set_xlabel('Шаг симуляции', fontsize=axis_label_fontsize)
    ax1.set_ylabel('ВВП (GDP)', color=color_gdp, fontsize=axis_label_fontsize)
    line_gdp, = ax1.plot(steps_plot, gdp_plot, color=color_gdp, label='ВВП (GDP)', **plot_kwargs)
    ax1.tick_params(axis='y', labelcolor=color_gdp, labelsize=axis_label_fontsize-1)
    ax1.grid(True, linestyle=':')

    # График Налоговой Ставки (на второй оси Y)
    ax2 = ax1.twinx()
    color_tax = 'tab:red'
    ax2.set_ylabel('Налоговая ставка', color=color_tax, fontsize=axis_label_fontsize)
    line_tax, = ax2.plot(steps_plot, tax_rate_plot, color=color_tax, linestyle='--', label='Налоговая ставка', **plot_kwargs)
    ax2.tick_params(axis='y', labelcolor=color_tax, labelsize=axis_label_fontsize-1)
    # Установим лимиты для налоговой ставки
    max_tax_rate = max(tax_rate_plot) if tax_rate_plot else 0.5
    ax2.set_ylim(bottom=0, top=max(max_tax_rate*1.1, 0.5))

    # Вертикальные линии
    v_lines = []
    for s_dec in policy_decision_steps_filtered:
        if steps_plot and min(steps_plot) <= s_dec <= max(steps_plot):
             vline = ax1.axvline(x=s_dec, color='green', linestyle=':', alpha=0.5, linewidth=0.8)
             if not v_lines: v_lines.append(vline)

    # Легенда
    handles = [line_gdp, line_tax]
    labels = [h.get_label() for h in handles]
    if v_lines:
        handles.append(v_lines[0])
        labels.append('Обновление политики')
    ax1.legend(handles, labels, loc='best', fontsize=legend_fontsize)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_single_market_results(results_data: Dict[str, Any]):
    if not results_data:
        print("Ошибка: Нет данных для визуализации.")
        return
    log = results_data.get('simulation_log')
    if not log:
        print("Ошибка: 'simulation_log' отсутствует или пуст.")
        return

    config = results_data.get('config', {})
    num_steps_total = len(log)
    print(f"Данные лога симуляции получены ({num_steps_total} шагов). Начинаем построение графиков...")

    is_high_iteration_mode = num_steps_total > HIGH_ITERATION_THRESHOLD

    # --- ИНИЦИАЛИЗАЦИЯ ПЕРЕМЕННЫХ ДЛЯ СТИЛЕЙ ГРАФИКОВ ---
    plot_kwargs_light = {'marker': None, 'linewidth': 1.0, 'alpha': 0.9}
    plot_kwargs_default = {'marker': '.', 'markersize': 4, 'linewidth': 1.2, 'alpha': 0.8}
    current_plot_kwargs = plot_kwargs_light if is_high_iteration_mode else plot_kwargs_default
    # ---------------------------------------------------------

    axis_label_fontsize = 9
    title_fontsize = 11
    legend_fontsize = 8

    downsample_factor = 1
    moving_average_window = 1

    if num_steps_total > 2000:
        downsample_factor = 10
        moving_average_window = 5
        print(f"Включен режим очень высокой детализации: downsample_factor={downsample_factor}, moving_average_window={moving_average_window}")
    elif is_high_iteration_mode:
        downsample_factor = 5
        print(f"Включен режим высокой детализации: downsample_factor={downsample_factor}")

    # --- Извлечение данных для графиков ---
    steps_orig = [get_metric(entry, 'step') for entry in log]
    market_price_orig = [get_metric(entry, 'market_price') for entry in log]
    traded_quantity_orig = [get_metric(entry, 'actual_traded_quantity') for entry in log]
    social_welfare_orig = [get_metric(entry, 'social_welfare') for entry in log]
    consumer_surplus_orig = [get_metric(entry, 'total_consumer_surplus') for entry in log]
    firm_profit_orig = [get_metric(entry, 'total_firm_profit') for entry in log]
    gov_revenue_orig = [get_metric(entry, 'government_net_revenue') for entry in log]
    tax_per_unit_orig = [get_metric(entry, 'tax_per_unit') for entry in log]
    subsidy_per_unit_orig = [get_metric(entry, 'subsidy_per_unit') for entry in log]
    price_ceiling_orig = [get_metric(entry, 'price_ceiling', -1) for entry in log]
    price_floor_orig = [get_metric(entry, 'price_floor', -1) for entry in log]
    gov_purchases_orig = [get_metric(entry, 'government_purchase_quantity') for entry in log]

    valid_indices = [i for i, s in enumerate(steps_orig) if s is not None]
    if not valid_indices:
        print("Ошибка: Не найдено валидных шагов.")
        return

    def filter_by_valid_indices(*data_lists):
        return tuple([data_list[i] for i in valid_indices] if data_list and len(data_list) > max(valid_indices, default=-1) else [] for data_lists in data_lists)


    (steps, market_price, traded_quantity, social_welfare, consumer_surplus, firm_profit, gov_revenue,
     tax_per_unit, subsidy_per_unit, price_ceiling, price_floor, gov_purchases) = filter_by_valid_indices(
        steps_orig, market_price_orig, traded_quantity_orig, social_welfare_orig,
        consumer_surplus_orig, firm_profit_orig, gov_revenue_orig, tax_per_unit_orig,
        subsidy_per_unit_orig, price_ceiling_orig, price_floor_orig, gov_purchases_orig
    )

    # Проверка на пустые списки после фильтрации
    if not steps:
        print("Ошибка: После фильтрации не осталось данных для шагов.")
        return

    (steps_plot, market_price_plot, traded_quantity_plot, social_welfare_plot,
     consumer_surplus_plot, firm_profit_plot, gov_revenue_plot, tax_per_unit_plot,
     subsidy_per_unit_plot, price_ceiling_plot, price_floor_plot, gov_purchases_plot
     ) = downsample_data(
        steps, market_price, traded_quantity, social_welfare, consumer_surplus, firm_profit, gov_revenue,
        tax_per_unit, subsidy_per_unit, price_ceiling, price_floor, gov_purchases,
        factor=downsample_factor
    )

    if moving_average_window > 1:
        # Убедимся, что передаем списки, а не None или пустые списки в moving_average
        market_price_plot = moving_average(market_price_plot if market_price_plot else [], moving_average_window)
        traded_quantity_plot = moving_average(traded_quantity_plot if traded_quantity_plot else [], moving_average_window)
        social_welfare_plot = moving_average(social_welfare_plot if social_welfare_plot else [], moving_average_window)
        consumer_surplus_plot = moving_average(consumer_surplus_plot if consumer_surplus_plot else [], moving_average_window) # Добавил
        firm_profit_plot = moving_average(firm_profit_plot if firm_profit_plot else [], moving_average_window)       # Добавил
        gov_revenue_plot = moving_average(gov_revenue_plot if gov_revenue_plot else [], moving_average_window)       # Добавил


    policy_decision_steps = [entry.get('step') for entry in log if entry.get('policy_decision') is not None]
    policy_decision_steps_filtered = [s for s in policy_decision_steps if s in steps] # Ищем в оригинальных 'steps'

    # --- Построение графиков ---
    fig, axs = plt.subplots(3, 2, figsize=(19, 16), sharex=True)
    fig.suptitle(f'Результаты симуляции: {config.get("economic_model_type", "N/A")} / {config.get("government_agent_type", "N/A")} ({num_steps_total} шагов)', fontsize=15)

    # 1. Рыночная цена и объем торгов
    ax = axs[0, 0]
    ax.plot(steps_plot, market_price_plot, label='Рыночная цена', color='tab:blue', **current_plot_kwargs)
    ax.set_ylabel('Цена', color='tab:blue', fontsize=axis_label_fontsize)
    ax.tick_params(axis='y', labelcolor='tab:blue', labelsize=axis_label_fontsize-1)
    ax.grid(True, linestyle=':')
    ax_twin = ax.twinx()
    ax_twin.plot(steps_plot, traded_quantity_plot, label='Объем торгов', color='tab:green', linestyle='--', **current_plot_kwargs)
    ax_twin.set_ylabel('Объем', color='tab:green', fontsize=axis_label_fontsize)
    ax_twin.tick_params(axis='y', labelcolor='tab:green', labelsize=axis_label_fontsize-1)
    ax.set_title('Рыночная цена и Объем торгов', fontsize=title_fontsize)
    ax.legend(loc='upper left', fontsize=legend_fontsize)
    ax_twin.legend(loc='upper right', fontsize=legend_fontsize)
    for s_dec in policy_decision_steps_filtered:
        ax.axvline(x=s_dec, color='green', linestyle=':', alpha=0.5, linewidth=0.8)

    # 2. Социальное благосостояние и его компоненты
    ax = axs[0, 1]
    ax.plot(steps_plot, social_welfare_plot, label='Общ. благосостояние', color='black', linewidth=1.5 if not is_high_iteration_mode else 1.2)
    ax.plot(steps_plot, consumer_surplus_plot, label='Излишек потреб.', color='tab:cyan', linestyle=':', **current_plot_kwargs)
    ax.plot(steps_plot, firm_profit_plot, label='Прибыль фирм', color='tab:orange', linestyle=':', **current_plot_kwargs)
    ax.plot(steps_plot, gov_revenue_plot, label='Доход бюджета', color='tab:gray', linestyle=':', **current_plot_kwargs) # Добавил доход бюджета сюда для полноты
    ax.set_title('Социальное благосостояние и компоненты', fontsize=title_fontsize)
    ax.set_ylabel('Значение', fontsize=axis_label_fontsize)
    ax.tick_params(axis='y', labelsize=axis_label_fontsize-1)
    ax.legend(fontsize=legend_fontsize)
    ax.grid(True, linestyle=':')
    for s_dec in policy_decision_steps_filtered:
        ax.axvline(x=s_dec, color='green', linestyle=':', alpha=0.5, linewidth=0.8)


    # 3. Налоги и Субсидии
    ax = axs[1, 0]
    tax_plot_data = tax_per_unit_plot if tax_per_unit_plot is not None else []
    subsidy_plot_data = subsidy_per_unit_plot if subsidy_per_unit_plot is not None else []

    ax.plot(steps_plot, tax_plot_data, label='Налог на ед.', color='tab:red', **current_plot_kwargs)
    ax.plot(steps_plot, subsidy_plot_data, label='Субсидия на ед.', color='tab:olive', **current_plot_kwargs)
    ax.set_title('Налоги и Субсидии (на ед. товара)', fontsize=title_fontsize)
    ax.set_ylabel('Ставка', fontsize=axis_label_fontsize)
    ax.tick_params(axis='y', labelsize=axis_label_fontsize-1)
    ax.legend(fontsize=legend_fontsize)
    ax.grid(True, linestyle=':')

    combined_tax_subsidy = [x for x in tax_plot_data + subsidy_plot_data if x is not None]
    min_y_val_tax = min(combined_tax_subsidy, default=0) if combined_tax_subsidy else 0
    max_y_val_tax = max(combined_tax_subsidy, default=0) if combined_tax_subsidy else 0
    ax.set_ylim(bottom=min(0, min_y_val_tax) - 0.5, top=max(max_y_val_tax * 1.1 + 0.5, 0.5)) # Адаптивные лимиты
    for s_dec in policy_decision_steps_filtered:
        ax.axvline(x=s_dec, color='green', linestyle=':', alpha=0.5, linewidth=0.8)

    # 4. Потолок и Пол цены
    ax = axs[1, 1]
    pc_plot_viz = [pc if pc != -1 and pc is not None else None for pc in (price_ceiling_plot if price_ceiling_plot is not None else [])]
    pf_plot_viz = [pf if pf != -1 and pf is not None else None for pf in (price_floor_plot if price_floor_plot is not None else [])]
    market_price_plot_safe = market_price_plot if market_price_plot is not None else []

    ax.plot(steps_plot, market_price_plot_safe, label='Рыночная цена', color='gray', linestyle=':', alpha=0.6, linewidth=0.8)
    ax.plot(steps_plot, pc_plot_viz, label='Потолок цены', color='tab:pink', linestyle='--', **current_plot_kwargs)
    ax.plot(steps_plot, pf_plot_viz, label='Пол цены', color='tab:brown', linestyle='--', **current_plot_kwargs)
    ax.set_title('Регулирование цен', fontsize=title_fontsize)
    ax.set_ylabel('Уровень цены', fontsize=axis_label_fontsize)
    ax.tick_params(axis='y', labelsize=axis_label_fontsize-1)
    ax.legend(fontsize=legend_fontsize)
    ax.grid(True, linestyle=':')
    ax.set_ylim(bottom=0) # Цена не может быть отрицательной
    for s_dec in policy_decision_steps_filtered:
        ax.axvline(x=s_dec, color='green', linestyle=':', alpha=0.5, linewidth=0.8)

    # 5. Госзакупки
    ax = axs[2, 0]
    gov_purchases_plot_safe = gov_purchases_plot if gov_purchases_plot is not None else []
    ax.plot(steps_plot, gov_purchases_plot_safe, label='Объем госзакупок', color='tab:purple', **current_plot_kwargs)
    ax.set_ylabel('Объем закупок', fontsize=axis_label_fontsize)
    ax.tick_params(axis='y', labelsize=axis_label_fontsize-1)
    ax.grid(True, linestyle=':')
    ax.set_title('Госзакупки', fontsize=title_fontsize)
    ax.legend(loc='best', fontsize=legend_fontsize)
    for s_dec in policy_decision_steps_filtered:
        ax.axvline(x=s_dec, color='green', linestyle=':', alpha=0.5, linewidth=0.8)

    # 6. Доход бюджета (отдельный график)
    ax = axs[2, 1]
    gov_revenue_plot_safe = gov_revenue_plot if gov_revenue_plot is not None else []
    ax.plot(steps_plot, gov_revenue_plot_safe, label='Чистый доход бюджета', color='tab:gray', **current_plot_kwargs)
    ax.set_ylabel('Доход бюджета', fontsize=axis_label_fontsize)
    ax.tick_params(axis='y', labelsize=axis_label_fontsize-1)
    ax.grid(True, linestyle=':')
    ax.set_title('Доход бюджета', fontsize=title_fontsize)
    ax.legend(loc='best', fontsize=legend_fontsize)
    for s_dec in policy_decision_steps_filtered:
        ax.axvline(x=s_dec, color='green', linestyle=':', alpha=0.5, linewidth=0.8)


    # Устанавливаем общие метки оси X
    for i in range(3):
        axs[i,0].set_xlabel('Шаг симуляции', fontsize=axis_label_fontsize)
        axs[i,1].set_xlabel('Шаг симуляции', fontsize=axis_label_fontsize)
        axs[i,0].tick_params(axis='x', labelsize=axis_label_fontsize-1)
        axs[i,1].tick_params(axis='x', labelsize=axis_label_fontsize-1)


    plt.subplots_adjust(
        left=0.07,
        right=0.93,
        top=0.94, # Поднял, чтобы дать место заголовку
        bottom=0.05, # Уменьшил, чтобы дать место меткам X
        hspace=0.35, # Уменьшил вертикальный интервал
        wspace=0.30  # Уменьшил горизонтальный интервал
    )

    print("Построение графиков для SingleMarketModel завершено. Отображение...")
    plt.show()


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    filename_to_load = DEFAULT_RESULTS_FILENAME
    results_file_path = script_dir.parent / "logs" / filename_to_load

    results = load_results(results_file_path)
    if results:
        model_type = results.get('config', {}).get('economic_model_type')
        print(f"Обнаружен тип модели: {model_type}")

        if model_type == "SingleMarketModel":
            plot_single_market_results(results)
        elif model_type == "SimpleGrowthModel":
            plot_simple_growth_results(results)
        elif model_type == "LinearStochasticSystem":
            plot_linear_system_results(results) # Вызываем функцию для LinearStochasticSystem
        else:
            print(f"Предупреждение: Неизвестный тип модели '{model_type}'. Визуализация не реализована.")

    else:
        print(f"Не удалось загрузить данные из {results_file_path}.")