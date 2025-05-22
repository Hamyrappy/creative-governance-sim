# visualize_linear_system.py
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List

# Keep only constants relevant to the remaining functions
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
    metrics_dict = entry.get('metrics')
    if isinstance(metrics_dict, dict):
        return metrics_dict.get(metric_name, default)
    return default


def downsample_data(*args, factor: int):
    """Уменьшает плотность данных, беря каждую factor-ую точку."""
    if factor <= 1:
        return args
    processed_args = []
    for arg in args:
        if arg is not None and isinstance(arg, (list, np.ndarray)) and len(arg) >= factor:
            processed_args.append(arg[::factor])
        else:
            processed_args.append(arg)
    return tuple(processed_args)


def moving_average(data: List[Optional[float]], window_size: int) -> List[Optional[float]]:
    """Расчет скользящего среднего. Обрабатывает None значения."""
    if window_size <= 1 or not data:
        return data

    np_data = np.array([x if x is not None else np.nan for x in data], dtype=float)

    if len(np_data) < window_size:
        return [None] * len(data)

    result = [None] * (window_size - 1)

    for i in range(window_size - 1, len(data)):
        window = np_data[i - window_size + 1 : i + 1]
        if np.isnan(window).all():
             mean_val = None
        else:
             mean_val = np.nanmean(window)
             if np.isnan(mean_val):
                 mean_val = None
        result.append(mean_val)

    return result


def plot_linear_system_results(results_data: Dict[str, Any]):
    """
    Визуализирует результаты ТОЛЬКО для модели LinearStochasticSystem.
    ГРАФИКИ ВЫВОДЯТСЯ ПО ОЧЕРЕДИ.
    На первом графике линия управления U_k отключена.
    Моменты принятия решения отмечены СПЛОШНЫМИ КРАСНЫМИ линиями на ОБОИХ графиках.
    """
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
    plot_kwargs = {'marker': None, 'linewidth': 2.0, 'alpha': 0.9} if is_high_iteration_mode else {'marker': '.', 'markersize': 6, 'linewidth': 2.0, 'alpha': 0.8}

    pub_base_fontsize = 18
    pub_title_fontsize = pub_base_fontsize + 4
    pub_label_fontsize = pub_base_fontsize + 2
    pub_tick_labelsize = pub_base_fontsize
    pub_legend_fontsize = pub_base_fontsize - 2
    # --- ------------------------------- ---

    downsample_factor = 1
    if num_steps_total > 1000:
        downsample_factor = 5
    if num_steps_total > 5000:
         downsample_factor = 10

    performance_window = agent_params.get('performance_window', 20)
    if performance_window <= 0: performance_window = 1
    if performance_window > num_steps_total: performance_window = num_steps_total

    plotting_ma_window = 1
    if num_steps_total > 200:
        plotting_ma_window = max(1, num_steps_total // 200)
    if plotting_ma_window > 20: plotting_ma_window = 20

    # --- Извлечение данных из лога ---
    steps_orig = [get_metric(entry, 'step') for entry in log]
    current_x_orig = [get_metric(entry, 'current_x') for entry in log]
    target_x_val = get_metric(log[0] if log else {}, 'target_x', None)
    policy_decision_steps_log = [entry.get('step') for entry in log if entry.get('policy_decision') is not None]
    current_u_orig = [get_metric(entry, 'current_u') for entry in log] # Нужен для MSU

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
    kpi_steps_full = []

    np_x_full = np.array(current_x_full, dtype=float)
    np_u_full = np.array(current_u_full, dtype=float)
    np_target_x = np.array(target_x_val, dtype=float) if target_x_val is not None else np.nan

    for k in range(performance_window - 1, len(steps_full)):
        window_x = np_x_full[k - performance_window + 1 : k + 1]
        window_u = np_u_full[k - performance_window + 1 : k + 1]
        mse_k = np.nanmean((window_x - np_target_x)**2) if len(window_x) > 0 and not np.isnan(np_target_x) else np.nan
        msu_k = np.nanmean(window_u**2) if len(window_u) > 0 else np.nan
        mse_values_full.append(mse_k if not np.isnan(mse_k) else None)
        msu_values_full.append(msu_k if not np.isnan(msu_k) else None)
        kpi_steps_full.append(steps_full[k])

    # --- Downsampling и MA для ПЛОТИНГА ---
    (steps_plot_x_u, current_x_plot) = downsample_data(
        steps_full, current_x_full, factor=downsample_factor
    )
    (kpi_steps_plot, mse_values_plot, msu_values_plot) = downsample_data(
        kpi_steps_full, mse_values_full, msu_values_full, factor=downsample_factor
    )

    if not steps_plot_x_u:
         print("Ошибка: После downsampling не осталось данных для графика состояния.")
         return

    current_x_plot_smooth = moving_average(list(current_x_plot) if current_x_plot is not None else [], plotting_ma_window)
    mse_values_plot_smooth = moving_average(list(mse_values_plot) if mse_values_plot is not None else [], plotting_ma_window)
    msu_values_plot_smooth = moving_average(list(msu_values_plot) if msu_values_plot is not None else [], plotting_ma_window)

    # --- ПОСТРОЕНИЕ ГРАФИКОВ ПО ОЧЕРЕДИ ---

    print("\n--- Построение Графика 1: Состояние системы и моменты принятия решений ---")
    fig1, ax1 = plt.subplots(1, 1, figsize=(14, 8))
    fig1.suptitle(f'Симуляция: {config.get("economic_model_type", "N/A")} / {config.get("government_agent_type", "N/A")}', fontsize=pub_title_fontsize + 2)

    # --- График Состояния (x_k) ---
    color_x = 'tab:blue'
    ax1.set_xlabel('Шаг симуляции', fontsize=pub_label_fontsize)
    ax1.set_ylabel('Состояние системы (x_k)', color=color_x, fontsize=pub_label_fontsize)
    line_x = None
    if len(steps_plot_x_u) == len(current_x_plot_smooth):
        line_x, = ax1.plot(steps_plot_x_u, current_x_plot_smooth, color=color_x, label=f'Состояние x_k (MA={plotting_ma_window})', **plot_kwargs)
    else:
        print(f"Предупреждение: Длина steps_plot_x_u ({len(steps_plot_x_u)}) не совпадает с длиной current_x_plot_smooth ({len(current_x_plot_smooth)}). Пропуск отрисовки x_k.")
    ax1.tick_params(axis='y', labelcolor=color_x, labelsize=pub_tick_labelsize)
    ax1.tick_params(axis='x', labelsize=pub_tick_labelsize)
    ax1.grid(True, linestyle=':', which='both')

    target_line = None
    if target_x_val is not None:
        target_line = ax1.axhline(y=target_x_val, color='black', linestyle='--', linewidth=1.5, label=f'Цель x* = {target_x_val}')

    # --- Вертикальные СПЛОШНЫЕ КРАСНЫЕ линии для моментов принятия решения ---
    decision_lines = []
    policy_decision_steps_plot1 = [s for s in policy_decision_steps_log
                                  if s is not None and steps_plot_x_u and min(steps_plot_x_u) <= s <= max(steps_plot_x_u)]
    for s_dec in policy_decision_steps_plot1:
         # Используем красный цвет, alpha и linewidth как раньше, но linestyle='-' (по умолчанию)
         vline = ax1.axvline(x=s_dec, color='red', linestyle='-', alpha=0.7, linewidth=1.2)
         if not decision_lines: decision_lines.append(vline)

    # --- Легенда для первого графика ---
    handles1 = []
    labels1 = []
    if line_x:
        handles1.append(line_x)
        labels1.append(line_x.get_label())
    if target_line:
        handles1.insert(0, target_line)
        labels1.insert(0, target_line.get_label())
    if decision_lines:
         handles1.append(decision_lines[0])
         labels1.append('Момент принятия решения')

    if handles1:
      ax1.legend(handles1, labels1, loc='best', fontsize=pub_legend_fontsize)
    ax1.set_title(f'Динамика состояния (x) и моменты принятия решений ({num_steps_total} шагов)', fontsize=pub_title_fontsize)

    fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # --- ПОСТРОЕНИЕ ГРАФИКА 2: KPI ---
    print("\n--- Построение Графика 2: Показатели KPI (MSE, MSU) ---")

    if not kpi_steps_plot:
        print("Нет данных KPI для построения второго графика.")
        return

    fig2, ax2 = plt.subplots(1, 1, figsize=(14, 8))
    fig2.suptitle(f'Симуляция: {config.get("economic_model_type", "N/A")} / {config.get("government_agent_type", "N/A")}', fontsize=pub_title_fontsize + 2)

    # --- График MSE ---
    ax2.set_xlabel('Шаг симуляции', fontsize=pub_label_fontsize)
    ax2.set_ylabel('Среднеквадратичная ошибка (MSE)', color='tab:purple', fontsize=pub_label_fontsize)
    line_mse = None
    if len(kpi_steps_plot) == len(mse_values_plot_smooth):
        line_mse, = ax2.plot(kpi_steps_plot, mse_values_plot_smooth, color='tab:purple', label=f'MSE (окно={performance_window}, MA={plotting_ma_window})', **plot_kwargs)
    else:
        print(f"Предупреждение: Длина kpi_steps_plot ({len(kpi_steps_plot)}) не совпадает с длиной mse_values_plot_smooth ({len(mse_values_plot_smooth)}). Пропуск отрисовки MSE.")
    ax2.tick_params(axis='y', labelcolor='tab:purple', labelsize=pub_tick_labelsize)
    ax2.tick_params(axis='x', labelsize=pub_tick_labelsize)
    ax2.grid(True, linestyle=':', which='both')

    # --- График MSU ---
    ax2_twin = ax2.twinx()
    color_msu = 'tab:orange'
    ax2_twin.set_ylabel('Среднеквадратичное управление (MSU)', color=color_msu, fontsize=pub_label_fontsize)
    line_msu = None
    if len(kpi_steps_plot) == len(msu_values_plot_smooth):
        line_msu, = ax2_twin.plot(kpi_steps_plot, msu_values_plot_smooth, color=color_msu, linestyle='--', label=f'MSU (окно={performance_window}, MA={plotting_ma_window})', **plot_kwargs)
    else:
        print(f"Предупреждение: Длина kpi_steps_plot ({len(kpi_steps_plot)}) не совпадает с длиной msu_values_plot_smooth ({len(msu_values_plot_smooth)}). Пропуск отрисовки MSU.")
    ax2_twin.tick_params(axis='y', labelcolor=color_msu, labelsize=pub_tick_labelsize)

    # --- Вертикальные СПЛОШНЫЕ КРАСНЫЕ линии для моментов принятия решения (на графике KPI) ---
    policy_decision_steps_plot2 = [s for s in policy_decision_steps_log
                                   if s is not None and kpi_steps_plot and min(kpi_steps_plot) <= s <= max(kpi_steps_plot)]
    for s_dec in policy_decision_steps_plot2:
         ax2.axvline(x=s_dec, color='red', linestyle='-', alpha=0.7, linewidth=1.2)

    # --- Легенда для второго графика (без явного упоминания красных линий, т.к. они есть на первом) ---
    handles2 = []
    labels2 = []
    if line_mse:
        handles2.append(line_mse)
        labels2.append(line_mse.get_label())
    if line_msu:
        handles2.append(line_msu)
        labels2.append(line_msu.get_label())

    if handles2:
        ax2.legend(handles2, labels2, loc='best', fontsize=pub_legend_fontsize)
    ax2.set_title(f'Показатели эффективности (KPI): MSE и MSU ({num_steps_total} шагов)', fontsize=pub_title_fontsize)

    # Настройка пределов Y
    if mse_values_plot_smooth:
      valid_mse = [v for v in mse_values_plot_smooth if v is not None]
      if valid_mse:
        min_mse, max_mse = min(valid_mse), max(valid_mse)
        if max_mse > min_mse : ax2.set_ylim(min_mse*0.9 if min_mse > 0 else -0.1 * max_mse if max_mse > 0 else -0.1, max_mse * 1.1 if max_mse > 0 else 0.1)

    if msu_values_plot_smooth:
      valid_msu = [v for v in msu_values_plot_smooth if v is not None]
      if valid_msu:
        min_msu, max_msu = min(valid_msu), max(valid_msu)
        if max_msu > min_msu: ax2_twin.set_ylim(min_msu*0.9 if min_msu > 0 else -0.1*max_msu if max_msu > 0 else -0.1, max_msu * 1.1 if max_msu > 0 else 0.1)

    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    filename_to_load = "simulation_results_exp4.json" # DEFAULT_RESULTS_FILENAME задается сверху файла
    results_file_path = script_dir.parent / "logs" / filename_to_load

    results = load_results(results_file_path)
    if results:
        model_type = results.get('config', {}).get('economic_model_type')
        print(f"Обнаружен тип модели: {model_type}")

        if model_type == "LinearStochasticSystem":
            plot_linear_system_results(results)
        else:
            print(f"Ошибка: Этот скрипт предназначен только для визуализации 'LinearStochasticSystem', "
                  f"а в файле указан тип '{model_type}'.")

    else:
        print(f"Не удалось загрузить данные из {results_file_path}.")