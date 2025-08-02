# visualize_linear_system.py
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List

# Keep only constants relevant to the remaining functions
DEFAULT_RESULTS_FILENAME = "simulation_results.json"
HIGH_ITERATION_THRESHOLD = 100
SHOCK_STEP_EXPERIMENT_4 = 500 # Define the shock step for clarity

def load_results(filepath: Path) -> Optional[Dict[str, Any]]:
    """Loads simulation results from a JSON file."""
    print(f"Attempting to load results from: {filepath}")
    if not filepath.is_file():
        print(f"Error: Results file '{filepath}' not found.")
        return None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            results_data = json.load(f)
        print("Results loaded successfully.")
        return results_data
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error during loading: {e}")
        return None

def get_metric(entry: Dict[str, Any], metric_name: str, default: Any = None) -> Any:
    """Safely extracts a metric from a log entry."""
    metrics_dict = entry.get('metrics')
    if isinstance(metrics_dict, dict):
        return metrics_dict.get(metric_name, default)
    return default


def downsample_data(*args, factor: int):
    """Downsamples data by taking every 'factor'-th point."""
    if factor <= 1:
        return args if len(args) > 0 else (None,) # Ensure tuple for single arg

    processed_args = []
    for arg in args:
        if arg is not None and isinstance(arg, (list, np.ndarray)) and len(arg) >= factor:
            processed_args.append(arg[::factor])
        else:
            processed_args.append(arg)
    return tuple(processed_args)


def moving_average(data: List[Optional[float]], window_size: int) -> List[Optional[float]]:
    """Calculates moving average, handling None values."""
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
    Visualizes results ONLY for the LinearStochasticSystem model.
    """
    if not results_data:
        print("Error: No data to visualize LinearStochasticSystem.")
        return
    log = results_data.get('simulation_log')
    if not log:
        print("Error: 'simulation_log' is missing or empty for LinearStochasticSystem.")
        return

    config = results_data.get('config', {})
    agent_params = results_data.get('agent_params', {})
    num_steps_total = len(log)
    print(f"LinearStochasticSystem log data retrieved ({num_steps_total} steps). Starting plotting...")

    is_high_iteration_mode = num_steps_total > HIGH_ITERATION_THRESHOLD
    plot_kwargs = {'marker': None, 'linewidth': 2.0, 'alpha': 0.9} if is_high_iteration_mode else {'marker': '.', 'markersize': 6, 'linewidth': 1.5, 'alpha': 0.8}

    base_fontsize = 16
    title_fontsize = base_fontsize + 2
    label_fontsize = base_fontsize
    tick_labelsize = base_fontsize - 1
    legend_fontsize = base_fontsize - 2

    downsample_factor = 1
    if num_steps_total > 1000: downsample_factor = 5
    if num_steps_total > 5000: downsample_factor = 10

    performance_window = agent_params.get('performance_window', 30)
    if performance_window <= 0: performance_window = 1
    if performance_window > num_steps_total: performance_window = num_steps_total

    plotting_ma_window_state = 5
    if num_steps_total <= 200: plotting_ma_window_state = 1
    if plotting_ma_window_state > 20: plotting_ma_window_state = 20
    
    plotting_ma_window_kpi = 5

    steps_orig = [get_metric(entry, 'step') for entry in log]
    current_x_orig = [get_metric(entry, 'current_x') for entry in log]
    target_x_history = [get_metric(entry, 'target_x') for entry in log]
    policy_decision_steps_log = [entry.get('step') for entry in log if entry.get('policy_decision') is not None]
    current_u_orig = [get_metric(entry, 'current_u') for entry in log]

    valid_indices = [i for i, s in enumerate(steps_orig) if s is not None]
    if not valid_indices:
        print("Error: No valid steps found in LinearStochasticSystem log.")
        return

    steps_full = np.array([steps_orig[i] for i in valid_indices])
    current_x_full = np.array([current_x_orig[i] for i in valid_indices])
    current_u_full = np.array([current_u_orig[i] for i in valid_indices])
    target_x_full_series = np.array([target_x_history[i] for i in valid_indices])

    if steps_full.size == 0:
         print("Error: No step data left after filtering for LinearStochasticSystem.")
         return

    mse_values_full, msu_values_full, kpi_steps_full = [], [], []
    for k_idx in range(performance_window - 1, len(steps_full)):
        start_idx = k_idx - performance_window + 1
        window_x = current_x_full[start_idx : k_idx + 1]
        window_u = current_u_full[start_idx : k_idx + 1]
        window_target_x = target_x_full_series[start_idx : k_idx + 1]
        mse_k = np.nanmean((window_x - window_target_x)**2) if window_x.size > 0 else np.nan
        msu_k = np.nanmean(window_u**2) if window_u.size > 0 else np.nan
        mse_values_full.append(mse_k if not np.isnan(mse_k) else None)
        msu_values_full.append(msu_k if not np.isnan(msu_k) else None)
        kpi_steps_full.append(steps_full[k_idx])

    downsampled_x_u_target = downsample_data(steps_full, current_x_full, target_x_full_series, factor=downsample_factor)
    steps_plot_x_u, current_x_plot, target_x_plot = downsampled_x_u_target[0], downsampled_x_u_target[1], downsampled_x_u_target[2]
    
    downsampled_kpis = downsample_data(np.array(kpi_steps_full), np.array(mse_values_full), np.array(msu_values_full), factor=downsample_factor)
    kpi_steps_plot_ds, mse_values_plot_ds, msu_values_plot_ds = downsampled_kpis[0], downsampled_kpis[1], downsampled_kpis[2]


    if steps_plot_x_u is None or len(steps_plot_x_u) == 0:
         print("Error: No data left for state plot after downsampling.")
         return

    current_x_plot_smooth = moving_average(list(current_x_plot) if current_x_plot is not None else [], plotting_ma_window_state)
    mse_values_plot_smooth = moving_average(list(mse_values_plot_ds) if mse_values_plot_ds is not None else [], plotting_ma_window_kpi)
    msu_values_plot_smooth = moving_average(list(msu_values_plot_ds) if msu_values_plot_ds is not None else [], plotting_ma_window_kpi)

    print("\n--- Plotting Graph 1: System State, Decisions, and Shock ---")
    fig1, ax1 = plt.subplots(1, 1, figsize=(14, 7))
    color_x = 'steelblue'
    ax1.set_xlabel('Simulation Step', fontsize=label_fontsize)
    ax1.set_ylabel('System State (x_k)', color=color_x, fontsize=label_fontsize)
    line_x = None
    if steps_plot_x_u is not None and current_x_plot_smooth is not None and len(steps_plot_x_u) == len(current_x_plot_smooth):
        line_x, = ax1.plot(steps_plot_x_u, current_x_plot_smooth, color=color_x, label='State x_k', **plot_kwargs)
    ax1.tick_params(axis='y', labelcolor=color_x, labelsize=tick_labelsize)
    ax1.tick_params(axis='x', labelsize=tick_labelsize)
    ax1.grid(True, linestyle=':', which='both', alpha=0.7)

    # Plot Target x* in two segments for a visual break at SHOCK_STEP_EXPERIMENT_4
    target_line_plot1, target_line_plot2 = None, None
    legend_label_target_x = 'Target x*' # Use this for the first segment that gets a label
    
    if steps_plot_x_u is not None and target_x_plot is not None and len(steps_plot_x_u) == len(target_x_plot):
        # Find index for splitting
        break_idx_candidates = np.where(steps_plot_x_u >= SHOCK_STEP_EXPERIMENT_4)[0]
        
        if len(break_idx_candidates) > 0:
            break_idx = break_idx_candidates[0]
            
            # Segment 1: Before or at the very start of the shock
            if break_idx > 0:
                # x-values up to and including the step just before the new target takes effect for plotting
                # y-values are the old target value
                x_pre = np.concatenate((steps_plot_x_u[:break_idx], [SHOCK_STEP_EXPERIMENT_4 - 1e-6])) # End just before shock step
                y_pre = np.concatenate((target_x_plot[:break_idx], [target_x_plot[break_idx-1]]))      # Use target active before break
                target_line_plot1, = ax1.plot(x_pre, y_pre, color='black', linestyle='--', linewidth=1.5, drawstyle='steps-post', label=legend_label_target_x)
                legend_label_target_x = None # Ensure label is not repeated
            else: # Shock is at the very beginning, only plot post segment
                 pass

            # Segment 2: At and after the shock
            # x-values from the shock step onwards
            # y-values are the new target value
            x_post = np.concatenate(([SHOCK_STEP_EXPERIMENT_4], steps_plot_x_u[break_idx:]))
            y_post = np.concatenate(([target_x_plot[break_idx]], target_x_plot[break_idx:]))

            # Correct for potential duplicate if shock is exactly on a data point
            if steps_plot_x_u[break_idx] == SHOCK_STEP_EXPERIMENT_4:
                 x_post_corrected = steps_plot_x_u[break_idx:]
                 y_post_corrected = target_x_plot[break_idx:]
            else: # Shock is between data points, prepend correctly
                 x_post_corrected = np.concatenate(([SHOCK_STEP_EXPERIMENT_4],steps_plot_x_u[break_idx:]))
                 y_post_corrected = np.concatenate(([target_x_plot[break_idx]],target_x_plot[break_idx:])) # New target starts at shock

            target_line_plot2, = ax1.plot(x_post_corrected, y_post_corrected, color='black', linestyle='--', linewidth=1.5, drawstyle='steps-post', label=legend_label_target_x)
            if legend_label_target_x and target_line_plot2: legend_label_target_x = None


        else: # Shock is beyond data range, plot all as one segment
            target_line_plot1, = ax1.plot(steps_plot_x_u, target_x_plot, color='black', linestyle='--', linewidth=1.5, label=legend_label_target_x, drawstyle='steps-post')
            legend_label_target_x = None


    decision_lines_handles = []
    policy_decision_steps_plot1 = [s for s in policy_decision_steps_log if s is not None and steps_plot_x_u is not None and len(steps_plot_x_u) > 0 and min(steps_plot_x_u) <= s <= max(steps_plot_x_u)]
    decision_line_plotted_legend = False
    for s_dec in policy_decision_steps_plot1:
         vline = ax1.axvline(x=s_dec, color='red', linestyle='-', alpha=0.6, linewidth=1.2)
         if not decision_line_plotted_legend:
             decision_lines_handles.append(vline)
             decision_line_plotted_legend = True
    
    shock_line1_handle = None
    if steps_plot_x_u is not None and len(steps_plot_x_u) > 0 and min(steps_plot_x_u) <= SHOCK_STEP_EXPERIMENT_4 <= max(steps_plot_x_u):
        shock_line1_handle = ax1.axvline(x=SHOCK_STEP_EXPERIMENT_4, color='green', linestyle=':', linewidth=2, alpha=0.9, label=f'Target Change (Step {SHOCK_STEP_EXPERIMENT_4})')

    handles1, labels1 = [], []
    if target_line_plot1: handles1.append(target_line_plot1); labels1.append(target_line_plot1.get_label())
    elif target_line_plot2: handles1.append(target_line_plot2); labels1.append(target_line_plot2.get_label()) # If only post shock was plotted
    if line_x: handles1.append(line_x); labels1.append(line_x.get_label())
    if decision_lines_handles: handles1.append(decision_lines_handles[0]); labels1.append('Decision Moment')
    if shock_line1_handle: handles1.append(shock_line1_handle); labels1.append(shock_line1_handle.get_label())
    
    if handles1: ax1.legend(handles1, labels1, loc='upper left', fontsize=legend_fontsize)
    ax1.set_title(f'State Dynamics (x) and Key Events ({num_steps_total} steps)', fontsize=title_fontsize)
    fig1.tight_layout()
    plt.show()

    print("\n--- Plotting Graph 2: Performance Metrics (KPI): MSE and MSU and Shock ---")
    if kpi_steps_plot_ds is None or len(kpi_steps_plot_ds) == 0:
        print("No KPI data to plot the second graph after downsampling.")
        return

    fig2, ax2 = plt.subplots(1, 1, figsize=(14, 7))
    ax2.set_xlabel('Simulation Step', fontsize=label_fontsize)
    ax2.set_ylabel('Mean Squared Error (MSE)', color='tab:purple', fontsize=label_fontsize)
    line_mse = None
    if kpi_steps_plot_ds is not None and mse_values_plot_smooth is not None and len(kpi_steps_plot_ds) == len(mse_values_plot_smooth):
        line_mse, = ax2.plot(kpi_steps_plot_ds, mse_values_plot_smooth, color='tab:purple', label='MSE', **plot_kwargs)
    ax2.tick_params(axis='y', labelcolor='tab:purple', labelsize=tick_labelsize)
    ax2.tick_params(axis='x', labelsize=tick_labelsize)
    ax2.grid(True, linestyle=':', which='both', alpha=0.7)

    ax2_twin = ax2.twinx()
    color_msu = 'darkorange'
    ax2_twin.set_ylabel('Mean Squared Control (MSU)', color=color_msu, fontsize=label_fontsize)
    line_msu = None
    if kpi_steps_plot_ds is not None and msu_values_plot_smooth is not None and len(kpi_steps_plot_ds) == len(msu_values_plot_smooth):
        line_msu, = ax2_twin.plot(kpi_steps_plot_ds, msu_values_plot_smooth, color=color_msu, linestyle='--', label='MSU', **plot_kwargs)
    ax2_twin.tick_params(axis='y', labelcolor=color_msu, labelsize=tick_labelsize)

    kpi_decision_lines_handles = []
    policy_decision_steps_plot2 = [s for s in policy_decision_steps_log if s is not None and kpi_steps_plot_ds is not None and len(kpi_steps_plot_ds)>0 and min(kpi_steps_plot_ds) <= s <= max(kpi_steps_plot_ds)]
    kpi_decision_line_plotted_legend = False
    for s_dec in policy_decision_steps_plot2:
         vline_kpi = ax2.axvline(x=s_dec, color='red', linestyle='-', alpha=0.6, linewidth=1.2)
         if not kpi_decision_line_plotted_legend:
            kpi_decision_lines_handles.append(vline_kpi)
            kpi_decision_line_plotted_legend = True

    shock_line2_handle = None
    if kpi_steps_plot_ds is not None and len(kpi_steps_plot_ds) > 0 and min(kpi_steps_plot_ds) <= SHOCK_STEP_EXPERIMENT_4 <= max(kpi_steps_plot_ds):
        shock_line2_handle = ax2.axvline(x=SHOCK_STEP_EXPERIMENT_4, color='green', linestyle=':', linewidth=2, alpha=0.9, label=f'Target Change (Step {SHOCK_STEP_EXPERIMENT_4})')

    handles2, labels2 = [], []
    if line_mse: handles2.append(line_mse); labels2.append(line_mse.get_label())
    if line_msu: handles2.append(line_msu); labels2.append(line_msu.get_label())
    if kpi_decision_lines_handles : handles2.append(kpi_decision_lines_handles[0]); labels2.append('Decision Moment')
    if shock_line2_handle : handles2.append(shock_line2_handle); labels2.append(shock_line2_handle.get_label())

    if handles2: ax2.legend(handles2, labels2, loc='upper left', fontsize=legend_fontsize)
    ax2.set_title(f'Performance Metrics (KPI): MSE and MSU ({num_steps_total} steps)', fontsize=title_fontsize)
    
    if mse_values_plot_smooth:
        valid_mse = [v for v in mse_values_plot_smooth if v is not None and not np.isnan(v)]
        if valid_mse:
            min_val, max_val = min(valid_mse), max(valid_mse)
            padding = (max_val - min_val) * 0.05 if (max_val - min_val) > 1e-9 else 0.02
            ax2.set_ylim(max(0, min_val - padding), max_val + padding)
    
    if msu_values_plot_smooth:
        valid_msu = [v for v in msu_values_plot_smooth if v is not None and not np.isnan(v)]
        if valid_msu:
            min_val, max_val = min(valid_msu), max(valid_msu)
            padding = (max_val - min_val) * 0.05 if (max_val - min_val) > 1e-9 else 0.02
            ax2_twin.set_ylim(max(0, min_val - padding), max_val + padding)

    fig2.tight_layout()
    plt.show()


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    filename_to_load = "simulation_results_exp4.json"
    possible_log_paths = [script_dir.parent / "logs", script_dir / "logs", Path.cwd() / "logs"]
    logs_dir = next((path for path in possible_log_paths if path.exists() and path.is_dir()), None)
    
    if not logs_dir:
        print(f"Error: 'logs' directory not found.")
        exit()

    results_file_path = logs_dir / filename_to_load
    results = load_results(results_file_path)
    if results:
        model_type = results.get('config', {}).get('economic_model_type')
        print(f"Detected model type: {model_type}")
        if model_type == "LinearStochasticSystem":
            plot_linear_system_results(results)
        else:
            print(f"Error: Script for 'LinearStochasticSystem' only, found '{model_type}'.")
    else:
        print(f"Failed to load data from {results_file_path}.")