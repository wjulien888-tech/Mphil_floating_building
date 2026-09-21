
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import scienceplots
import numpy as np
import torch
from pathlib import Path
from src.dataloader.csv_dataset import CSVDataset
from src.training_and_evaluation.learner import METRICS_TENDENCY
import pandas as pd
import os
import tikzplotlib
plt.style.use('ieee')
logging = logging.getLogger('pytorch_lightning')


def plot_predictions(experiment_path: str,
                     dataset: CSVDataset,
                     predictions: torch.Tensor,
                     targets: torch.Tensor,
                     output_features: List[str],
                     t_initial: int, t_final: int,
                     save_folder_name: str = "predictions",
                     data_split: str = "test") -> None:
    """This method is used to plot the predictions vs. ground truth.

    Args:
        dataset (CSVDataset): the dataset to obtain the grounf truth profiles.
        predictions (torch.Tensor): the predicted profiles.
        targets (torch.Tensor): the ground truth profiles.
        output_features (List[str]): the features that have been predicted.
        t_initial (int): the initial time step to be plotted.
        t_final (int): the final time step to be plotted.
        save_folder_name (str): the name of the folder where the plots will be saved.
        data_split (str): The data split to be used for the plot.
    """
    # Legacy utility import is intentionally lazy: importing src.utils also
    # initialises optional Syne Tune backends which closed-loop plotting does
    # not need.
    from src.utils import denormalise

    # Get ground truth profiles
    profiles = targets

    # Create predictions folder
    Path(os.path.join(experiment_path, save_folder_name, data_split)).mkdir(
        parents=True, exist_ok=True)
    save_directory = os.path.join(experiment_path, save_folder_name, data_split)

    prediction_horizon = predictions.shape[1]
    for prediction_sample in range(prediction_horizon):
        # Get predicted profiles and rescale them
        profiles_hat = predictions[:, prediction_sample, :]
        profiles_targets = profiles[:, prediction_sample, :]

        scaling_factors_min = dataset.scaling_factors_min[dataset.indices_output_features]
        scaling_factors_max = dataset.scaling_factors_max[dataset.indices_output_features]

        profiles_hat = denormalise(
            profiles_hat, scaling_factors_min, scaling_factors_max)
        profiles_targets = denormalise(
            profiles_targets, scaling_factors_min, scaling_factors_max)

        # Plot the profiles
        fig, axs = plt.subplots(
            len(dataset.available_features), 1, figsize=(10, 6))
        fig.suptitle('Predictions vs. Ground Truth')
        for i in range(len(dataset.available_features)):
            # Subplot title
            axs[i].set_title(dataset.available_features[i])
            # Plot the ground truth profile
            axs[i].grid()
            # Plot predicted profiles
            if dataset.available_features[i] in output_features:
                # if the profile is in the predicted features, plot the prediction
                idx = output_features.index(dataset.available_features[i])
                axs[i].plot(profiles_targets[t_initial:t_final, idx], label="Ground Truth" if i == len(
                dataset.available_features) - 1 else None)
                axs[i].plot(profiles_hat[t_initial:t_final, idx], label='Prediction' if i == len(
                    dataset.available_features) - 1 else None)

        # Add the legend horizontally below the subplots, outside the figure
        axs[-1].legend(loc='upper center',
                       bbox_to_anchor=(0.5, -0.2), fancybox=False, shadow=False, ncol=5)
        # Save figure
        if save_folder_name == "predictions":
            plt.savefig(os.path.join(
                save_directory,
                f"predictions_prediction_sample_{prediction_sample}.png"), bbox_inches='tight')
            plt.close(fig)
            plt.clf()
        if save_folder_name != "predictions" and prediction_sample == 0:
            plt.savefig(os.path.join(save_directory,
                f"predictions_prediction_sample_{prediction_sample}.png"), bbox_inches='tight')
            tikzplotlib.save(os.path.join(save_directory,
                f"predictions_prediction_sample_{prediction_sample}.tex"))
            plt.close(fig)

def plot_decision_variables(experiment_path: str,
                            decision_variables: Dict[str, torch.Tensor],
                            constraints: Dict[str, List[Union[float, torch.Tensor]]],
                            data_split: str) -> None:
    """Plot the decision variables.

    Args:
        experiment_path (str): Path to the experiment folder.
        decision_variables (Dict[str, torch.Tensor]): Dictionary of decision variables.
        constraints (Dict[str, List[Union[float, torch.Tensor]]]): Dictionary of constraints.
        data_split (str): Data split.
    """
    prediction_horizon = decision_variables['Ps_in'].shape[1]

    # Create decisions folder
    Path(os.path.join(experiment_path, "decisions", data_split)).mkdir(
        parents=True, exist_ok=True)
    save_directory = os.path.join(experiment_path, "decisions", data_split)

    for prediction_sample in range(prediction_horizon):
        # Create figure
        fig, axs = plt.subplots(len(decision_variables), figsize=(15, 10))
        fig.suptitle('Decision Variables')
        # Plot decision variables
        for i, (key, value) in enumerate(decision_variables.items()):
            # Subplot title
            axs[i].set_title(key)
            # Plot the decision variable
            axs[i].plot(value[:, prediction_sample])
            # Plot constraints
            constr = constraints[key]
            for j in range(len(constr)):
                # Plot horizontal line for grid and storage
                if type(constr[j]) == float:
                    axs[i].axhline(y=constr[j], color='r',
                                   linestyle='dashed')
                # Plot profile for renewable and load
                elif type(constr[j]) == torch.Tensor and data_split == "test":
                    axs[i].plot(constr[j], color='r',
                                linestyle='dashed')
            axs[i].grid()
        # Save figure
        plt.savefig(os.path.join(save_directory,
                    f"decision_variables_prediction_sample_{prediction_sample}.png"), bbox_inches='tight')
        plt.close()
        plt.clf()


def plot_trajectory(experiment_path: str,
                    state: torch.Tensor, 
                    constraints_state: Tuple[float, float],
                    controller_outputs: torch.Tensor,
                    constraints_input: Tuple[float, float],
                    t_initial: int, t_final: int,
                    save_folder_name: str = "trajectory") -> None:
    """This function is used to plot the state and controller output trajectories.

    Args:
        experiment_path (str): Path to the experiment folder.
        state (torch.Tensor): State trajectory.
        constraints_state (List): List of state constraints.
        controller_outputs (torch.Tensor): Outputs trajectory.
        constraints_input (List): List of input constraints. 
        t_initial (int): Initial time step to plot.
        t_final (int): Final time step to plot.
        folder_name (str): Name of the folder where to save the plots.
    """
    # Configure plot
    constraints_color = 'r'
    constraints_style = 'dashed'

    # Create trajectory folder
    Path(os.path.join(experiment_path, save_folder_name)).mkdir(
        parents=True, exist_ok=True)
    save_directory = os.path.join(experiment_path, save_folder_name)

    # Create figure
    fig, axs = plt.subplots(
        state.shape[1] + controller_outputs.shape[1], figsize=(15, 10))
    # State plot
    for i in range(state.shape[1]):
        axs[i].set_title('State')
        axs[i].plot(state[t_initial:t_final, i].cpu().numpy())
        for j in range(len(constraints_state)):
            axs[i].axhline(y=constraints_state[j],
                        color=constraints_color,
                        linestyle=constraints_style)
        axs[i].grid()
    # Input plot
    for i in range(controller_outputs.shape[1]):
        j = i + state.shape[1]
        axs[j].set_title('Input')
        axs[j].plot(controller_outputs[t_initial:t_final-1, i])
        for k in range(len(constraints_input[i])):
            axs[j].axhline(y=constraints_input[i][k],
                           color=constraints_color,
                           linestyle=constraints_style)
        axs[j].grid()
    # Save figure
    plt.savefig(os.path.join(save_directory,
                "state_and_input.png"), bbox_inches='tight')
    if save_folder_name != "trajectory":
        tikzplotlib.save(os.path.join(save_directory,"state_and_input.tex"))
    plt.close()


def total_cost(
    price: torch.Tensor,
    Pg: torch.Tensor,
    dt_hours: float = 0.25,
) -> float:
    """This function computes the total cost of the power generation.

    Args:
        price (torch.Tensor): electricity price in [€/MWh].
        Pg (torch.Tensor): power generation in [kW].
        dt_hours (float): duration of one sample in hours (HKUST: 0.25 h).

    Returns:
        float: total cost in [€].
    """
    if not np.isfinite(dt_hours) or dt_hours <= 0:
        raise ValueError("dt_hours must be a positive finite number")
    return -torch.sum(price * Pg).item() * float(dt_hours) / 1000


def plot_and_log_total_cost(experiment_path: str,
                            price: torch.Tensor,
                            Pg: torch.Tensor,
                            results: Dict[str, Any],
                            data_split: str,
                            prediction_horizon: int,
                            dt_hours: float = 0.25) -> None:
    """This function is used to plot the cost for all the prediction horizons.

    Args:
        experiment_path (str): Path to the experiment folder.
        price (torch.Tensor): Price.
        Pg (torch.Tensor): Power generation.
        results (Dict[str, Any]): Dictionary of results.
        data_split (str): Data split.
        prediction_horizon (int): Prediction horizon.
        dt_hours (float): duration of one sample in hours (HKUST: 0.25 h).
    """

    # Create cost folder
    Path(os.path.join(experiment_path, "cost", data_split)).mkdir(
        parents=True, exist_ok=True)
    save_directory = os.path.join(experiment_path, "cost", data_split)

    results['total_cost'] = {}
    total_costs = []
    logging.info("Total cost for data split %s:", data_split)
    for prediction_sample in range(prediction_horizon):
        try:
            results['total_cost'][prediction_sample] = total_cost(
                price, Pg[:, prediction_sample], dt_hours=dt_hours)
        except:
            results['total_cost'][prediction_sample] = 0
        total_costs.append(results['total_cost'][prediction_sample])
        logging.info(
            "Total cost for prediction sample %d: %f", prediction_sample, results['total_cost'][prediction_sample])

    # Create figure
    fig, axs = plt.subplots(1, figsize=(15, 10))
    fig.suptitle('Total Cost')
    x = np.arange(prediction_horizon)
    y = total_costs

    plt.plot(x, y)
    plt.xlabel('Prediction Sample')
    plt.ylabel('Total Cost [€]')
    plt.grid()
    plt.savefig(os.path.join(save_directory,
                f"total_cost.png"), bbox_inches='tight')
    plt.close()
    plt.clf()


def plot_different_runs_and_metrics(results_df: pd.DataFrame, experiment_path: str) -> None:
    """Plot each different run separately on the same graph with respect to the iteration and all the logged metrics."""
    # Get unique runs
    # Color each run differently
    unique_runs = results_df["trial_id"].unique()
    # Get all the metrics from the results dataframe which where the columns can be found in the METRICS_TENDENCY dictionary's keys
    unique_metrics = [
        column for column in results_df.columns if column in METRICS_TENDENCY]
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_runs)))
    for metric in unique_metrics:
        last_value = None
        best_id = None
        ymin = float("inf")
        ymax = float("-inf")
        for i, run in enumerate(unique_runs):
            run_df = results_df[results_df["trial_id"] == run]
            x = np.arange(0, len(run_df[metric].values))
            plt.plot(x, run_df[metric].values, color=colors[i])
            ymin = min(ymin, np.min(run_df[metric].values))
            ymax = max(ymax, np.max(run_df[metric].values))
            if last_value is None:
                last_value = run_df[metric].values[-1]
                best_id = run
            else:
                if METRICS_TENDENCY[metric] == "min":
                    if run_df[metric].values[-1] < last_value:
                        last_value = run_df[metric].values[-1]
                        best_id = run
                else:
                    if run_df[metric].values[-1] > last_value:
                        last_value = run_df[metric].values[-1]
                        best_id = run

        # Plot the best run with respect to black color
        best_run = results_df[results_df["trial_id"] == best_id]
        x = np.arange(0, len(best_run[metric].values))
        plt.plot(x, best_run[metric].values, color="black")
        plt.xlabel("Tuning step")
        plt.ylabel(metric)
        plt.grid()
        plt.ylim(ymin - (ymax - ymin) * 0.1, ymax + (ymax - ymin) * 0.1)
        plt.title(
            f"Individual runs, All runs: {len(unique_runs)}, Best run id: {best_id}", fontsize=10)
        plt.savefig(os.path.join(experiment_path, f"{metric}_individual_runs.png"),
                    bbox_inches="tight")
        plt.close()
        plt.clf()


# ---------------------------------------------------------------------------
# HKUST 15-minute closed-loop plots
# ---------------------------------------------------------------------------

def _require_frame(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")
    if frame.empty:
        raise ValueError(f"{name} must not be empty")
    return frame.copy()


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> np.ndarray:
    if column not in frame:
        return np.full(len(frame), default, dtype=float)
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Column {column!r} contains missing or non-finite values")
    return values


def _timestamps(frame: pd.DataFrame, column: str, name: str) -> pd.DatetimeIndex:
    _require_columns(frame, [column], name)
    values = pd.to_datetime(frame[column], errors="coerce")
    if values.isna().any():
        raise ValueError(f"{name}.{column} contains invalid timestamps")
    timestamps = pd.DatetimeIndex(values)
    if timestamps.has_duplicates:
        raise ValueError(f"{name}.{column} contains duplicate timestamps")
    if not timestamps.is_monotonic_increasing:
        raise ValueError(f"{name}.{column} must be chronological")
    return timestamps


def _step_hours(actual: pd.DataFrame, dt_hours: float) -> np.ndarray:
    if not np.isfinite(dt_hours) or dt_hours <= 0:
        raise ValueError("dt_hours must be a positive finite number")
    if "dt_hours" not in actual:
        return np.full(len(actual), float(dt_hours), dtype=float)
    observed = _numeric(actual, "dt_hours")
    if np.any(observed <= 0):
        raise ValueError("actual.dt_hours must be positive")
    if not np.allclose(observed, dt_hours, rtol=0.0, atol=1e-12):
        unique = np.unique(observed)
        raise ValueError(
            f"Requested dt_hours={dt_hours} disagrees with actual.dt_hours={unique.tolist()}"
        )
    return observed


def _first_step_plans(plans: pd.DataFrame, actual_length: int) -> pd.DataFrame:
    _require_columns(plans, ["decision_step", "horizon_step"], "plans")
    decision_step = pd.to_numeric(plans["decision_step"], errors="coerce")
    horizon_step = pd.to_numeric(plans["horizon_step"], errors="coerce")
    if decision_step.isna().any() or horizon_step.isna().any():
        raise ValueError("plans decision_step/horizon_step must be numeric")

    first = plans.loc[horizon_step == 0].copy()
    if first.empty:
        raise ValueError("plans contains no horizon_step == 0 rows")
    first["decision_step"] = pd.to_numeric(first["decision_step"]).astype(int)
    if first["decision_step"].duplicated().any():
        raise ValueError("plans contains duplicate first-step decisions")
    first = first.set_index("decision_step").reindex(range(actual_length))
    if first.index.size != actual_length or first.isna().all(axis=1).any():
        raise ValueError(
            "plans must contain one horizon_step=0 row for every actual execution row"
        )
    return first.reset_index(drop=False)


def _actual_energy(
    actual: pd.DataFrame,
    stem: str,
    step_hours: np.ndarray,
) -> np.ndarray:
    energy_column = f"{stem}_kwh"
    power_column = f"{stem}_kw"
    if energy_column in actual:
        return _numeric(actual, energy_column)
    if power_column in actual:
        return _numeric(actual, power_column) * step_hours
    return np.zeros(len(actual), dtype=float)


def _cost_components(actual: pd.DataFrame, step_hours: np.ndarray) -> Dict[str, np.ndarray]:
    if "grid_cost" in actual:
        grid = _numeric(actual, "grid_cost")
    else:
        grid = (
            _actual_energy(actual, "grid_import", step_hours)
            * _numeric(actual, "import_price_per_kwh")
            - _actual_energy(actual, "grid_export", step_hours)
            * _numeric(actual, "export_price_per_kwh")
        )
    return {
        "Grid net": grid,
        "Battery cycling": _numeric(actual, "battery_cost"),
        "Critical unmet": _numeric(actual, "critical_loss_load_cost"),
        "Non-critical unmet": _numeric(actual, "noncritical_loss_load_cost"),
        "PV curtailment": _numeric(actual, "pv_curtailment_cost"),
        "Overgeneration": _numeric(actual, "overgeneration_cost"),
    }


def _format_time_axis(axis: plt.Axes) -> None:
    locator = mdates.AutoDateLocator(minticks=4, maxticks=10)
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    axis.grid(True, alpha=0.28)


def _signed_flow_plot(
    axis: plt.Axes,
    x: pd.DatetimeIndex,
    planned_in: np.ndarray,
    planned_out: np.ndarray,
    actual_in: np.ndarray,
    actual_out: np.ndarray,
    in_label: str,
    out_label: str,
) -> None:
    """Plot incoming flow positive and outgoing flow negative."""

    axis.step(
        x,
        planned_in,
        where="post",
        color="#2f855a",
        linestyle="--",
        linewidth=1.1,
        label=f"Plan {in_label}",
    )
    axis.step(
        x,
        -planned_out,
        where="post",
        color="#c05621",
        linestyle="--",
        linewidth=1.1,
        label=f"Plan {out_label}",
    )
    axis.step(
        x,
        actual_in,
        where="post",
        color="#2f855a",
        linewidth=1.8,
        label=f"Actual {in_label}",
    )
    axis.step(
        x,
        -actual_out,
        where="post",
        color="#c05621",
        linewidth=1.8,
        label=f"Actual {out_label}",
    )
    axis.axhline(0.0, color="#4a5568", linewidth=0.7)


def _save_figure(fig: plt.Figure, path: Path, dpi: int) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_closed_loop_overview(
    actual: pd.DataFrame,
    first_plan: pd.DataFrame,
    output_path: Path,
    *,
    step_hours: np.ndarray,
    currency: str,
    dpi: int,
) -> None:
    x = _timestamps(actual, "timestamp", "actual")
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
    axes = axes.ravel()

    # Forecast and observed energy-system inputs.
    axes[0].plot(
        x,
        _numeric(first_plan, "pv_forecast_kw"),
        color="#2b6cb0",
        linestyle="--",
        linewidth=1.4,
        label="Plan PV forecast",
    )
    axes[0].plot(
        x,
        _numeric(actual, "pv_available_kw"),
        color="#d69e2e",
        linewidth=2.0,
        label="Actual PV",
    )
    axes[0].plot(
        x,
        _numeric(actual, "load_kw"),
        color="#4a5568",
        linewidth=1.3,
        label="Actual load",
    )
    axes[0].set_title("PV forecast and actual supply/demand")
    axes[0].set_ylabel("Power [kW]")

    # Planned first-step SOC versus simulator feedback.
    planned_soc = (
        _numeric(first_plan, "soc_end")
        if "soc_end" in first_plan
        else _numeric(first_plan, "soc")
    )
    axes[1].step(
        x,
        planned_soc,
        where="post",
        color="#2b6cb0",
        linestyle="--",
        linewidth=1.3,
        label="Plan SOC after step",
    )
    axes[1].step(
        x,
        _numeric(actual, "soc_after_kwh"),
        where="post",
        color="#1a202c",
        linewidth=2.0,
        label="Actual SOC after step",
    )
    axes[1].set_title("Battery state feedback")
    axes[1].set_ylabel("SOC [kWh]")

    _signed_flow_plot(
        axes[2],
        x,
        _numeric(first_plan, "charge"),
        _numeric(first_plan, "discharge"),
        _numeric(actual, "battery_charge_kw"),
        _numeric(actual, "battery_discharge_kw"),
        "charge",
        "discharge",
    )
    axes[2].set_title("Battery action: positive charge, negative discharge")
    axes[2].set_ylabel("Power [kW]")

    _signed_flow_plot(
        axes[3],
        x,
        _numeric(first_plan, "grid_import"),
        _numeric(first_plan, "grid_export"),
        _numeric(actual, "grid_import_kw"),
        _numeric(actual, "grid_export_kw"),
        "import",
        "export",
    )
    axes[3].set_title("Grid exchange: positive import, negative export")
    axes[3].set_ylabel("Power [kW]")

    # Resilience outcomes. Plans are dashed, executed outcomes are solid.
    resilience = (
        ("pv_curtailment", "pv_curtailed_kw", "PV curtailed", "#d69e2e"),
        ("unmet_critical", "critical_loss_load_kw", "Critical unmet", "#c53030"),
        (
            "unmet_noncritical",
            "noncritical_loss_load_kw",
            "Non-critical unmet",
            "#805ad5",
        ),
    )
    for plan_column, actual_column, label, color in resilience:
        axes[4].step(
            x,
            _numeric(first_plan, plan_column),
            where="post",
            color=color,
            linestyle="--",
            linewidth=1.1,
            label=f"Plan {label.lower()}",
        )
        axes[4].step(
            x,
            _numeric(actual, actual_column),
            where="post",
            color=color,
            linewidth=1.8,
            label=f"Actual {label.lower()}",
        )
    axes[4].set_title("Curtailment and unmet load")
    axes[4].set_ylabel("Power [kW]")

    components = _cost_components(actual, step_hours)
    step_cost = np.sum(np.vstack(list(components.values())), axis=0)
    axes[5].bar(
        x,
        step_cost,
        width=step_hours / 24.0 * 0.82,
        color=np.where(step_cost >= 0.0, "#718096", "#38a169"),
        alpha=0.65,
        label="Actual step cost",
    )
    cumulative_axis = axes[5].twinx()
    cumulative_axis.plot(
        x,
        np.cumsum(step_cost),
        color="#1a202c",
        linewidth=1.8,
        label="Actual cumulative cost",
    )
    axes[5].set_title("Realised operating cost")
    axes[5].set_ylabel(f"Step cost [{currency}]")
    cumulative_axis.set_ylabel(f"Cumulative [{currency}]")
    handles, labels = axes[5].get_legend_handles_labels()
    handles_2, labels_2 = cumulative_axis.get_legend_handles_labels()
    axes[5].legend(handles + handles_2, labels + labels_2, loc="best", fontsize=8)

    for axis in axes:
        _format_time_axis(axis)
        if axis is not axes[5]:
            axis.legend(loc="best", fontsize=7, ncol=2)
    axes[-2].set_xlabel("Execution time")
    axes[-1].set_xlabel("Execution time")
    fig.suptitle(
        "HKUST PV-LSTM + MPC closed loop\n"
        "dashed = first-step plan; solid = actual Pymgrid execution",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    _save_figure(fig, output_path, dpi)


def _plot_energy_cost_summary(
    actual: pd.DataFrame,
    output_path: Path,
    *,
    step_hours: np.ndarray,
    currency: str,
    dpi: int,
) -> None:
    energy_stems = (
        ("pv_available", "PV available"),
        ("pv_used", "PV used"),
        ("load", "Load"),
        ("battery_charge", "Battery charge"),
        ("battery_discharge", "Battery discharge"),
        ("grid_import", "Grid import"),
        ("grid_export", "Grid export"),
        ("pv_curtailed", "PV curtailed"),
        ("critical_loss_load", "Critical unmet"),
        ("noncritical_loss_load", "Non-critical unmet"),
    )
    energy_labels = [label for _, label in energy_stems]
    energy_values = [
        float(_actual_energy(actual, stem, step_hours).sum())
        for stem, _ in energy_stems
    ]
    component_series = _cost_components(actual, step_hours)
    cost_labels = list(component_series)
    cost_values = [float(values.sum()) for values in component_series.values()]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    colors = [
        "#d69e2e",
        "#ecc94b",
        "#4a5568",
        "#2f855a",
        "#c05621",
        "#c53030",
        "#2b6cb0",
        "#b7791f",
        "#9b2c2c",
        "#805ad5",
    ]
    positions = np.arange(len(energy_labels))
    axes[0].barh(positions, energy_values, color=colors)
    axes[0].set_yticks(positions, labels=energy_labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Energy [kWh]")
    axes[0].set_title("Actual executed energy totals")
    axes[0].grid(axis="x", alpha=0.28)
    for position, value in zip(positions, energy_values):
        axes[0].text(value, position, f" {value:.2f}", va="center", fontsize=8)

    cost_positions = np.arange(len(cost_labels))
    cost_colors = ["#c53030" if value >= 0 else "#38a169" for value in cost_values]
    axes[1].barh(cost_positions, cost_values, color=cost_colors)
    axes[1].set_yticks(cost_positions, labels=cost_labels)
    axes[1].invert_yaxis()
    axes[1].axvline(0.0, color="#4a5568", linewidth=0.7)
    axes[1].set_xlabel(f"Cost [{currency}]")
    axes[1].set_title(
        f"Actual cost components (total {sum(cost_values):.2f} {currency})"
    )
    axes[1].grid(axis="x", alpha=0.28)
    for position, value in zip(cost_positions, cost_values):
        horizontal = "left" if value >= 0 else "right"
        axes[1].text(value, position, f" {value:.3f} ", va="center", ha=horizontal, fontsize=8)

    fig.suptitle("Closed-loop energy and realised cost summary", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save_figure(fig, output_path, dpi)


def plot_mpc_plan_vs_actual(
    actual: pd.DataFrame,
    plans: pd.DataFrame,
    output_path: Union[str, Path],
    *,
    decision_step: int = 0,
    dt_hours: float = 0.25,
    dpi: int = 180,
) -> Path:
    """Plot one complete MPC horizon and any overlapping actual execution.

    Planned quantities use dashed lines. Actual Pymgrid values use solid lines
    and appear only where the closed-loop execution covers the horizon.
    ``dt_hours`` is validated against ``actual.dt_hours`` so energy labels
    cannot silently use an hourly assumption.
    """

    actual = _require_frame(actual, "actual")
    plans = _require_frame(plans, "plans")
    _step_hours(actual, dt_hours)
    _require_columns(
        actual,
        [
            "timestamp",
            "pv_available_kw",
            "load_kw",
            "soc_after_kwh",
            "battery_charge_kw",
            "battery_discharge_kw",
            "grid_import_kw",
            "grid_export_kw",
            "pv_curtailed_kw",
            "critical_loss_load_kw",
            "noncritical_loss_load_kw",
        ],
        "actual",
    )
    _require_columns(
        plans,
        ["decision_step", "horizon_step", "horizon_timestamp", "pv_forecast_kw"],
        "plans",
    )
    decision_values = pd.to_numeric(plans["decision_step"], errors="coerce")
    plan = plans.loc[decision_values == decision_step].copy()
    if plan.empty:
        raise ValueError(f"plans contains no decision_step={decision_step}")
    plan = plan.sort_values("horizon_step")
    x = _timestamps(plan, "horizon_timestamp", "plans")
    actual_x = _timestamps(actual, "timestamp", "actual")
    actual_lookup = actual.copy()
    actual_lookup.index = actual_x
    overlapping = actual_lookup.reindex(x)

    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    axes[0].plot(
        x,
        _numeric(plan, "pv_forecast_kw"),
        color="#2b6cb0",
        linestyle="--",
        label="Plan PV forecast",
    )
    axes[0].plot(
        x,
        _numeric(plan, "load"),
        color="#718096",
        linestyle="--",
        label="Plan load input",
    )
    if overlapping["pv_available_kw"].notna().any():
        axes[0].plot(
            x,
            pd.to_numeric(overlapping["pv_available_kw"], errors="coerce"),
            color="#d69e2e",
            linewidth=1.8,
            marker="o",
            markersize=2.4,
            label="Actual PV",
        )
        axes[0].plot(
            x,
            pd.to_numeric(overlapping["load_kw"], errors="coerce"),
            color="#1a202c",
            linewidth=1.4,
            marker="o",
            markersize=2.2,
            label="Actual load",
        )
    axes[0].set_ylabel("Power [kW]")
    axes[0].set_title("Forecast horizon inputs")

    planned_soc = _numeric(plan, "soc_end" if "soc_end" in plan else "soc")
    axes[1].step(
        x,
        planned_soc,
        where="post",
        color="#2b6cb0",
        linestyle="--",
        label="Plan SOC",
    )
    if "soc_after_kwh" in overlapping and overlapping["soc_after_kwh"].notna().any():
        axes[1].step(
            x,
            pd.to_numeric(overlapping["soc_after_kwh"], errors="coerce"),
            where="post",
            color="#1a202c",
            linewidth=1.8,
            marker="o",
            markersize=2.4,
            label="Actual SOC",
        )
    axes[1].set_ylabel("SOC [kWh]")
    axes[1].set_title("Planned trajectory and simulator feedback")

    axes[2].step(
        x,
        _numeric(plan, "charge"),
        where="post",
        color="#2f855a",
        linestyle="--",
        label="Plan charge",
    )
    axes[2].step(
        x,
        -_numeric(plan, "discharge"),
        where="post",
        color="#c05621",
        linestyle="--",
        label="Plan discharge",
    )
    if "battery_charge_kw" in overlapping:
        axes[2].step(
            x,
            pd.to_numeric(overlapping["battery_charge_kw"], errors="coerce"),
            where="post",
            color="#2f855a",
            linewidth=1.7,
            marker="o",
            markersize=2.2,
            label="Actual charge",
        )
        axes[2].step(
            x,
            -pd.to_numeric(overlapping["battery_discharge_kw"], errors="coerce"),
            where="post",
            color="#c05621",
            linewidth=1.7,
            marker="o",
            markersize=2.2,
            label="Actual discharge",
        )
    axes[2].axhline(0.0, color="#4a5568", linewidth=0.7)
    axes[2].set_ylabel("Power [kW]")
    axes[2].set_title("Battery: positive charge, negative discharge")

    for plan_column, label, color, sign in (
        ("grid_import", "Plan grid import", "#c53030", 1.0),
        ("grid_export", "Plan grid export", "#2b6cb0", -1.0),
        ("pv_curtailment", "Plan PV curtailed", "#d69e2e", 1.0),
        ("unmet_critical", "Plan critical unmet", "#9b2c2c", 1.0),
        ("unmet_noncritical", "Plan non-critical unmet", "#805ad5", 1.0),
    ):
        axes[3].step(
            x,
            sign * _numeric(plan, plan_column),
            where="post",
            color=color,
            linestyle="--",
            linewidth=1.1,
            label=label,
        )
    for actual_column, label, color, sign in (
        ("grid_import_kw", "Actual grid import", "#c53030", 1.0),
        ("grid_export_kw", "Actual grid export", "#2b6cb0", -1.0),
        ("pv_curtailed_kw", "Actual PV curtailed", "#d69e2e", 1.0),
        ("critical_loss_load_kw", "Actual critical unmet", "#9b2c2c", 1.0),
        (
            "noncritical_loss_load_kw",
            "Actual non-critical unmet",
            "#805ad5",
            1.0,
        ),
    ):
        axes[3].step(
            x,
            sign * pd.to_numeric(overlapping[actual_column], errors="coerce"),
            where="post",
            color=color,
            linewidth=1.7,
            marker="o",
            markersize=2.2,
            label=label,
        )
    axes[3].axhline(0.0, color="#4a5568", linewidth=0.7)
    axes[3].set_ylabel("Power [kW]")
    axes[3].set_title("Planned grid and resilience actions")
    axes[3].set_xlabel("Horizon time")

    for axis in axes:
        _format_time_axis(axis)
        axis.legend(loc="best", fontsize=8, ncol=2)
    fig.suptitle(
        f"MPC plan at decision step {decision_step}: dashed plan vs solid execution",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    _save_figure(fig, output, dpi)
    return output


def plot_closed_loop_results(
    actual: pd.DataFrame,
    plans: pd.DataFrame,
    output_dir: Union[str, Path],
    *,
    dt_hours: float = 0.25,
    currency: str = "CNY",
    prefix: str = "closed_loop",
    plan_decision_step: Optional[int] = 0,
    dpi: int = 180,
) -> Dict[str, Path]:
    """Create the standard HKUST closed-loop result figures.

    Parameters
    ----------
    actual:
        One row per action actually executed by ``PymgridAdapter``.
    plans:
        Long-form MPC plans from ``OnlineLearner.run_closed_loop``. The
        overview compares ``horizon_step == 0`` with each executed action.
    output_dir:
        Destination directory. No GUI window is opened.
    dt_hours:
        Duration of one control interval. The HKUST default is 0.25 h and is
        checked against the adapter's ``actual.dt_hours`` column.
    currency:
        Label for cost axes; the default tariff is CNY/kWh.
    plan_decision_step:
        Also render one complete horizon. Set to ``None`` to skip that figure.

    Returns
    -------
    Dict[str, pathlib.Path]
        Paths keyed by ``overview``, ``energy_cost_summary`` and, when
        requested, ``plan_horizon``.
    """

    actual = _require_frame(actual, "actual")
    plans = _require_frame(plans, "plans")
    _require_columns(
        actual,
        [
            "timestamp",
            "pv_available_kw",
            "load_kw",
            "soc_after_kwh",
            "battery_charge_kw",
            "battery_discharge_kw",
            "grid_import_kw",
            "grid_export_kw",
            "pv_curtailed_kw",
            "critical_loss_load_kw",
            "noncritical_loss_load_kw",
        ],
        "actual",
    )
    step_hours = _step_hours(actual, dt_hours)
    first_plan = _first_step_plans(plans, len(actual))
    _require_columns(
        first_plan,
        [
            "pv_forecast_kw",
            "charge",
            "discharge",
            "grid_import",
            "grid_export",
            "pv_curtailment",
            "unmet_critical",
            "unmet_noncritical",
        ],
        "first-step plans",
    )

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "overview": output / f"{prefix}_overview.png",
        "energy_cost_summary": output / f"{prefix}_energy_cost_summary.png",
    }
    _plot_closed_loop_overview(
        actual,
        first_plan,
        paths["overview"],
        step_hours=step_hours,
        currency=currency,
        dpi=dpi,
    )
    _plot_energy_cost_summary(
        actual,
        paths["energy_cost_summary"],
        step_hours=step_hours,
        currency=currency,
        dpi=dpi,
    )
    if plan_decision_step is not None:
        paths["plan_horizon"] = output / (
            f"{prefix}_plan_step_{plan_decision_step:03d}.png"
        )
        plot_mpc_plan_vs_actual(
            actual,
            plans,
            paths["plan_horizon"],
            decision_step=plan_decision_step,
            dt_hours=dt_hours,
            dpi=dpi,
        )
    logging.info("Saved closed-loop figures: %s", paths)
    return paths
