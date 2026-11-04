"""Shared, quality-filtered M1/M2 episodes from the existing scenario template.

Checkpoint-declared chronological windows are used for training and for
validation. PV values and their original M0 scaler are never changed. Episode
SOC is initialized at the cropped episode start, not at the month's start.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))
from m0_scenarios import DT, H, TIERS, Scenario, load_cases

TRAIN_START = pd.Timestamp("2021-07-01")
TRAIN_END = pd.Timestamp("2023-09-01")
VAL_END = pd.Timestamp("2023-11-01")
PROTOCOL_VERSION = "task-training-episodes-v2-multiyear"
LEAD_HOURS = 24
RECOVERY_HOURS = 24


@dataclass
class TrainingEpisode:
    name: str
    split: str
    frame: pd.DataFrame
    histories: torch.Tensor
    case: Scenario
    steps: int
    metadata: dict[str, Any]


@dataclass
class TrainingDataBundle:
    episodes: dict[str, list[TrainingEpisode]]
    metadata: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _indexed(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    frame = frame.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="raise")
    index = pd.DatetimeIndex(frame["datetime"])
    if index.tz is not None or index.hasnans or index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError(f"{label} requires ordered, unique local timestamps")
    return frame.set_index("datetime")


def _require_coverage(frame: pd.DataFrame, label: str, start: pd.Timestamp,
                      end: pd.Timestamp) -> None:
    """Missing years are an input error, not quality-filtered episodes."""
    expected = pd.date_range(start, end, freq="15min", inclusive="left")
    actual = frame.loc[start:end - pd.Timedelta(minutes=15)].index
    if not actual.equals(expected):
        missing = expected.difference(actual)
        extra = actual.difference(expected)
        raise ValueError(
            f"{label} must cover the complete 15-minute interval [{start}, {end}); "
            f"missing {len(missing)} timestamps, off-grid {len(extra)}; "
            f"first missing: {str(missing[0]) if len(missing) else 'none'}. "
            "Prepare multiyear inputs; refusing to silently shorten training."
        )


def validate_checkpoint_splits(config: dict) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """Use the checkpoint's chronological boundaries, never historical constants."""
    windows = config.get("split_windows", {})
    parsed = {}
    for name in ("train", "val"):
        window = windows.get(name, {})
        if window.get("start") is None or window.get("end") is None:
            raise ValueError("Checkpoint must declare complete train/val dates")
        start, end = pd.Timestamp(window["start"]), pd.Timestamp(window["end"])
        if (pd.isna(start) or pd.isna(end) or start.tz is not None or end.tz is not None
                or start >= end or start != start.to_period("M").start_time
                or end != end.to_period("M").start_time):
            raise ValueError("Task split dates must be nonempty local calendar-month boundaries")
        parsed[name] = (start, end)
    if parsed["train"][1] != parsed["val"][0]:
        raise ValueError("Task train/val intervals must be contiguous and nonoverlapping")
    if "test" in windows:
        test_start, test_end = (pd.Timestamp(windows["test"][key]) for key in ("start", "end"))
        if (pd.isna(test_start) or pd.isna(test_end) or test_start.tz is not None
                or test_end.tz is not None or test_start < parsed["val"][1] or test_end <= test_start):
            raise ValueError("Test must follow train/val without overlap")
    return parsed["train"][0], parsed["train"][1], parsed["val"][1]


def prepare_training_data(dataset: Any, scenario_config_path: str | Path,
                          dispatch_path: str | Path, tiers_path: str | Path) -> TrainingDataBundle:
    """Build a deterministic manifest and episodes; never run a model or solver.

    Each frame has ``steps + 95`` rows; histories additionally contains the
    original lookback before its first row. At decision i, use
    ``histories[i:i+lookback]``. Forecast inputs must use ``case.information``
    for grid availability; frame.grid_available is the realised environment.
    """
    paths = {"scenario_config": Path(scenario_config_path), "dispatch": Path(dispatch_path),
             "load_tiers": Path(tiers_path), "pv_timeseries": Path(dataset.config.data_path)}
    config = json.loads(paths["scenario_config"].read_text())
    templates = load_cases(config)
    lookback = int(dataset.lookback_window)
    if int(dataset.prediction_horizon) != H or lookback < 1:
        raise ValueError("Task episodes require the existing 96-step horizon and a positive lookback")
    train_start, train_end, val_end = validate_checkpoint_splits({"split_windows": {
        name: {"start": window.start, "end": window.end}
        for name, window in dataset.config.split_windows.items()
    }})
    minimum, maximum = float(dataset.pv_min), float(dataset.pv_max)
    if not np.isfinite([minimum, maximum]).all() or maximum <= minimum:
        raise ValueError("Invalid original M0 PV scaler")

    # Slice before any selection or energy calculation. Historical PV may
    # precede July 2021; execution and all prediction targets cannot precede it.
    history_start = train_start - pd.Timedelta(minutes=15 * lookback)
    pv = _indexed(dataset.source_frame, "PV").loc[history_start:val_end - pd.Timedelta(minutes=15)]
    if "pv" not in pv and "pv_kW" in pv:
        pv = pv.rename(columns={"pv_kW": "pv"})
    required = {"pv", "history_valid", "pv_target_valid", "split"}
    if not required.issubset(pv):
        raise ValueError(f"Missing PV columns: {sorted(required - set(pv.columns))}")
    _require_coverage(pv, "PV history and targets", history_start, val_end)
    dispatch_columns = ["datetime", "load_kW", "import_price", "export_price", "grid_available", "load_scenario_valid"]
    tier_columns = ["datetime", "load_kW"] + ["minimum_operation_" + k.upper() + "_kW" for k in TIERS]
    dispatch = _indexed(pd.read_csv(paths["dispatch"], usecols=dispatch_columns), "Dispatch")
    tiers = _indexed(pd.read_csv(paths["load_tiers"], usecols=tier_columns), "Load tiers")
    _require_coverage(dispatch, "Dispatch", train_start, val_end)
    _require_coverage(tiers, "Load tiers", train_start, val_end)
    dispatch = dispatch.loc[train_start:val_end - pd.Timedelta(minutes=15)]
    tiers = tiers.loc[train_start:val_end - pd.Timedelta(minutes=15)]
    dispatch = dispatch.rename(columns={"load_kW": "load"})
    tiers = tiers.rename(columns={"minimum_operation_" + k.upper() + "_kW": k for k in TIERS})
    base = dispatch.join(tiers[["load_kW", *TIERS]], how="outer")
    base = base.join(pv[["pv", "split"]], how="outer").loc[train_start:val_end - pd.Timedelta(minutes=15)]
    episodes: dict[str, list[TrainingEpisode]] = {"train": [], "val": []}
    selections = []

    def candidate(event: pd.Timestamp, template: Scenario, month: pd.Timestamp,
                  next_month: pd.Timestamp, split: str) -> tuple[dict[str, Any], pd.DataFrame | None, pd.DataFrame | None]:
        duration = max(template.outage_hours, template.announced_hours, template.selection_hours) or 24
        start = event - pd.Timedelta(hours=LEAD_HOURS)
        end = event + pd.Timedelta(hours=duration + RECOVERY_HOURS)
        steps = int((end - start) / pd.Timedelta(minutes=15))
        axis = pd.date_range(start, periods=steps + H - 1, freq="15min")
        record = {"event_start": str(event), "episode_start": str(start), "episode_end_exclusive": str(end),
                  "last_forecast_target": str(axis[-1]), "steps": steps, "rejection_reasons": []}
        reasons = record["rejection_reasons"]
        lower, upper = (train_start, train_end) if split == "train" else (train_end, val_end)
        if start < month or axis[-1] >= next_month:
            reasons.append("execution_or_last_forecast_crosses_month")
        if start < lower or axis[-1] >= upper:
            reasons.append("execution_or_last_forecast_crosses_split")
        if reasons:
            return record, None, None
        history_axis = pd.date_range(start - pd.Timedelta(minutes=15 * lookback), axis[-1], freq="15min")
        history = pv.reindex(history_axis)
        power = pd.to_numeric(history.pv, errors="coerce")
        good_pv = history.history_valid.eq(1) & history.pv_target_valid.eq(1) & np.isfinite(power) & power.ge(0)
        if not good_pv.all():
            reasons.append("invalid_or_missing_pv_history_or_target")
            record["invalid_pv_points"] = int((~good_pv).sum())
        frame = base.reindex(axis)
        numeric = frame[["pv", "load", "load_kW", *TIERS, "import_price", "export_price", "grid_available"]].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(numeric.to_numpy()).all() or (numeric.to_numpy() < 0).any():
            reasons.append("invalid_or_missing_dispatch_or_tiers")
        if not frame.load_scenario_valid.eq(1).all():
            reasons.append("invalid_load_scenario")
            record["invalid_load_points"] = int((~frame.load_scenario_valid.eq(1)).sum())
        if not frame.grid_available.eq(1).all():
            reasons.append("source_grid_not_connected")
        if not frame["split"].eq(split).all():
            reasons.append("pv_split_labels_mismatch")
        if not (np.allclose(numeric.load, numeric.load_kW, rtol=0, atol=1e-6)
                and np.allclose(numeric[list(TIERS)].sum(axis=1), numeric.load, rtol=0, atol=1e-6)):
            reasons.append("tier_load_sum_mismatch")
        if reasons:
            return record, None, None
        event_end = event + pd.Timedelta(hours=template.selection_hours or duration)
        event_rows = frame.loc[(frame.index >= event) & (frame.index < event_end)]
        record.update(pv_kwh=float(event_rows.pv.sum() * DT), load_kwh=float(event_rows.load.sum() * DT))
        return record, frame, history

    for month in pd.date_range(train_start, val_end, freq="MS", inclusive="left"):
        next_month = month + pd.offsets.MonthBegin(1)
        split = "train" if month < train_end else "val"
        for template in templates:
            name = f"{month:%Y-%m}_{template.name}"
            low = template.pv_regime == "observed_low"
            nominal = month + pd.Timedelta(hours=template.event_hour)
            clock = template.selection_clock if low else template.event_hour % 24
            starts = pd.date_range(month + pd.Timedelta(hours=clock), next_month, freq="1d", inclusive="left")
            checked = [candidate(t, template, month, next_month, split) for t in starts]
            valid = [entry for entry in checked if not entry[0]["rejection_reasons"]]
            nominal_record = next((entry[0] for entry in checked if entry[0]["event_start"] == str(nominal)), None)
            nominal_reasons = (nominal_record["rejection_reasons"] if nominal_record is not None
                               else ["nominal_event_outside_month"])
            method = ("minimum_contiguous_pv_energy_fixed_clock_earliest_tie" if low
                      else "nominal_else_nearest_quality_valid_same_clock_earlier_tie")
            selection = {"name": name, "template_name": template.name, "split": split, "month": f"{month:%Y-%m}",
                         "method": method, "selection_method": method,
                         "candidate_count": len(checked), "eligible_candidate_count": len(valid),
                         "candidates": [entry[0] for entry in checked]}
            if not low:
                selection.update(nominal_event_start=str(nominal), nominal_rejection_reasons=nominal_reasons)
            if not valid:
                selection.update(status="excluded", exclusion_reason="no_complete_quality_valid_episode")
                selections.append(selection)
                continue
            if low:
                chosen = min(valid, key=lambda entry: (entry[0]["pv_kwh"], entry[0]["event_start"]))
            else:
                chosen = min(valid, key=lambda entry: (
                    abs(pd.Timestamp(entry[0]["event_start"]) - nominal), entry[0]["event_start"]))
            record, frame, history = chosen
            case = replace(template, name=name, event_hour=float(LEAD_HOURS))
            case.validate(record["steps"] * DT / 24)
            frame = frame.drop(columns=["load_kW", "split"]).copy()
            frame["critical_load"] = frame.cl1 + frame.cl2
            frame["grid_available"] = case.actual_grid(np.arange(len(frame)) * DT)
            raw = torch.tensor(history.pv.to_numpy(dtype=np.float32), dtype=torch.float32).reshape(-1, 1)
            histories = (raw - minimum) / (maximum - minimum)
            metadata = {**record, "name": name, "template_name": template.name, "split": split,
                        "month": f"{month:%Y-%m}", "case": asdict(case), "template_case": asdict(template),
                        "selection_method": method,
                        "initial_soc_meaning": "Fraction at cropped episode start, 24h before the event; not month-start test SOC.",
                        "pv_repairs": 0}
            if not low:
                shift_hours = (pd.Timestamp(record["event_start"]) - nominal).total_seconds() / 3600
                resolution = {"nominal_event_start": str(nominal), "resolved_event_start": record["event_start"],
                              "shift_hours": shift_hours, "nominal_rejection_reasons": nominal_reasons,
                              "reason": "nominal_quality_valid" if shift_hours == 0 else "nominal_invalid_use_nearest_quality_valid_same_clock"}
                metadata.update(resolution)
                selection.update(resolution)
            episodes[split].append(TrainingEpisode(name, split, frame, histories, case, record["steps"], metadata))
            selection.update(status="included", selected=metadata)
            selections.append(selection)
    monthly_coverage = []
    for month in pd.date_range(train_start, val_end, freq="MS", inclusive="left"):
        month_name = f"{month:%Y-%m}"
        rows = [row for row in selections if row["month"] == month_name]
        included = [row["selected"] for row in rows if row["status"] == "included"]
        monthly_coverage.append({"month": month_name, "split": "train" if month < train_end else "val",
                                 "requested": len(rows), "included": len(included),
                                 "excluded": len(rows) - len(included),
                                 "steps": sum(row["steps"] for row in included)})
    for key in ("dispatch", "load_tiers"):
        source_manifest = paths[key].parent / "manifest.json"
        if source_manifest.is_file():
            paths[key + "_manifest"] = source_manifest
    manifest = {"protocol_version": PROTOCOL_VERSION, "source_scenario_protocol": config["protocol_version"],
                "sources": {key: {"path": str(path.resolve()), "sha256": _sha256(path)} for key, path in paths.items()},
                "split_intervals": {"train": [str(train_start), str(train_end)], "val": [str(train_end), str(val_end)]},
                "lookback_steps": lookback, "horizon_steps": H, "dt_hours": DT,
                "pv_scaler": {"min": minimum, "max": maximum, "policy": "unchanged M0 checkpoint/dataset scaler"},
                "episode_rule": {"lead_hours": LEAD_HOURS, "recovery_hours": RECOVERY_HOURS,
                                 "ordinary_grid_duration_hours": 24, "initial_soc": "template fraction at episode start"},
                "information_rule": config["information_rule"],
                "quality_rule": "All history, execution and final 96-step planning PV must pass both quality flags; valid load scenario and tier sums; no repair, scaling, interpolation or time remapping.",
                "selection_rule": "Execution and all forecast targets stay in one calendar month and split; historical PV may precede the month. Low-PV cases select minimum actual PV energy. Original cases prefer their nominal monthly date, then the nearest complete quality-valid date at the same clock, earlier on ties, without ranking PV. This fallback applies only to training/validation construction, not the test scenario configuration. Event selection never accesses test PV.",
                "template_count": len(templates), "episode_counts": {key: len(value) for key, value in episodes.items()},
                "requested_episode_counts": {key: sum(row["requested"] for row in monthly_coverage if row["split"] == key) for key in episodes},
                "monthly_coverage": monthly_coverage,
                "excluded_counts": {key: sum(x["split"] == key and x["status"] == "excluded" for x in selections) for key in episodes},
                "selections": selections}
    if any(not values for values in episodes.values()):
        raise ValueError("At least one complete quality-valid episode is required in both train and val")
    return TrainingDataBundle(episodes, manifest)


def write_event_tables(bundle: TrainingDataBundle, directory: str | Path) -> None:
    """Write both the full audit trail and a compact, readable event table."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "event_manifest.json").write_text(
        json.dumps(bundle.metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    rows = []
    for selection in bundle.metadata["selections"]:
        chosen = selection.get("selected", {})
        case = chosen.get("case", {})
        rows.append({"name": selection["name"], "split": selection["split"],
                     "month": selection["month"], "status": selection["status"],
                     "group": case.get("group", ""), "template": selection["template_name"],
                     "event_start": chosen.get("event_start", ""),
                     "episode_start": chosen.get("episode_start", ""),
                     "episode_end_exclusive": chosen.get("episode_end_exclusive", ""),
                     "last_forecast_target": chosen.get("last_forecast_target", ""),
                     "steps": chosen.get("steps", ""), "outage_hours": case.get("outage_hours", ""),
                     "pv_regime": case.get("pv_regime", ""), "pv_kWh": chosen.get("pv_kwh", ""),
                     "load_kWh": chosen.get("load_kwh", ""),
                     "eligible_candidates": selection["eligible_candidate_count"],
                     "selection_method": selection["selection_method"],
                     "exclusion_reason": selection.get("exclusion_reason", "")})
    pd.DataFrame(rows).to_csv(directory / "events.csv", index=False)
    pd.DataFrame(bundle.metadata["monthly_coverage"]).to_csv(directory / "monthly_coverage.csv", index=False)
