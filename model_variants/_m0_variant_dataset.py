"""Leakage-safe HKUST 15-minute dataset for PV forecasting and MPC.

The recurrent model sees the preceding 96 quarter-hours and predicts the next
quarter-hour only. This isolated copy preserves the original H=96 loader.
Future load, electricity prices, grid availability and critical load remain in
physical units and bypass the neural network as explicit MPC parameters.

Split convention
----------------
Each sample is assigned by its *complete forecast target window*.  For example,
a validation sample is accepted only when its single target timestamp lies inside
the configured validation interval.  Its lookback may use preceding training
observations, which is information available at that validation timestamp.
The min/max scaler is fitted once, using only valid rows in the training interval.
Quality flags exclude whole forecast samples without deleting source timestamps.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from configurations.hkust_15min import DEFAULT_CONFIG as BASE_CONFIG, HKUSTConfig, SplitWindow

DEFAULT_CONFIG = replace(BASE_CONFIG, lookback_steps=96, prediction_horizon=1)


TensorDict = Dict[str, Any]


@dataclass(frozen=True)
class _PreparedData:
    frame: pd.DataFrame
    history_raw: torch.Tensor
    history_normalised: torch.Tensor
    history_min: torch.Tensor
    history_max: torch.Tensor
    unix_seconds: torch.Tensor
    anchors_by_split: Mapping[str, np.ndarray]
    generated_columns: Mapping[str, str]
    scaler_start: pd.Timestamp
    scaler_end: pd.Timestamp
    quality_summary: Mapping[str, Any]


def _as_path(value: Union[str, Path]) -> Path:
    return Path(value).expanduser().resolve()


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{label} CSV does not exist: {path}")
    try:
        return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - pandas supplies the detail
        raise ValueError(f"Could not read {label} CSV {path}: {exc}") from exc


def _parse_timestamps(
    frame: pd.DataFrame,
    timestamp_column: str,
    label: str,
) -> pd.DatetimeIndex:
    if timestamp_column not in frame.columns:
        raise ValueError(
            f"{label} CSV is missing timestamp column {timestamp_column!r}. "
            f"Available columns: {list(frame.columns)}"
        )
    parsed = pd.to_datetime(frame[timestamp_column], errors="coerce")
    if parsed.isna().any():
        bad_rows = parsed.index[parsed.isna()].tolist()[:5]
        raise ValueError(f"{label} CSV has invalid timestamps at rows {bad_rows}")
    timestamps = pd.DatetimeIndex(parsed)
    if timestamps.tz is not None:
        timestamps = timestamps.tz_localize(None)
    if timestamps.has_duplicates:
        duplicate = timestamps[timestamps.duplicated()][0]
        raise ValueError(f"{label} CSV has duplicate timestamp {duplicate}")
    if not timestamps.is_monotonic_increasing:
        raise ValueError(
            f"{label} timestamps are not strictly increasing; sort/fix the source CSV "
            "rather than silently reordering it in the model loader"
        )
    return timestamps


def _validate_frequency(
    timestamps: pd.DatetimeIndex,
    frequency: str,
    label: str,
) -> None:
    if len(timestamps) < 2:
        raise ValueError(f"{label} must contain at least two timestamps")
    expected = pd.Timedelta(frequency)
    differences = timestamps.to_series(index=np.arange(len(timestamps))).diff().iloc[1:]
    invalid = differences != expected
    if invalid.any():
        first_position = int(invalid[invalid].index[0])
        actual = differences.loc[first_position]
        previous = timestamps[first_position - 1]
        current = timestamps[first_position]
        raise ValueError(
            f"{label} must have an uninterrupted {frequency} time axis; "
            f"found {actual} between {previous} and {current}"
        )


def _require_numeric(
    source: pd.DataFrame,
    columns: Sequence[str],
    label: str,
) -> pd.DataFrame:
    missing = [column for column in columns if column not in source.columns]
    if missing:
        raise ValueError(
            f"{label} CSV is missing required columns {missing}. "
            f"Available columns: {list(source.columns)}"
        )
    converted = pd.DataFrame(index=source.index)
    for column in columns:
        values = pd.to_numeric(source[column], errors="coerce")
        if values.isna().any():
            bad_rows = values.index[values.isna()].tolist()[:5]
            raise ValueError(f"{label} column {column!r} is non-numeric/empty at rows {bad_rows}")
        array = values.to_numpy(dtype=np.float64)
        if not np.isfinite(array).all():
            raise ValueError(f"{label} column {column!r} contains non-finite values")
        converted[column] = array
    return converted


def _first_existing_column(frame: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    return next((column for column in candidates if column in frame.columns), None)


def _generated_import_tariff(
    timestamps: pd.DatetimeIndex,
    temperature: pd.Series,
    config: HKUSTConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate the project's documented Guangdong-style TOU scenario."""

    dates = timestamps.normalize()
    daily_max = pd.Series(temperature.to_numpy(), index=dates).groupby(level=0).max()
    hot_days = set(daily_max[daily_max >= config.high_temperature_threshold_C].index)
    hours = timestamps.hour.to_numpy() + timestamps.minute.to_numpy() / 60.0
    months = timestamps.month.to_numpy()
    sharp = (
        ((months >= 7) & (months <= 9)) | np.array([date in hot_days for date in dates])
    ) & (((hours >= 11) & (hours < 12)) | ((hours >= 15) & (hours < 17)))
    valley = (hours >= 0) & (hours < 8)
    peak = ((hours >= 10) & (hours < 12)) | ((hours >= 14) & (hours < 19))

    periods = np.full(len(timestamps), "flat", dtype=object)
    periods[valley] = "valley"
    periods[peak] = "peak"
    periods[sharp] = "sharp_peak"
    prices = np.array(
        [config.import_tariffs_CNY_per_kWh[str(period)] for period in periods],
        dtype=np.float64,
    )
    return periods, prices


def _canonical_frame(config: HKUSTConfig) -> Tuple[pd.DataFrame, Dict[str, str]]:
    source_path = _as_path(config.data_path)
    source = _read_csv(source_path, "dispatch input")
    timestamps = _parse_timestamps(source, config.timestamp_column, "dispatch input")
    _validate_frequency(timestamps, config.frequency, "dispatch input")

    source_names = [config.source_columns[name] for name in config.source_columns]
    numeric = _require_numeric(source, source_names, "dispatch input")
    frame = pd.DataFrame({"datetime": timestamps})
    for canonical, source_name in config.source_columns.items():
        frame[canonical] = numeric[source_name].to_numpy(dtype=np.float64)

    for quality_column in ("history_valid", "pv_target_valid"):
        if quality_column not in source:
            if config.require_quality_flags:
                raise ValueError(
                    f"Frozen training data must contain {quality_column!r}; "
                    "run prepare_m0_data.py and use its frozen snapshot."
                )
            frame[quality_column] = True
        else:
            values = _require_numeric(source, [quality_column], "quality flags")[quality_column]
            if not values.isin([0, 1]).all():
                raise ValueError(f"{quality_column} must contain only 0 or 1")
            frame[quality_column] = values.to_numpy(dtype=bool)

    generated: Dict[str, str] = {}

    import_column = _first_existing_column(source, config.import_price_candidates)
    if import_column is None:
        tariff_period, import_price = _generated_import_tariff(
            timestamps, frame["temperature"], config
        )
        frame["tariff_period"] = tariff_period
        frame["import_price"] = import_price
        generated["import_price"] = config.tariff_provenance
    else:
        frame["import_price"] = _require_numeric(
            source, [import_column], "dispatch input"
        )[import_column].to_numpy(dtype=np.float64)

    export_column = _first_existing_column(source, config.export_price_candidates)
    if export_column is None:
        frame["export_price"] = float(config.export_price_CNY_per_kWh)
        generated["export_price"] = config.tariff_provenance
    else:
        frame["export_price"] = _require_numeric(
            source, [export_column], "dispatch input"
        )[export_column].to_numpy(dtype=np.float64)

    grid_column = _first_existing_column(source, config.grid_available_candidates)
    if grid_column is None:
        frame["grid_available"] = float(config.default_grid_available)
        generated["grid_available"] = config.grid_availability_provenance
    else:
        frame["grid_available"] = _require_numeric(
            source, [grid_column], "dispatch input"
        )[grid_column].to_numpy(dtype=np.float64)

    frame, critical_generated = _attach_critical_load(
        frame=frame,
        primary_source=source,
        timestamps=timestamps,
        config=config,
    )
    generated.update(critical_generated)
    return frame, generated


def _attach_critical_load(
    frame: pd.DataFrame,
    primary_source: pd.DataFrame,
    timestamps: pd.DatetimeIndex,
    config: HKUSTConfig,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Prefer the transparent minimum-operation CL1+CL2 tier file."""

    tiers_path = _as_path(config.load_tiers_path)
    if tiers_path.is_file():
        tiers = _read_csv(tiers_path, "load tiers")
        tier_timestamps = _parse_timestamps(tiers, config.timestamp_column, "load tiers")
        _validate_frequency(tier_timestamps, config.frequency, "load tiers")
        if not tier_timestamps.equals(timestamps):
            raise ValueError(
                "load tiers timestamps must exactly match dispatch input timestamps; "
                "refusing a potentially shifted critical-load merge"
            )
        cl1_source, cl2_source = config.critical_load_columns
        values = _require_numeric(tiers, [cl1_source, cl2_source], "load tiers")
        frame["critical_load_cl1"] = values[cl1_source].to_numpy(dtype=np.float64)
        frame["critical_load_cl2"] = values[cl2_source].to_numpy(dtype=np.float64)
        frame["critical_load"] = (
            frame["critical_load_cl1"] + frame["critical_load_cl2"]
        )
        return frame, {"critical_load": config.critical_load_definition}

    fallback = _first_existing_column(primary_source, ("critical_load", "critical_load_kW"))
    if fallback is None:
        raise FileNotFoundError(
            f"load tiers CSV does not exist: {tiers_path}. The dispatch CSV also has no "
            "critical_load/critical_load_kW fallback column."
        )
    critical = _require_numeric(primary_source, [fallback], "dispatch input")[fallback]
    frame["critical_load_cl1"] = critical.to_numpy(dtype=np.float64)
    frame["critical_load_cl2"] = 0.0
    frame["critical_load"] = critical.to_numpy(dtype=np.float64)
    return frame, {}


def _validate_config(config: HKUSTConfig) -> None:
    if config.dt_hours != pd.Timedelta(config.frequency).total_seconds() / 3600.0:
        raise ValueError(
            f"dt_hours={config.dt_hours} disagrees with frequency={config.frequency!r}"
        )
    if config.frequency != "15min" or config.dt_hours != 0.25:
        raise ValueError("This variant requires a 15-minute interval")
    if config.prediction_horizon != 1 or config.lookback_steps != 96:
        raise ValueError("This variant requires lookback_steps=96 and prediction_horizon=1")
    if config.target_feature != "pv":
        raise ValueError("The HKUST LSTM target must be PV only")
    if config.target_feature not in config.history_features:
        raise ValueError("PV must be present in history_features to build the target scaler")
    if len(set(config.history_features)) != len(config.history_features):
        raise ValueError("history_features contains duplicates")
    expected_splits = ("train", "val", "test")
    if tuple(config.split_windows) != expected_splits:
        raise ValueError(
            f"split_windows must be ordered as {expected_splits}; got {tuple(config.split_windows)}"
        )
    previous_end: Optional[pd.Timestamp] = None
    for name in expected_splits:
        window = config.split_windows[name]
        start, end = pd.Timestamp(window.start), pd.Timestamp(window.end)
        if start >= end:
            raise ValueError(f"{name} split start must be before end")
        if previous_end is not None and start != previous_end:
            raise ValueError("train/val/test split windows must be contiguous and non-overlapping")
        previous_end = end
    valid_soc_order = (
        0.0
        <= config.min_soc_fraction
        < config.initial_soc_fraction
        < config.max_soc_fraction
        <= 1.0
    )
    if not valid_soc_order:
        raise ValueError("SOC fractions must satisfy 0 <= min < initial < max <= 1")
    if not (0.0 < config.roundtrip_efficiency <= 1.0):
        raise ValueError("roundtrip_efficiency must be in (0, 1]")
    if config.no_arbitrage_margin_CNY_per_kWh < 0.0:
        raise ValueError("no_arbitrage_margin_CNY_per_kWh must be non-negative")


def _anchors_for_window(
    timestamps: pd.DatetimeIndex,
    window: SplitWindow,
    lookback_steps: int,
    prediction_horizon: int,
    frequency: str,
) -> np.ndarray:
    start = pd.Timestamp(window.start)
    end = pd.Timestamp(window.end)
    final_offset = (prediction_horizon - 1) * pd.Timedelta(frequency)
    positions = np.arange(len(timestamps), dtype=np.int64)
    valid = (
        (timestamps >= start)
        & ((timestamps + final_offset) < end)
        & (positions >= lookback_steps)
        & ((positions + prediction_horizon) <= len(timestamps))
    )
    return positions[np.asarray(valid)]


def _prepare(config: HKUSTConfig) -> _PreparedData:
    _validate_config(config)
    frame, generated_columns = _canonical_frame(config)
    timestamps = pd.DatetimeIndex(frame["datetime"])

    required_canonical = set(config.history_features) | {
        "pv",
        "load",
        "import_price",
        "export_price",
        "grid_available",
        "critical_load",
        "critical_load_cl1",
        "critical_load_cl2",
    }
    missing = sorted(required_canonical.difference(frame.columns))
    if missing:
        raise ValueError(f"Canonical dataset is missing columns {missing}")

    nonnegative = [
        "pv",
        "load",
        "import_price",
        "export_price",
        "critical_load",
        "critical_load_cl1",
        "critical_load_cl2",
    ]
    for column in nonnegative:
        if (frame[column] < -1e-9).any():
            raise ValueError(f"Canonical column {column!r} contains negative values")
    if ((frame["grid_available"] < 0.0) | (frame["grid_available"] > 1.0)).any():
        raise ValueError("grid_available must lie in [0, 1]")
    if (frame["critical_load"] > frame["load"] + 1e-6).any():
        raise ValueError("critical_load (CL1+CL2) cannot exceed total load")

    train_window = config.split_windows["train"]
    scaler_start, scaler_end = pd.Timestamp(train_window.start), pd.Timestamp(train_window.end)
    train_mask = (
        (timestamps >= scaler_start)
        & (timestamps < scaler_end)
        & frame["history_valid"].to_numpy(dtype=bool)
    )
    if not train_mask.any():
        raise ValueError("No valid history rows fall inside the configured training interval")

    history_array = frame.loc[:, list(config.history_features)].to_numpy(dtype=np.float32)
    history_raw = torch.from_numpy(history_array.copy())
    train_history = history_raw[torch.from_numpy(np.asarray(train_mask))]
    history_min = train_history.amin(dim=0)
    history_max = train_history.amax(dim=0)
    scale = history_max - history_min
    safe_scale = torch.where(scale > 0.0, scale, torch.ones_like(scale))
    history_normalised = (history_raw - history_min) / safe_scale

    anchors_by_split = {
        name: _anchors_for_window(
            timestamps,
            window,
            config.lookback_steps,
            config.prediction_horizon,
            config.frequency,
        )
        for name, window in config.split_windows.items()
    }
    before_quality = {name: len(anchors) for name, anchors in anchors_by_split.items()}
    # Prefix sums allow exact whole-window validation in O(number of rows).
    # A held-out sample may use earlier observed history, but its entire label
    # window must remain in its own split and contain valid PV labels.
    history_bad = np.r_[0, np.cumsum(~frame["history_valid"].to_numpy(dtype=bool))]
    target_bad = np.r_[0, np.cumsum(~frame["pv_target_valid"].to_numpy(dtype=bool))]
    for name, anchors in anchors_by_split.items():
        valid_history = history_bad[anchors] == history_bad[anchors - config.lookback_steps]
        valid_target = target_bad[anchors + config.prediction_horizon] == target_bad[anchors]
        anchors_by_split[name] = anchors[valid_history & valid_target]
    quality_summary = {
        "required_quality_flags": config.require_quality_flags,
        "invalid_history_rows": int((~frame["history_valid"]).sum()),
        "invalid_pv_target_rows": int((~frame["pv_target_valid"]).sum()),
        "scaler_valid_training_rows": int(np.asarray(train_mask).sum()),
        "samples_before_quality_filter": before_quality,
        "samples_rejected_by_quality": {
            name: before_quality[name] - len(anchors)
            for name, anchors in anchors_by_split.items()
        },
        "sample_policy": "all lookback history_valid and all forecast pv_target_valid",
    }
    empty = [name for name, anchors in anchors_by_split.items() if len(anchors) == 0]
    if empty:
        raise ValueError(f"No complete lookback/forecast samples for splits {empty}")

    unix_seconds = torch.from_numpy(
        (timestamps.asi8 // 1_000_000_000).astype(np.int64, copy=True)
    )
    return _PreparedData(
        frame=frame,
        history_raw=history_raw,
        history_normalised=history_normalised,
        history_min=history_min,
        history_max=history_max,
        unix_seconds=unix_seconds,
        anchors_by_split=anchors_by_split,
        generated_columns=generated_columns,
        scaler_start=scaler_start,
        scaler_end=scaler_end,
        quality_summary=quality_summary,
    )


class CSVDataset(Dataset):
    """HKUST PV-forecast dataset with explicit future MPC inputs.

    Preferred construction is ``CSVDataset(split="train")`` followed by
    ``dataset.for_split("val")`` and ``dataset.for_split("test")``.  Split views
    share exactly the same training-fitted scaler and canonical source frame.

    Returned tensor shapes are:

    - ``history``: ``[lookback_steps, input_dim]`` (normalised)
    - ``pv_target``: ``[1, 1]`` (kW; the only training target)
    - ``pv_target_scaled``: ``[1, 1]`` (optional normalised PV target)
    - ``pv_actual`` and each physical MPC input: ``[1, 1]`` (raw units)
    - ``future_timestamps``: ``[1]`` (Unix seconds, local-naive source time)

    Legacy constructor names are accepted so the surrounding migration can be
    staged, but ``site_id`` and online batch-size arguments no longer select or
    fabricate data.
    """

    SPLIT_ALIASES = {"validation": "val", "valid": "val"}
    PHYSICAL_COLUMNS = (
        "pv",
        "load",
        "import_price",
        "export_price",
        "grid_available",
        "critical_load",
        "critical_load_cl1",
        "critical_load_cl2",
    )

    def __init__(
        self,
        site_id: Optional[int] = None,
        lookback_window: Optional[int] = None,
        prediction_horizon: Optional[int] = None,
        train_batchsize: int = 1,
        valid_batchsize: int = 1,
        input_features: Optional[Sequence[str]] = None,
        output_features: Optional[Sequence[str]] = None,
        *,
        config: Optional[HKUSTConfig] = None,
        split: str = "train",
        data_path: Optional[Union[str, Path]] = None,
        load_tiers_path: Optional[Union[str, Path]] = None,
        column_map: Optional[Mapping[str, str]] = None,
        _prepared: Optional[_PreparedData] = None,
    ) -> None:
        del site_id  # Retained only for source-level migration compatibility.
        chosen = config or DEFAULT_CONFIG
        updates: Dict[str, Any] = {}
        if lookback_window is not None:
            updates["lookback_steps"] = int(lookback_window)
        if prediction_horizon is not None:
            updates["prediction_horizon"] = int(prediction_horizon)
        if input_features is not None:
            updates["history_features"] = tuple(input_features)
        if data_path is not None:
            updates["data_path"] = _as_path(data_path)
        if load_tiers_path is not None:
            updates["load_tiers_path"] = _as_path(load_tiers_path)
        if column_map is not None:
            merged_columns = dict(chosen.source_columns)
            merged_columns.update(column_map)
            updates["source_columns"] = merged_columns
        if output_features is not None and tuple(output_features) != ("pv",):
            raise ValueError(
                "HKUST LSTM output_features must be exactly ['pv']; load, price, grid "
                "availability and critical load are explicit MPC inputs"
            )
        if updates:
            if _prepared is not None:
                raise ValueError("Cannot change configuration while reusing prepared data")
            chosen = replace(chosen, **updates)

        split = self.SPLIT_ALIASES.get(split, split)
        if split not in chosen.split_windows:
            raise ValueError(
                f"Unknown split {split!r}; choose one of {tuple(chosen.split_windows)}"
            )

        self.config = chosen
        self.split = split
        self._prepared = _prepared or _prepare(chosen)
        self._anchors = self._prepared.anchors_by_split[split]

        # Public model/checkpoint interface.
        self.lookback_window = chosen.lookback_steps
        self.prediction_horizon = chosen.prediction_horizon
        self.train_batchsize = int(train_batchsize)
        self.valid_batchsize = int(valid_batchsize)
        self.input_features = list(chosen.history_features)
        self.output_features = [chosen.target_feature]
        self.available_features = list(chosen.history_features)
        self.bypass_features: List[str] = []
        self.indices_input_features = list(range(len(self.input_features)))
        self.indices_output_features = [self.input_features.index(chosen.target_feature)]
        self.indices_bypass_features: List[int] = []
        self.scaling_factors_min = self._prepared.history_min.clone()
        self.scaling_factors_max = self._prepared.history_max.clone()

        self._validate_split_view()

    @property
    def input_dim(self) -> int:
        return len(self.input_features)

    @property
    def pv_min(self) -> float:
        index = self.input_features.index(self.config.target_feature)
        return float(self.scaling_factors_min[index])

    @property
    def pv_max(self) -> float:
        index = self.input_features.index(self.config.target_feature)
        return float(self.scaling_factors_max[index])

    @property
    def timestamps(self) -> pd.DatetimeIndex:
        """Forecast-start timestamps for this split."""

        return pd.DatetimeIndex(self._prepared.frame.iloc[self._anchors]["datetime"])

    @property
    def generated_columns(self) -> Mapping[str, str]:
        """Scenario columns created because no measured/source column existed."""

        return dict(self._prepared.generated_columns)

    @property
    def source_frame(self) -> pd.DataFrame:
        """Read-only-by-convention canonical frame; callers receive a copy."""

        return self._prepared.frame.copy()

    def for_split(self, split: str) -> "CSVDataset":
        """Return a chronological view sharing this exact scaler and source data."""

        split = self.SPLIT_ALIASES.get(split, split)
        return CSVDataset(
            config=self.config,
            split=split,
            train_batchsize=self.train_batchsize,
            valid_batchsize=self.valid_batchsize,
            _prepared=self._prepared,
        )

    def split_views(self) -> Dict[str, "CSVDataset"]:
        return {name: self.for_split(name) for name in self.config.split_windows}

    def state_dict(self) -> Dict[str, Any]:
        """Checkpoint only the training-fitted scaler and its feature order."""

        return {
            "scaling_factors_min": self.scaling_factors_min.clone(),
            "scaling_factors_max": self.scaling_factors_max.clone(),
            "scaler_features": tuple(self.input_features),
            "scaler_fit_interval": (
                self._prepared.scaler_start.isoformat(),
                self._prepared.scaler_end.isoformat(),
            ),
            "scaler_fit_quality_policy": "training interval AND history_valid",
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Load a compatible scaler without re-reading or re-splitting the CSV."""

        feature_order = tuple(state_dict.get("scaler_features", self.input_features))
        if feature_order != tuple(self.input_features):
            raise ValueError(
                f"Scaler feature order {feature_order} does not match {tuple(self.input_features)}"
            )
        minimum = torch.as_tensor(state_dict["scaling_factors_min"], dtype=torch.float32)
        maximum = torch.as_tensor(state_dict["scaling_factors_max"], dtype=torch.float32)
        expected_shape = (self.input_dim,)
        if minimum.shape != expected_shape or maximum.shape != expected_shape:
            raise ValueError(
                f"Scaler tensors must have shape {expected_shape}; "
                f"got {minimum.shape}, {maximum.shape}"
            )
        if not torch.isfinite(minimum).all() or not torch.isfinite(maximum).all():
            raise ValueError("Scaler bounds must be finite")
        if torch.any(maximum < minimum):
            raise ValueError("Scaler maximum cannot be smaller than minimum")
        scale = maximum - minimum
        safe_scale = torch.where(scale > 0.0, scale, torch.ones_like(scale))
        normalised = (self._prepared.history_raw - minimum) / safe_scale
        scaler_interval = state_dict.get("scaler_fit_interval")
        if scaler_interval is None:
            scaler_start = self._prepared.scaler_start
            scaler_end = self._prepared.scaler_end
        else:
            if len(scaler_interval) != 2:
                raise ValueError("scaler_fit_interval must contain start and end")
            scaler_start = pd.Timestamp(scaler_interval[0])
            scaler_end = pd.Timestamp(scaler_interval[1])
            if scaler_start >= scaler_end:
                raise ValueError("scaler_fit_interval start must precede end")
            training_window = self.config.split_windows["train"]
            if (scaler_start, scaler_end) != (
                pd.Timestamp(training_window.start), pd.Timestamp(training_window.end)
            ):
                raise ValueError("Checkpoint scaler interval differs from the training split")
        self._prepared = replace(
            self._prepared,
            history_normalised=normalised,
            history_min=minimum.clone(),
            history_max=maximum.clone(),
            scaler_start=scaler_start,
            scaler_end=scaler_end,
        )
        self.scaling_factors_min = minimum.clone()
        self.scaling_factors_max = maximum.clone()

    def denormalise_pv(self, values: torch.Tensor) -> torch.Tensor:
        """Convert normalised LSTM PV output back to kW."""

        return values * (self.pv_max - self.pv_min) + self.pv_min

    def actual_at(self, index: int) -> Dict[str, Any]:
        """Return raw first-step values for closed-loop execution/logging."""

        anchor = int(self._anchors[index])
        row = self._prepared.frame.iloc[anchor]
        return {
            "timestamp": pd.Timestamp(row["datetime"]),
            **{column: float(row[column]) for column in self.PHYSICAL_COLUMNS},
        }

    def get_physical_profile(self) -> pd.DataFrame:
        """Return one actual row per forecast start in this split."""

        columns = ["datetime", *self.PHYSICAL_COLUMNS]
        return self._prepared.frame.iloc[self._anchors][columns].reset_index(drop=True).copy()

    def get_profile(self) -> torch.Tensor:
        """Compatibility profile ordered as PV, load and import price."""

        profile = self.get_physical_profile()
        return torch.as_tensor(
            profile[["pv", "load", "import_price"]].to_numpy(), dtype=torch.float32
        )

    def validation_summary(self) -> Dict[str, Any]:
        timestamps = pd.DatetimeIndex(self._prepared.frame["datetime"])
        return {
            "data_path": str(_as_path(self.config.data_path)),
            "load_tiers_path": str(_as_path(self.config.load_tiers_path)),
            "source_columns": dict(self.config.source_columns),
            "source_rows": len(self._prepared.frame),
            "source_start": timestamps[0].isoformat(),
            "source_end": timestamps[-1].isoformat(),
            "frequency": self.config.frequency,
            "lookback_steps": self.lookback_window,
            "prediction_horizon": self.prediction_horizon,
            "history_features": tuple(self.input_features),
            "target_features": tuple(self.output_features),
            "split_samples": {
                name: len(anchors)
                for name, anchors in self._prepared.anchors_by_split.items()
            },
            "split_dates": self.config.split_dates(),
            "scaler_fit_interval": (
                self._prepared.scaler_start.isoformat(),
                self._prepared.scaler_end.isoformat(),
            ),
            "generated_columns": dict(self.generated_columns),
            "quality": dict(self._prepared.quality_summary),
            "grid_available_min": float(self._prepared.frame["grid_available"].min()),
            "grid_available_max": float(self._prepared.frame["grid_available"].max()),
            "outage_rows": int(
                (self._prepared.frame["grid_available"] < 1.0).sum()
            ),
        }

    def _validate_split_view(self) -> None:
        window = self.config.split_windows[self.split]
        start, end = pd.Timestamp(window.start), pd.Timestamp(window.end)
        forecast_starts = self.timestamps
        forecast_ends = forecast_starts + (
            self.prediction_horizon - 1
        ) * pd.Timedelta(self.config.frequency)
        if (forecast_starts < start).any() or (forecast_ends >= end).any():
            raise RuntimeError(f"{self.split} sample targets cross a split boundary")
        if self.output_features != ["pv"]:
            raise RuntimeError("Dataset target leakage: output must be PV only")

    def __len__(self) -> int:
        return len(self._anchors)

    def forecast_item(self, index: int) -> TensorDict:
        """Tensor-only M0 sample, avoiding unused MPC DataFrame conversions."""
        anchor = int(self._anchors[index])
        stop = anchor + self.prediction_horizon
        pv_index = self.input_features.index(self.config.target_feature)
        pv = self._prepared.history_raw[:, pv_index : pv_index + 1]
        # Keep a fixed one-day lag even when the forecast horizon is one step.
        # Last-value persistence is a different, short-term baseline.
        persistence = pv[anchor - 96 : stop - 96].clone()
        return {
            "history": self._prepared.history_normalised[
                anchor - self.lookback_window : anchor
            ].clone(),
            "pv_target": pv[anchor:stop].clone(),
            "timestamp": self._prepared.unix_seconds[anchor].clone(),
            "future_timestamps": self._prepared.unix_seconds[anchor:stop].clone(),
            "persistence": persistence,
            "persistence_last": pv[anchor - 1 : anchor].clone(),
        }

    def forecast_view(self) -> "ForecastDataset":
        return ForecastDataset(self)

    def __getitem__(self, index: int) -> TensorDict:
        anchor = int(self._anchors[index])
        history_start = anchor - self.lookback_window
        future_end = anchor + self.prediction_horizon
        frame = self._prepared.frame

        def future(column: str, dtype: torch.dtype = torch.float64) -> torch.Tensor:
            values = frame[column].iloc[anchor:future_end].to_numpy(dtype=np.float64)
            return torch.as_tensor(values.copy(), dtype=dtype).unsqueeze(-1)

        pv_index = self.input_features.index(self.config.target_feature)
        actual = self.actual_at(index)
        actual_tensors = {
            key: torch.tensor(value, dtype=torch.float64)
            for key, value in actual.items()
            if key != "timestamp"
        }
        return {
            "history": self._prepared.history_normalised[history_start:anchor].clone(),
            "pv_target": future("pv", dtype=torch.float32),
            "pv_target_scaled": self._prepared.history_normalised[
                anchor:future_end, pv_index : pv_index + 1
            ].clone(),
            "pv_actual": future("pv"),
            "load": future("load"),
            "import_price": future("import_price"),
            "export_price": future("export_price"),
            "grid_available": future("grid_available"),
            "critical_load": future("critical_load"),
            "critical_load_cl1": future("critical_load_cl1"),
            "critical_load_cl2": future("critical_load_cl2"),
            "timestamp": self._prepared.unix_seconds[anchor].clone(),
            "future_timestamps": self._prepared.unix_seconds[anchor:future_end].clone(),
            "actual": actual_tensors,
        }

    @staticmethod
    def add_specific_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add HKUST dataset arguments while retaining legacy CLI spellings."""

        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--data_path",
            "--data_csv",
            dest="data_path",
            type=str,
            default=str(DEFAULT_CONFIG.data_path),
            help="15-minute dispatch-input CSV.",
        )
        parser.add_argument(
            "--load_tiers_path",
            type=str,
            default=str(DEFAULT_CONFIG.load_tiers_path),
            help="15-minute CL1/CL2/CL3 load-tier CSV.",
        )
        parser.add_argument(
            "--dataset_split",
            choices=("train", "val", "test"),
            default="train",
            help="Chronological dataset view.",
        )
        parser.add_argument(
            "--site_id",
            type=int,
            default=12,
            help="Deprecated compatibility argument; HKUST paths now select the data.",
        )
        parser.add_argument(
            "--train_batchsize",
            type=int,
            default=1,
            help="Retained for learner compatibility; samples are no longer fabricated online.",
        )
        parser.add_argument(
            "--valid_batchsize",
            type=int,
            default=1,
            help="Retained for learner compatibility.",
        )
        parser.add_argument(
            "--input_features",
            nargs="+",
            default=list(DEFAULT_CONFIG.history_features),
            help="Normalised historical LSTM features.",
        )
        parser.add_argument(
            "--output_features",
            nargs="+",
            default=["pv"],
            help="Must remain 'pv'; physical future inputs bypass the LSTM.",
        )
        parser.add_argument(
            "--lookback_window",
            type=int,
            default=DEFAULT_CONFIG.lookback_steps,
            help="Past 15-minute samples supplied to the LSTM.",
        )
        parser.add_argument(
            "--prediction_horizon",
            type=int,
            default=DEFAULT_CONFIG.prediction_horizon,
            help="Fixed at one step (the next 15-minute interval).",
        )
        return parser


class ForecastDataset(Dataset):
    """Lightweight view of an already validated/scaled chronological dataset."""

    def __init__(self, dataset: CSVDataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> TensorDict:
        return self.dataset.forecast_item(index)
