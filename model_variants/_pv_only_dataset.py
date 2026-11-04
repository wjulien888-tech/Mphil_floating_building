"""Frozen 2021--2023 constructed PV with configurable history and output windows.

This forecast-only loader reads no demand, tariff or weather columns. Missing
timestamps remain on the source axis; quality flags reject complete windows.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from configurations.hkust_15min import SplitWindow


FROZEN_ROOT = Path(__file__).resolve().parents[1] / "output/frozen/pv_only_2021_2023_96to1_v1_20260920"


@dataclass(frozen=True)
class PVForecastConfig:
    data_path: Path = FROZEN_ROOT / "pv_timeseries.csv"
    timestamp_column: str = "datetime"
    source_columns: Mapping[str, str] = field(default_factory=lambda: {"pv": "pv_kW"})
    history_features: tuple[str, ...] = ("pv",)
    target_feature: str = "pv"
    timezone: str = "Asia/Shanghai"
    frequency: str = "15min"
    dt_hours: float = 0.25
    lookback_steps: int = 96
    prediction_horizon: int = 1
    anchor_lookback_steps: int = 96
    require_quality_flags: bool = True
    pv_input_provenance: str = (
        "Constructed floating-building PV: each year's PVGIS building baseline "
        "times a causal disturbance derived from HKUST measured PV; not measured "
        "floating-building telemetry."
    )
    split_windows: Mapping[str, SplitWindow] = field(default_factory=lambda: {
        "train": SplitWindow("2021-07-01", "2023-09-01"),
        "val": SplitWindow("2023-09-01", "2023-11-01"),
        "test": SplitWindow("2023-11-01", "2024-01-01"),
    })

    def split_dates(self) -> dict[str, tuple[str, str]]:
        return {name: (window.start, window.end) for name, window in self.split_windows.items()}


DEFAULT_CONFIG = PVForecastConfig()


@dataclass(frozen=True)
class _PreparedPV:
    frame: pd.DataFrame
    raw: torch.Tensor
    normalised: torch.Tensor
    minimum: torch.Tensor
    maximum: torch.Tensor
    unix_seconds: torch.Tensor
    anchors: Mapping[str, np.ndarray]
    quality: Mapping[str, Any]


def _prepare(config: PVForecastConfig) -> _PreparedPV:
    for name in ("lookback_steps", "prediction_horizon", "anchor_lookback_steps"):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    # A shared history guard makes comparisons use identical forecast origins.
    # Retain at least one day for the previous-day persistence baseline.
    history_guard = max(96, config.lookback_steps, config.anchor_lookback_steps)
    horizon = config.prediction_horizon
    if tuple(config.history_features) != ("pv",) or config.target_feature != "pv":
        raise ValueError("PV-only variants require history_features=['pv'] and target_feature='pv'")
    if pd.Timedelta(config.frequency) != pd.Timedelta(minutes=15) or config.dt_hours != 0.25:
        raise ValueError("PV-only variants require a 15-minute time axis")
    if set(config.split_windows) != {"train", "val", "test"}:
        raise ValueError("Exactly train, val and test split windows are required")
    windows = [(pd.Timestamp(config.split_windows[name].start),
                pd.Timestamp(config.split_windows[name].end)) for name in ("train", "val", "test")]
    if any(start >= end for start, end in windows) or any(
        windows[index][1] > windows[index + 1][0] for index in range(2)
    ):
        raise ValueError("Split windows must be ordered, nonempty and nonoverlapping")

    source = pd.read_csv(config.data_path)
    pv_column = config.source_columns["pv"]
    required = [config.timestamp_column, pv_column, "history_valid", "pv_target_valid", "split"]
    missing = set(required).difference(source.columns)
    if missing:
        raise ValueError(f"PV-only CSV is missing columns: {sorted(missing)}")
    frame = source[required].rename(columns={config.timestamp_column: "datetime", pv_column: "pv"}).copy()
    timestamps = pd.DatetimeIndex(pd.to_datetime(frame["datetime"], errors="raise"))
    if len(timestamps) < history_guard + horizon or timestamps.hasnans or timestamps.tz is not None:
        raise ValueError("PV-only timestamps must be local naive and contain a complete history/target window")
    if not timestamps.is_monotonic_increasing or timestamps.has_duplicates:
        raise ValueError("PV-only timestamps must be ordered and unique")
    if not np.all(np.diff(timestamps.asi8) == pd.Timedelta(minutes=15).value):
        raise ValueError("PV-only CSV must retain the complete 15-minute axis, including missing rows")
    frame["datetime"] = timestamps
    for column in ("history_valid", "pv_target_valid"):
        numeric = pd.to_numeric(frame[column], errors="raise")
        if numeric.isna().any() or not numeric.isin([0, 1]).all():
            raise ValueError(f"{column} must contain only 0 or 1")
        frame[column] = numeric.astype(bool)
    frame["pv"] = pd.to_numeric(frame["pv"], errors="raise")
    valid = frame["history_valid"] | frame["pv_target_valid"]
    if not np.isfinite(frame.loc[valid, "pv"]).all() or (frame.loc[valid, "pv"] < 0).any():
        raise ValueError("Valid PV history/targets must be finite nonnegative power in kW")

    expected_split = np.full(len(frame), "excluded", dtype=object)
    for name, window in config.split_windows.items():
        mask = (timestamps >= pd.Timestamp(window.start)) & (timestamps < pd.Timestamp(window.end))
        expected_split[mask] = name
    if not np.array_equal(frame["split"].to_numpy(), expected_split):
        raise ValueError("CSV split labels disagree with configured chronological split windows")
    train_mask = (frame["split"] == "train") & frame["history_valid"]
    if not train_mask.any():
        raise ValueError("No valid training PV rows for fitting the scaler")
    raw = torch.from_numpy(frame[["pv"]].to_numpy(dtype=np.float32, copy=True))
    train = raw[torch.from_numpy(train_mask.to_numpy())]
    minimum, maximum = train.amin(dim=0), train.amax(dim=0)
    scale = torch.where(maximum > minimum, maximum - minimum, torch.ones_like(minimum))
    normalised = (raw - minimum) / scale
    history_bad = np.r_[0, np.cumsum(~frame["history_valid"].to_numpy())]
    target_bad = np.r_[0, np.cumsum(~frame["pv_target_valid"].to_numpy())]
    candidates = np.arange(history_guard, len(frame) - horizon + 1, dtype=np.int64)
    valid_window = (
        (history_bad[candidates] == history_bad[candidates - history_guard])
        & (target_bad[candidates + horizon] == target_bad[candidates])
    )
    anchors, before, boundary_rejections = {}, {}, {}
    for name in config.split_windows:
        starts_in_split = expected_split[candidates] == name
        in_split = starts_in_split & (expected_split[candidates + horizon - 1] == name)
        boundary_rejections[name] = int((starts_in_split & ~in_split).sum())
        before[name] = int(in_split.sum())
        anchors[name] = candidates[in_split & valid_window]
        if not len(anchors[name]):
            raise ValueError(f"No valid complete {config.lookback_steps}-to-{horizon} PV samples in {name}")
    return _PreparedPV(
        frame, raw, normalised, minimum, maximum,
        torch.from_numpy((timestamps.asi8 // 1_000_000_000).astype(np.int64, copy=True)),
        anchors, {
            "invalid_history_rows": int((~frame["history_valid"]).sum()),
            "invalid_pv_target_rows": int((~frame["pv_target_valid"]).sum()),
            "scaler_valid_training_rows": int(train_mask.sum()),
            "samples_before_quality_filter": before,
            "samples_rejected_by_quality": {name: before[name] - len(anchors[name]) for name in anchors},
            "samples_rejected_by_split_boundary": boundary_rejections,
            "anchor_lookback_steps": history_guard,
            "sample_policy": (
                f"all {history_guard} previous history_valid rows AND all {horizon} "
                "pv_target_valid targets; the complete target window belongs to one split"
            ),
            "scaler_fit_quality_policy": "split=train AND history_valid",
        },
    )


class PVOnlyDataset(Dataset):
    """Forecast samples and split views sharing a training-only PV scaler."""

    def __init__(self, config: PVForecastConfig | None = None, split: str = "train",
                 *, _prepared: _PreparedPV | None = None) -> None:
        self.config = config or DEFAULT_CONFIG
        if split not in self.config.split_windows:
            raise ValueError(f"Unknown split: {split}")
        self.split = split
        self._prepared = _prepared if _prepared is not None else _prepare(self.config)
        self._anchors = self._prepared.anchors[split]
        self.input_features = ["pv"]
        self.output_features = ["pv"]
        self.lookback_window = self.config.lookback_steps
        self.prediction_horizon = self.config.prediction_horizon
        self.scaling_factors_min = self._prepared.minimum.clone()
        self.scaling_factors_max = self._prepared.maximum.clone()

    @property
    def input_dim(self) -> int:
        return 1

    @property
    def pv_min(self) -> float:
        return float(self.scaling_factors_min[0])

    @property
    def pv_max(self) -> float:
        return float(self.scaling_factors_max[0])

    @property
    def timestamps(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self._prepared.frame.iloc[self._anchors]["datetime"])

    @property
    def source_frame(self) -> pd.DataFrame:
        return self._prepared.frame.copy()

    def for_split(self, split: str) -> "PVOnlyDataset":
        return PVOnlyDataset(self.config, split, _prepared=self._prepared)

    def split_views(self) -> dict[str, "PVOnlyDataset"]:
        return {name: self.for_split(name) for name in self.config.split_windows}

    def state_dict(self) -> dict[str, Any]:
        window = self.config.split_windows["train"]
        return {
            "scaling_factors_min": self.scaling_factors_min.clone(),
            "scaling_factors_max": self.scaling_factors_max.clone(),
            "scaler_features": ("pv",),
            "scaler_fit_interval": (pd.Timestamp(window.start).isoformat(), pd.Timestamp(window.end).isoformat()),
            "scaler_fit_quality_policy": "split=train AND history_valid",
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        expected = self.state_dict()
        for name in ("scaler_features", "scaler_fit_interval", "scaler_fit_quality_policy"):
            actual = state_dict.get(name)
            if name != "scaler_fit_quality_policy" and actual is not None:
                actual = tuple(actual)
            if actual != expected[name]:
                raise ValueError(f"Checkpoint {name} differs from PV-only frozen data")
        for name in ("scaling_factors_min", "scaling_factors_max"):
            value = torch.as_tensor(state_dict[name], dtype=torch.float32)
            if not torch.equal(value, expected[name]):
                raise ValueError(f"Checkpoint {name} differs from the training-fitted PV scaler")

    def denormalise_pv(self, values: torch.Tensor) -> torch.Tensor:
        return values * (self.pv_max - self.pv_min) + self.pv_min

    def validation_summary(self) -> dict[str, Any]:
        frame = self._prepared.frame
        return {
            "data_path": str(self.config.data_path),
            "artifact_type": ("pv_only_forecast_96to1" if (self.lookback_window, self.prediction_horizon) == (96, 1)
                              else "pv_only_forecast_multistep"),
            "source_columns": dict(self.config.source_columns), "source_rows": len(frame),
            "source_start": frame.datetime.iloc[0].isoformat(), "source_end": frame.datetime.iloc[-1].isoformat(),
            "frequency": self.config.frequency, "timezone": self.config.timezone,
            "lookback_steps": self.lookback_window, "prediction_horizon": self.prediction_horizon,
            "anchor_lookback_steps": self.config.anchor_lookback_steps,
            "history_features": ["pv"], "target_features": ["pv"],
            "split_samples": {name: len(anchors) for name, anchors in self._prepared.anchors.items()},
            "split_dates": self.config.split_dates(),
            "scaler_fit_interval": self.state_dict()["scaler_fit_interval"],
            "pv_min_kw": self.pv_min, "pv_max_kw": self.pv_max,
            "quality": dict(self._prepared.quality), "pv_input_provenance": self.config.pv_input_provenance,
        }

    def __len__(self) -> int:
        return len(self._anchors)

    def forecast_item(self, index: int) -> dict[str, torch.Tensor]:
        anchor = int(self._anchors[index])
        stop = anchor + self.prediction_horizon
        previous_day_indices = anchor - 96 + torch.arange(self.prediction_horizon) % 96
        return {
            "history": self._prepared.normalised[anchor - self.lookback_window:anchor].clone(),
            "pv_target": self._prepared.raw[anchor:stop].clone(),
            "timestamp": self._prepared.unix_seconds[anchor].clone(),
            "future_timestamps": self._prepared.unix_seconds[anchor:stop].clone(),
            "persistence": self._prepared.raw[previous_day_indices].clone(),
            "persistence_last": self._prepared.raw[anchor - 1:anchor].repeat(self.prediction_horizon, 1),
        }

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.forecast_item(index)

    def forecast_view(self) -> "PVOnlyDataset":
        return self
