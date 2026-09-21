"""Single source of truth for the HKUST 15-minute LSTM--MPC experiment.

The two input CSV files live in the enclosing floating-building project rather
than in this vendored repository.  Paths are therefore resolved from this
file's location instead of from the process working directory.

The default electricity prices and grid availability are *scenario inputs*,
not measured HKUST data.  Their provenance is recorded below and the data
loader exposes which columns were generated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Tuple


OL_EMS_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = PROJECT_ROOT / "output"
FROZEN_DATA_ROOT = OUTPUT_ROOT / "frozen" / "hkust_m0_v1_20260909"


@dataclass(frozen=True)
class SplitWindow:
    """A half-open chronological interval: ``start <= t < end``."""

    start: str
    end: str


@dataclass(frozen=True)
class HKUSTConfig:
    """Configuration for the first HKUST 15-minute closed-loop experiment.

    Units are explicit in every field name.  The LSTM target is only ``pv``;
    load, prices, grid availability and critical load are future physical MPC
    inputs and are never model targets.
    """

    # Data sources and schema -------------------------------------------------
    data_path: Path = FROZEN_DATA_ROOT / "dispatch.csv"
    load_tiers_path: Path = FROZEN_DATA_ROOT / "load_tiers.csv"
    require_quality_flags: bool = True
    timestamp_column: str = "datetime"
    source_columns: Mapping[str, str] = field(
        default_factory=lambda: {
            "pv": "pv_floating_disturbed_raw_kW",
            "load": "load_kW",
            "temperature": "temperature_C",
            "relative_humidity": "relative_humidity_pct",
            "irradiance": "irradiance_W_m2",
            "rainfall": "rainfall_mm",
        }
    )
    pv_input_provenance: str = (
        "pv_floating_disturbed_raw_kW: PVGIS floating-building baseline multiplied "
        "by a causal trailing-history disturbance factor derived from HKUST measured PV, "
        "initialised with December 2022 observations; "
        "this is a constructed floating-PV scenario, not direct floating-building telemetry."
    )
    history_features: Tuple[str, ...] = (
        "pv",
        "temperature",
        "relative_humidity",
        "irradiance",
        "rainfall",
    )
    target_feature: str = "pv"
    critical_load_columns: Tuple[str, str] = (
        "minimum_operation_CL1_kW",
        "minimum_operation_CL2_kW",
    )
    critical_load_definition: str = (
        "minimum-operation CL1 + CL2 proxy from the frozen load_tiers.csv"
    )

    # Time axis and forecasting ----------------------------------------------
    timezone: str = "Asia/Shanghai"
    frequency: str = "15min"
    dt_hours: float = 0.25
    prediction_horizon: int = 96
    lookback_steps: int = 7 * 96
    split_windows: Mapping[str, SplitWindow] = field(
        default_factory=lambda: {
            # Chronological 8/2/2-month split.  Validation and test targets are
            # never used to fit the scaler.
            "train": SplitWindow("2023-01-01 00:00:00", "2023-09-01 00:00:00"),
            "val": SplitWindow("2023-09-01 00:00:00", "2023-11-01 00:00:00"),
            "test": SplitWindow("2023-11-01 00:00:00", "2024-01-01 00:00:00"),
        }
    )

    # Fixed first-stage battery ----------------------------------------------
    battery_capacity_kWh: float = 215.0
    battery_power_kW: float = 50.0
    roundtrip_efficiency: float = 0.92
    allow_grid_charging: bool = False
    min_soc_fraction: float = 0.10
    max_soc_fraction: float = 0.90
    initial_soc_fraction: float = 0.50
    self_discharge_per_step: float = 0.0

    # Grid limits.  No measured interconnection rating is currently available;
    # 4 MW preserves the non-binding limit used by the original ol-ems example.
    grid_import_limit_kW: float = 4000.0
    grid_export_limit_kW: float = 4000.0
    grid_limit_provenance: str = (
        "Compatibility assumption inherited from ol-ems; replace with the measured "
        "point-of-connection rating when available."
    )

    # Fallback scenario columns ----------------------------------------------
    import_price_candidates: Tuple[str, ...] = (
        "import_price",
        "import_price_CNY_per_kWh",
        "import_tariff_CNY_per_kWh",
    )
    export_price_candidates: Tuple[str, ...] = (
        "export_price",
        "export_price_CNY_per_kWh",
    )
    grid_available_candidates: Tuple[str, ...] = (
        "grid_available",
        "grid_status",
    )
    import_tariffs_CNY_per_kWh: Mapping[str, float] = field(
        default_factory=lambda: {
            "valley": 0.29526875,
            "flat": 0.73196875,
            "peak": 1.22496875,
            "sharp_peak": 1.52436875,
        }
    )
    export_price_CNY_per_kWh: float = 0.36
    no_arbitrage_margin_CNY_per_kWh: float = 1.0e-3
    no_arbitrage_policy: str = (
        "Inside the convex MPC only, effective export price is capped below the "
        "same-step import price by the configured margin. Realised exports retain "
        "the scenario tariff in Pymgrid accounting."
    )
    high_temperature_threshold_C: float = 35.0
    default_grid_available: float = 1.0
    tariff_provenance: str = (
        "Scenario tariff copied from the existing project function "
        "milp_e4b_full_year_grid_fixed_battery.add_tariff_columns; it is not a "
        "measured HKUST billing series."
    )
    grid_availability_provenance: str = (
        "Default all-grid-connected scenario; outage experiments must provide an "
        "explicit grid_available column or override the generated series."
    )

    # MPC objective weights.  Monetary terms use CNY and energy uses kWh.
    objective_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "energy_cost": 1.0,
            "pv_curtailment": 1.0e-3,
            "battery_cycle": 1.0e-4,
            "critical_unmet": 1.0e4,
            "noncritical_unmet": 1.0e3,
        }
    )

    @property
    def charge_efficiency(self) -> float:
        """Symmetric one-way efficiency whose product is the round-trip value."""

        return self.roundtrip_efficiency**0.5

    @property
    def discharge_efficiency(self) -> float:
        return self.roundtrip_efficiency**0.5

    @property
    def min_soc_kWh(self) -> float:
        return self.battery_capacity_kWh * self.min_soc_fraction

    @property
    def max_soc_kWh(self) -> float:
        return self.battery_capacity_kWh * self.max_soc_fraction

    @property
    def initial_soc_kWh(self) -> float:
        return self.battery_capacity_kWh * self.initial_soc_fraction

    @property
    def soc_min_max_kWh(self) -> Tuple[float, float]:
        return (self.min_soc_kWh, self.max_soc_kWh)

    @property
    def storage_power_min_max_kW(self) -> Tuple[float, float]:
        return (-self.battery_power_kW, self.battery_power_kW)

    @property
    def grid_power_min_max_kW(self) -> Tuple[float, float]:
        """Signed net-grid limits: negative import, positive export."""

        return (-self.grid_import_limit_kW, self.grid_export_limit_kW)

    def split_dates(self) -> Dict[str, Tuple[str, str]]:
        """Return plain serialisable split boundaries for logs/checkpoints."""

        return {
            name: (window.start, window.end)
            for name, window in self.split_windows.items()
        }


DEFAULT_CONFIG = HKUSTConfig()
CONFIG = DEFAULT_CONFIG

# Concise aliases for scripts that prefer module-level constants.
DATA_PATH = DEFAULT_CONFIG.data_path
LOAD_TIERS_PATH = DEFAULT_CONFIG.load_tiers_path
COLUMN_MAP = dict(DEFAULT_CONFIG.source_columns)
DT_HOURS = DEFAULT_CONFIG.dt_hours
PREDICTION_HORIZON = DEFAULT_CONFIG.prediction_horizon
LOOKBACK_STEPS = DEFAULT_CONFIG.lookback_steps
BATTERY_CAPACITY_KWH = DEFAULT_CONFIG.battery_capacity_kWh
BATTERY_POWER_KW = DEFAULT_CONFIG.battery_power_kW
ROUNDTRIP_EFFICIENCY = DEFAULT_CONFIG.roundtrip_efficiency
SOC_MIN_FRACTION = DEFAULT_CONFIG.min_soc_fraction
SOC_MAX_FRACTION = DEFAULT_CONFIG.max_soc_fraction
INITIAL_SOC_FRACTION = DEFAULT_CONFIG.initial_soc_fraction
GRID_POWER_MIN_MAX_KW = DEFAULT_CONFIG.grid_power_min_max_kW
SOC_MIN_MAX_KWH = DEFAULT_CONFIG.soc_min_max_kWh
STORAGE_POWER_MIN_MAX_KW = DEFAULT_CONFIG.storage_power_min_max_kW
SPLIT_DATES = DEFAULT_CONFIG.split_dates()
OBJECTIVE_WEIGHTS = dict(DEFAULT_CONFIG.objective_weights)


def configuration() -> HKUSTConfig:
    """Return the immutable default configuration."""

    return DEFAULT_CONFIG
