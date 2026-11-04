"""Explicit, process-wide hardware override shared by scenario MPC and M1.

No environment setting retains the historical configuration. Profiles affect
runtime dispatch hardware, never rewrite forecast checkpoint metadata.
"""
from dataclasses import replace
import json
import math
from pathlib import Path


def load_hardware_profile(base, path):
    if not path:
        return base
    profile = json.loads(Path(path).expanduser().read_text())
    overrides = profile['hardware']
    allowed = {'battery_capacity_kWh', 'battery_power_kW'}
    if not isinstance(overrides, dict) or not overrides or set(overrides) - allowed:
        raise ValueError('Hardware profile must override only battery capacity and/or power')
    for name, value in overrides.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be a finite positive number')
    return replace(base, **overrides)
