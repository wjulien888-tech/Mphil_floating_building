"""Build transparent CL1/CL2/CL3 proxy load scenarios from existing end uses.

The source load is synthetic and contains five end uses rather than physical
equipment or circuit measurements. This script therefore creates *functional
proxy* tiers, not a claimed as-built electrical inventory.

For every scenario and every 15-minute step:

    CL1_kW + CL2_kW + CL3_kW == load_kW

The allocation matrices are explicit and exported with the results so they can
later be replaced by design or metering evidence without changing downstream
resilience models.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INPUT_CSV = ROOT / "output" / "load_15min_2023.csv"
OUTPUT_CSV = ROOT / "output" / "load_tiers_15min_2023.csv"
MATRIX_CSV = ROOT / "output" / "load_tier_allocation_matrix.csv"
SUMMARY_CSV = ROOT / "output" / "load_tiers_15min_summary.csv"
SUMMARY_MD = ROOT / "output" / "load_tiers_15min_summary.md"

DT_HOURS = 0.25
END_USES = ("base_kW", "plug_kW", "lighting_kW", "hvac_kW", "aux_kW")
TIERS = ("CL1", "CL2", "CL3")

# These fractions are research scenarios, not measured equipment shares.
# Each end-use row must sum to 1.0 within each scenario.
ALLOCATIONS = {
    "emergency": {
        "base_kW": {"CL1": 0.30, "CL2": 0.30, "CL3": 0.40},
        "plug_kW": {"CL1": 0.05, "CL2": 0.25, "CL3": 0.70},
        "lighting_kW": {"CL1": 0.15, "CL2": 0.30, "CL3": 0.55},
        "hvac_kW": {"CL1": 0.00, "CL2": 0.10, "CL3": 0.90},
        "aux_kW": {"CL1": 0.30, "CL2": 0.30, "CL3": 0.40},
    },
    "minimum_operation": {
        "base_kW": {"CL1": 0.40, "CL2": 0.40, "CL3": 0.20},
        "plug_kW": {"CL1": 0.10, "CL2": 0.35, "CL3": 0.55},
        "lighting_kW": {"CL1": 0.20, "CL2": 0.40, "CL3": 0.40},
        "hvac_kW": {"CL1": 0.00, "CL2": 0.25, "CL3": 0.75},
        "aux_kW": {"CL1": 0.40, "CL2": 0.30, "CL3": 0.30},
    },
    "extended_operation": {
        "base_kW": {"CL1": 0.50, "CL2": 0.40, "CL3": 0.10},
        "plug_kW": {"CL1": 0.15, "CL2": 0.45, "CL3": 0.40},
        "lighting_kW": {"CL1": 0.30, "CL2": 0.50, "CL3": 0.20},
        "hvac_kW": {"CL1": 0.00, "CL2": 0.50, "CL3": 0.50},
        "aux_kW": {"CL1": 0.50, "CL2": 0.40, "CL3": 0.10},
    },
}

FUNCTIONAL_LABELS = {
    "base_kW": {
        "CL1": "controls, BMS/EMS and necessary communications proxy",
        "CL2": "minimum IT and core building operation proxy",
        "CL3": "other base consumption proxy",
    },
    "plug_kW": {
        "CL1": "critical communications/control plug proxy",
        "CL2": "essential work equipment proxy",
        "CL3": "general plug load proxy",
    },
    "lighting_kW": {
        "CL1": "emergency and escape-route lighting proxy",
        "CL2": "minimum task lighting proxy",
        "CL3": "general and decorative lighting proxy",
    },
    "hvac_kW": {
        "CL1": "no safety-critical HVAC assumed without project evidence",
        "CL2": "minimum ventilation, humidity and temperature control proxy",
        "CL3": "comfort HVAC proxy",
    },
    "aux_kW": {
        "CL1": "safety, dewatering and protection auxiliary proxy",
        "CL2": "minimum water and building-service auxiliary proxy",
        "CL3": "deferrable auxiliary proxy",
    },
}


def validate_allocations() -> None:
    for scenario, end_uses in ALLOCATIONS.items():
        if set(end_uses) != set(END_USES):
            raise ValueError(f"{scenario}: end-use columns do not match {END_USES}")
        for end_use, fractions in end_uses.items():
            if set(fractions) != set(TIERS):
                raise ValueError(f"{scenario}/{end_use}: tier columns do not match {TIERS}")
            total = sum(fractions.values())
            if abs(total - 1.0) > 1e-12:
                raise ValueError(f"{scenario}/{end_use}: fractions sum to {total}, not 1")


def write_matrix() -> None:
    MATRIX_CSV.parent.mkdir(parents=True, exist_ok=True)
    with MATRIX_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("scenario", "end_use", "tier", "fraction", "functional_proxy", "evidence_status"),
        )
        writer.writeheader()
        for scenario, end_uses in ALLOCATIONS.items():
            for end_use, fractions in end_uses.items():
                for tier, fraction in fractions.items():
                    writer.writerow(
                        {
                            "scenario": scenario,
                            "end_use": end_use,
                            "tier": tier,
                            "fraction": f"{fraction:.6f}",
                            "functional_proxy": FUNCTIONAL_LABELS[end_use][tier],
                            "evidence_status": "scenario assumption; replace when project evidence is available",
                        }
                    )


def build_timeseries() -> list[dict[str, str | float]]:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing {INPUT_CSV}")

    output_rows: list[dict[str, str | float]] = []
    max_reconstruction_error = defaultdict(float)

    with INPUT_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(("datetime", "load_kW", *END_USES)).difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Input is missing columns: {sorted(missing)}")

        for source in reader:
            end_use_values = {name: float(source[name]) for name in END_USES}
            total_load = float(source["load_kW"])
            row: dict[str, str | float] = {"datetime": source["datetime"], "load_kW": total_load}

            for scenario, allocations in ALLOCATIONS.items():
                tier_values = {
                    tier: sum(end_use_values[end_use] * allocations[end_use][tier] for end_use in END_USES)
                    for tier in TIERS
                }
                for tier, value in tier_values.items():
                    row[f"{scenario}_{tier}_kW"] = value
                reconstructed = sum(tier_values.values())
                max_reconstruction_error[scenario] = max(
                    max_reconstruction_error[scenario], abs(reconstructed - total_load)
                )

            output_rows.append(row)

    for scenario, error in max_reconstruction_error.items():
        if error > 1e-8:
            raise ValueError(f"{scenario}: maximum reconstruction error is {error}")
    return output_rows


def write_timeseries(rows: list[dict[str, str | float]]) -> None:
    fieldnames = ["datetime", "load_kW"]
    for scenario in ALLOCATIONS:
        fieldnames.extend(f"{scenario}_{tier}_kW" for tier in TIERS)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, str | float]]) -> list[dict[str, str | float]]:
    total_energy = sum(float(row["load_kW"]) for row in rows) * DT_HOURS
    summary_rows: list[dict[str, str | float]] = []
    for scenario in ALLOCATIONS:
        for tier in TIERS:
            col = f"{scenario}_{tier}_kW"
            energy = sum(float(row[col]) for row in rows) * DT_HOURS
            peak = max(float(row[col]) for row in rows)
            summary_rows.append(
                {
                    "scenario": scenario,
                    "tier": tier,
                    "annual_energy_kWh": energy,
                    "annual_energy_fraction": energy / total_energy,
                    "peak_power_kW": peak,
                }
            )
    return summary_rows


def write_summary(
    rows: list[dict[str, str | float]],
    timeseries_rows: list[dict[str, str | float]],
) -> None:
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    lookup = {(str(row["scenario"]), str(row["tier"])): row for row in rows}
    lines = [
        "# CL1/CL2/CL3 Proxy Load Scenario Summary",
        "",
        "These are transparent research scenarios built from the existing synthetic end uses. ",
        "They are not an as-built equipment or circuit inventory.",
        "",
        "| Scenario | CL1 | CL2 | CL3 | CL1+CL2 | Peak CL1 | Peak CL1+CL2 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scenario in ALLOCATIONS:
        tier_rows = [lookup[(scenario, tier)] for tier in TIERS]
        fractions = [float(row["annual_energy_fraction"]) for row in tier_rows]
        peak_cl1 = float(tier_rows[0]["peak_power_kW"])
        # Peak of the sum cannot be obtained by summing individual annual statistics.
        peak_cl1_cl2 = max(
            float(row[f"{scenario}_CL1_kW"]) + float(row[f"{scenario}_CL2_kW"])
            for row in timeseries_rows
        )
        lines.append(
            f"| {scenario} | {fractions[0] * 100:.1f}% | {fractions[1] * 100:.1f}% | "
            f"{fractions[2] * 100:.1f}% | {(fractions[0] + fractions[1]) * 100:.1f}% | "
            f"{peak_cl1:.2f} kW | {peak_cl1_cl2:.2f} kW |"
        )
    lines.extend(
        [
            "",
            "All scenarios reconstruct the original `load_kW` at every 15-minute step.",
            "Use `emergency` for short safety-only islanding, `minimum_operation` as the central scenario, ",
            "and `extended_operation` as a higher-service sensitivity case.",
            "",
        ]
    )
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    validate_allocations()
    write_matrix()
    rows = build_timeseries()
    write_timeseries(rows)
    summary_rows = summarize(rows)
    write_summary(summary_rows, rows)
    print(f"Wrote {len(rows):,} rows to {OUTPUT_CSV.relative_to(ROOT)}")
    print(f"Wrote allocation matrix to {MATRIX_CSV.relative_to(ROOT)}")
    print(f"Wrote summaries to {SUMMARY_CSV.relative_to(ROOT)} and {SUMMARY_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
