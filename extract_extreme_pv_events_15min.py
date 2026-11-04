"""
Extract low-PV event windows from the 15-minute disturbed PV scenario.

The output turns the planned E2 extreme-weather experiment into a concrete
dataset: lowest 24 h, 48 h, 72 h, and 7 d windows based on floating-building
raw disturbed PV energy.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output"
PLOTS_DIR = ROOT / "plots"

DISPATCH_INPUT_CSV = OUT_DIR / "dispatch_input_15min_disturbed_2023.csv"
EVENTS_CSV = OUT_DIR / "extreme_low_pv_events_15min.csv"
EVENTS_MD = OUT_DIR / "extreme_low_pv_events_15min.md"
EVENTS_SVG = PLOTS_DIR / "extreme_low_pv_events_15min.svg"

DT_HOURS = 0.25
WINDOWS_HOURS = [24, 48, 72, 168]


def extract_events(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    work = df.sort_values("datetime").reset_index(drop=True).copy()
    work["pv_energy_kWh_step"] = work["pv_floating_disturbed_raw_kW"] * DT_HOURS
    work["load_energy_kWh_step"] = work["load_kW"] * DT_HOURS

    for hours in WINDOWS_HOURS:
        steps = int(hours / DT_HOURS)
        pv_roll = work["pv_energy_kWh_step"].rolling(steps, min_periods=steps).sum()
        idx = int(pv_roll.idxmin())
        start_idx = idx - steps + 1
        event = work.iloc[start_idx : idx + 1]

        pv_kwh = float(event["pv_energy_kWh_step"].sum())
        load_kwh = float(event["load_energy_kWh_step"].sum())
        baseline_pv_kwh = float((event["pvgis_floating_baseline_kW"] * DT_HOURS).sum())
        annual_preserved_pv_kwh = float(
            (event["pv_floating_disturbed_annual_preserved_kW"] * DT_HOURS).sum()
        )
        mean_factor_daylight = float(
            event.loc[event["pvgis_floating_baseline_kW"] > 0.1, "disturbance_factor"].mean()
        )

        rows.append(
            {
                "window_hours": hours,
                "start": event["datetime"].iloc[0],
                "end": event["datetime"].iloc[-1],
                "raw_disturbed_pv_kWh": pv_kwh,
                "annual_preserved_pv_kWh": annual_preserved_pv_kwh,
                "pvgis_baseline_pv_kWh": baseline_pv_kwh,
                "load_kWh": load_kwh,
                "pv_to_load_ratio": pv_kwh / load_kwh if load_kwh else float("nan"),
                "raw_to_baseline_pv_ratio": pv_kwh / baseline_pv_kwh if baseline_pv_kwh else float("nan"),
                "mean_daylight_disturbance_factor": mean_factor_daylight,
                "mean_irradiance_W_m2": float(event["irradiance_W_m2"].mean()),
                "rainfall_mm": float(event["rainfall_mm"].sum()),
                "max_load_kW": float(event["load_kW"].max()),
                "min_raw_pv_kW": float(event["pv_floating_disturbed_raw_kW"].min()),
                "max_raw_pv_kW": float(event["pv_floating_disturbed_raw_kW"].max()),
            }
        )

    return pd.DataFrame(rows)


def write_markdown(events: pd.DataFrame) -> None:
    display = events.copy()
    for col in ["start", "end"]:
        display[col] = pd.to_datetime(display[col]).dt.strftime("%Y-%m-%d %H:%M")

    lines = [
        "# Extreme Low-PV Event Windows",
        "",
        "These windows are extracted from `dispatch_input_15min_disturbed_2023.csv` using",
        "the lowest rolling raw disturbed PV energy for each duration.",
        "",
        "| Window | Start | End | Raw PV (kWh) | Load (kWh) | PV/Load | Raw/PVGIS | Daylight disturbance | Rainfall (mm) |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in display.itertuples():
        lines.append(
            f"| {int(row.window_hours)} h | {row.start} | {row.end} | "
            f"{row.raw_disturbed_pv_kWh:.1f} | {row.load_kWh:.1f} | "
            f"{row.pv_to_load_ratio:.2f} | {row.raw_to_baseline_pv_ratio:.2f} | "
            f"{row.mean_daylight_disturbance_factor:.2f} | {row.rainfall_mm:.1f} |"
        )

    lines.extend(
        [
            "",
            "Interpretation for slides:",
            "",
            "- These are candidate E2 extreme-weather / low-PV stress-test windows.",
            "- They are derived from measured HKUST PV disturbance factors applied to the floating-building PVGIS baseline.",
            "- The next dispatch step is to run RBC / MILP / MPC under grid-connected and islanded assumptions for these windows.",
            "",
        ]
    )
    EVENTS_MD.write_text("\n".join(lines), encoding="utf-8")


def write_event_chart(df: pd.DataFrame, events: pd.DataFrame) -> None:
    width = 1060
    height = 540
    margin_left = 70
    margin_right = 32
    margin_top = 88
    margin_bottom = 54
    chart_w = width - margin_left - margin_right
    chart_h = height - margin_top - margin_bottom

    daily = df.copy()
    daily["date"] = daily["datetime"].dt.floor("D")
    daily_energy = (
        daily.groupby("date")[["pv_floating_disturbed_raw_kW", "load_kW"]].sum()
        * DT_HOURS
    ).reset_index()
    daily_energy["day_index"] = range(len(daily_energy))

    y_max = float(max(daily_energy["load_kW"].max(), daily_energy["pv_floating_disturbed_raw_kW"].max()) * 1.12)

    def x_scale(day_index: float) -> float:
        return margin_left + day_index / (len(daily_energy) - 1) * chart_w

    def y_scale(value: float) -> float:
        return margin_top + (y_max - value) / y_max * chart_h

    pv_points = " ".join(
        f"{x_scale(float(row.day_index)):.1f},{y_scale(float(row.pv_floating_disturbed_raw_kW)):.1f}"
        for row in daily_energy.itertuples()
    )
    load_points = " ".join(
        f"{x_scale(float(row.day_index)):.1f},{y_scale(float(row.load_kW)):.1f}"
        for row in daily_energy.itertuples()
    )

    event_rects = []
    event_labels = []
    color_by_window = {
        24: "#d99000",
        48: "#c75c5c",
        72: "#6b8f3a",
        168: "#665a9e",
    }
    for idx, row in enumerate(events.itertuples()):
        start_day = pd.Timestamp(row.start).floor("D")
        end_day = pd.Timestamp(row.end).floor("D")
        start_index = int(daily_energy.index[daily_energy["date"] == start_day][0])
        end_index = int(daily_energy.index[daily_energy["date"] == end_day][0])
        x = x_scale(start_index)
        w = max(4.0, x_scale(end_index) - x + chart_w / len(daily_energy))
        color = color_by_window[int(row.window_hours)]
        event_rects.append(
            f'<rect x="{x:.1f}" y="{margin_top}" width="{w:.1f}" height="{chart_h}" '
            f'fill="{color}" opacity="0.13" />'
        )
        label_y = margin_top + 18 + idx * 18
        event_labels.append(
            f'<rect x="{width - 266}" y="{label_y - 11}" width="12" height="12" fill="{color}" opacity="0.55" />'
            f'<text x="{width - 248}" y="{label_y}" font-size="12" fill="#263238">'
            f'{int(row.window_hours)} h: {pd.Timestamp(row.start).strftime("%m-%d")} to {pd.Timestamp(row.end).strftime("%m-%d")}</text>'
        )

    month_ticks = daily_energy[daily_energy["date"].dt.day == 1]
    x_labels = "\n".join(
        f'<text x="{x_scale(float(row.day_index)):.1f}" y="{height - 22}" text-anchor="middle" '
        f'font-size="11" fill="#2f3437">{row.date.strftime("%b")}</text>'
        for row in month_ticks.itertuples()
    )
    x_grid = "\n".join(
        f'<line x1="{x_scale(float(row.day_index)):.1f}" y1="{margin_top}" '
        f'x2="{x_scale(float(row.day_index)):.1f}" y2="{margin_top + chart_h}" '
        f'stroke="#e6e0d6" stroke-width="1" />'
        for row in month_ticks.itertuples()
    )
    y_ticks = [y_max * i / 5 for i in range(6)]
    y_grid = "\n".join(
        f'<line x1="{margin_left}" y1="{y_scale(y):.1f}" x2="{margin_left + chart_w}" '
        f'y2="{y_scale(y):.1f}" stroke="#e6e0d6" stroke-width="1" />'
        for y in y_ticks
    )
    y_labels = "\n".join(
        f'<text x="{margin_left - 10}" y="{y_scale(y) + 4:.1f}" text-anchor="end" '
        f'font-size="12" fill="#2f3437">{y:.0f}</text>'
        for y in y_ticks
    )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#fbfaf7" />
  <text x="{margin_left}" y="34" font-size="22" font-weight="700" fill="#263238">Extracted low-PV event windows</text>
  <text x="{margin_left}" y="54" font-size="13" fill="#687076">Daily raw disturbed PV and load, with lowest rolling PV windows highlighted</text>
  <line x1="{margin_left}" y1="74" x2="{margin_left + 28}" y2="74" stroke="#d99000" stroke-width="3" />
  <text x="{margin_left + 36}" y="78" font-size="12" fill="#263238">Raw disturbed PV</text>
  <line x1="{margin_left + 174}" y1="74" x2="{margin_left + 202}" y2="74" stroke="#334e68" stroke-width="3" />
  <text x="{margin_left + 210}" y="78" font-size="12" fill="#263238">Load</text>
  {x_grid}
  {y_grid}
  {''.join(event_rects)}
  <polyline points="{pv_points}" fill="none" stroke="#d99000" stroke-width="2.0" stroke-linejoin="round" />
  <polyline points="{load_points}" fill="none" stroke="#334e68" stroke-width="1.8" stroke-linejoin="round" opacity="0.82" />
  <line x1="{margin_left}" y1="{margin_top + chart_h}" x2="{margin_left + chart_w}" y2="{margin_top + chart_h}" stroke="#333" stroke-width="1.4" />
  <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + chart_h}" stroke="#333" stroke-width="1.4" />
  {x_labels}
  {y_labels}
  <text x="18" y="{margin_top + chart_h / 2:.1f}" transform="rotate(-90 18 {margin_top + chart_h / 2:.1f})" text-anchor="middle" font-size="14" fill="#263238">Daily energy (kWh)</text>
  {''.join(event_labels)}
</svg>
"""
    EVENTS_SVG.write_text(svg, encoding="utf-8")


def main() -> None:
    if not DISPATCH_INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing {DISPATCH_INPUT_CSV}; run build_pv_disturbance_15min.py first.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DISPATCH_INPUT_CSV, parse_dates=["datetime"])
    events = extract_events(df)
    events.to_csv(EVENTS_CSV, index=False)
    write_markdown(events)
    write_event_chart(df, events)

    print(f"Extracted {len(events)} low-PV windows")
    for row in events.itertuples():
        print(
            f"{int(row.window_hours):>3} h: {pd.Timestamp(row.start):%Y-%m-%d %H:%M} "
            f"to {pd.Timestamp(row.end):%Y-%m-%d %H:%M}, "
            f"PV {row.raw_disturbed_pv_kWh:.1f} kWh, load {row.load_kWh:.1f} kWh"
        )
    print(f"Saved: {EVENTS_CSV.relative_to(ROOT)}")
    print(f"Saved: {EVENTS_MD.relative_to(ROOT)}")
    print(f"Saved: {EVENTS_SVG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
