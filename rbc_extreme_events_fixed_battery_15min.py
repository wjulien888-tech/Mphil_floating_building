"""
Fixed-battery RBC dispatch for extracted extreme low-PV event windows.

This script keeps the battery configuration fixed at the current project
baseline:

    215 kWh capacity / 50 kW power / 0.92 round-trip efficiency

It runs the existing RBC controller inside each extracted low-PV event window
and writes event-level tables and slide-ready SVG figures.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from dispatch_rbc_15min import (
    DISPATCH_INPUT_CSV,
    DT_HOURS,
    BatteryParams,
    run_rbc,
    summarize_dispatch,
)


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "output"
PLOTS_DIR = ROOT / "plots"

EVENTS_CSV = OUT_DIR / "extreme_low_pv_events_15min.csv"
SUMMARY_CSV = OUT_DIR / "rbc_extreme_events_fixed_battery_summary.csv"
SUMMARY_MD = OUT_DIR / "rbc_extreme_events_fixed_battery_summary.md"
SUMMARY_SVG = PLOTS_DIR / "rbc_extreme_events_fixed_battery_summary.svg"

BATTERY = BatteryParams()


def _event_label(hours: int) -> str:
    return f"{hours}h" if hours < 168 else "7d"


def run_event_dispatches(df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for event in events.itertuples():
        hours = int(event.window_hours)
        start = pd.Timestamp(event.start)
        end = pd.Timestamp(event.end)
        event_df = df[(df["datetime"] >= start) & (df["datetime"] <= end)].copy()
        dispatched = run_rbc(event_df, BATTERY)
        summary = summarize_dispatch(dispatched, BATTERY).iloc[0].to_dict()

        detail_csv = OUT_DIR / f"rbc_extreme_event_{_event_label(hours)}_fixed_battery_15min.csv"
        dispatched.to_csv(detail_csv, index=False)

        rows.append(
            {
                "window_hours": hours,
                "event_label": _event_label(hours),
                "start": start,
                "end": end,
                "time_steps": len(dispatched),
                "battery_capacity_kWh": BATTERY.capacity_kWh,
                "battery_power_kW": BATTERY.power_kW,
                "initial_soc_kWh": BATTERY.initial_soc_kWh,
                "load_kWh": summary["load_kWh"],
                "pv_kWh": summary["pv_kWh"],
                "served_load_kWh": summary["served_load_kWh"],
                "unmet_load_kWh": summary["unmet_load_kWh"],
                "unmet_load_fraction": summary["unmet_load_fraction"],
                "curtailment_kWh": summary["curtailment_kWh"],
                "battery_discharge_kWh_ac": summary["battery_discharge_kWh_ac"],
                "min_soc_kWh": summary["min_soc_kWh"],
                "end_soc_kWh": summary["end_soc_kWh"],
                "max_continuous_unmet_hours": summary["max_continuous_unmet_hours"],
                "rainfall_mm": float(event.rainfall_mm),
                "mean_daylight_disturbance_factor": float(event.mean_daylight_disturbance_factor),
                "detail_csv": str(detail_csv.relative_to(ROOT)),
            }
        )

        write_event_dispatch_svg(dispatched, hours, start, end)

    out = pd.DataFrame(rows)
    out["unmet_load_pct"] = out["unmet_load_fraction"] * 100.0
    out["served_load_pct"] = 100.0 - out["unmet_load_pct"]
    return out


def _x_scale(value: float, min_value: float, max_value: float, left: float, width: float) -> float:
    if max_value == min_value:
        return left
    return left + (value - min_value) / (max_value - min_value) * width


def _y_scale(value: float, max_value: float, top: float, height: float) -> float:
    if max_value <= 0:
        return top + height
    return top + (max_value - value) / max_value * height


def _line_path(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def write_event_dispatch_svg(out: pd.DataFrame, hours: int, start: pd.Timestamp, end: pd.Timestamp) -> None:
    width = 1060
    height = 650
    margin_left = 72
    margin_right = 34
    top_1 = 72
    h_1 = 320
    top_2 = 455
    h_2 = 130
    chart_w = width - margin_left - margin_right

    plot = out.copy().reset_index(drop=True)
    plot["hour_from_start"] = (
        (plot["datetime"] - plot["datetime"].iloc[0]).dt.total_seconds() / 3600.0
    )
    x_min = 0.0
    x_max = max(float(plot["hour_from_start"].iloc[-1]), 1.0)

    power_max = float(
        max(
            plot["load_kW"].max(),
            plot["pv_kW"].max(),
            plot["battery_discharge_kW"].max(),
            plot["unmet_load_kW"].max(),
        )
        * 1.16
    )
    soc_max = BATTERY.capacity_kWh

    load_points = []
    pv_points = []
    soc_points = []
    for row in plot.itertuples():
        x = _x_scale(float(row.hour_from_start), x_min, x_max, margin_left, chart_w)
        load_points.append((x, _y_scale(float(row.load_kW), power_max, top_1, h_1)))
        pv_points.append((x, _y_scale(float(row.pv_kW), power_max, top_1, h_1)))
        soc_points.append((x, _y_scale(float(row.battery_soc_kWh), soc_max, top_2, h_2)))

    bar_w = max(1.0, chart_w / max(len(plot), 1) * 0.7)
    unmet_bars = []
    discharge_bars = []
    charge_bars = []
    for row in plot.itertuples():
        x = _x_scale(float(row.hour_from_start), x_min, x_max, margin_left, chart_w) - bar_w / 2
        if row.unmet_load_kW > 1e-6:
            y = _y_scale(float(row.unmet_load_kW), power_max, top_1, h_1)
            unmet_bars.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
                f'height="{top_1 + h_1 - y:.1f}" fill="#c44e52" opacity="0.45" />'
            )
        if row.battery_discharge_kW > 1e-6:
            y = _y_scale(float(row.battery_discharge_kW), power_max, top_1, h_1)
            discharge_bars.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
                f'height="{top_1 + h_1 - y:.1f}" fill="#8f4f4f" opacity="0.22" />'
            )
        if row.battery_charge_kW > 1e-6:
            y = _y_scale(float(row.battery_charge_kW), power_max, top_1, h_1)
            charge_bars.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
                f'height="{top_1 + h_1 - y:.1f}" fill="#5aa469" opacity="0.20" />'
            )

    x_ticks = [0, hours / 4, hours / 2, hours * 3 / 4, hours]
    x_ticks = [tick for tick in x_ticks if tick <= x_max + 0.5]
    y_ticks_power = [power_max * i / 4 for i in range(5)]
    y_ticks_soc = [soc_max * i / 4 for i in range(5)]

    x_grid = "\n".join(
        f'<line x1="{_x_scale(t, x_min, x_max, margin_left, chart_w):.1f}" y1="{top_1}" '
        f'x2="{_x_scale(t, x_min, x_max, margin_left, chart_w):.1f}" y2="{top_2 + h_2}" '
        f'stroke="#e6e0d6" stroke-width="1" />'
        for t in x_ticks
    )
    power_y_grid = "\n".join(
        f'<line x1="{margin_left}" y1="{_y_scale(y, power_max, top_1, h_1):.1f}" '
        f'x2="{margin_left + chart_w}" y2="{_y_scale(y, power_max, top_1, h_1):.1f}" '
        f'stroke="#e6e0d6" stroke-width="1" />'
        for y in y_ticks_power
    )
    soc_y_grid = "\n".join(
        f'<line x1="{margin_left}" y1="{_y_scale(y, soc_max, top_2, h_2):.1f}" '
        f'x2="{margin_left + chart_w}" y2="{_y_scale(y, soc_max, top_2, h_2):.1f}" '
        f'stroke="#e6e0d6" stroke-width="1" />'
        for y in y_ticks_soc
    )
    x_labels = "\n".join(
        f'<text x="{_x_scale(t, x_min, x_max, margin_left, chart_w):.1f}" y="{height - 24}" '
        f'text-anchor="middle" font-size="12" fill="#2f3437">{t:.0f}</text>'
        for t in x_ticks
    )
    power_y_labels = "\n".join(
        f'<text x="{margin_left - 10}" y="{_y_scale(y, power_max, top_1, h_1) + 4:.1f}" '
        f'text-anchor="end" font-size="12" fill="#2f3437">{y:.0f}</text>'
        for y in y_ticks_power
    )
    soc_y_labels = "\n".join(
        f'<text x="{margin_left - 10}" y="{_y_scale(y, soc_max, top_2, h_2) + 4:.1f}" '
        f'text-anchor="end" font-size="12" fill="#2f3437">{y:.0f}</text>'
        for y in y_ticks_soc
    )

    total_load = float(plot["load_kW"].sum() * DT_HOURS)
    total_pv = float(plot["pv_kW"].sum() * DT_HOURS)
    total_unmet = float(plot["unmet_load_kW"].sum() * DT_HOURS)
    unmet_pct = total_unmet / total_load * 100 if total_load else 0.0

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#fbfaf7" />
  <text x="{margin_left}" y="34" font-size="22" font-weight="700" fill="#263238">Fixed-battery RBC during {_event_label(hours)} low-PV event</text>
  <text x="{margin_left}" y="55" font-size="13" fill="#687076">{start:%Y-%m-%d %H:%M} to {end:%Y-%m-%d %H:%M}; battery 215 kWh / 50 kW, initial SOC 50%</text>
  {x_grid}
  {power_y_grid}
  {soc_y_grid}
  {''.join(charge_bars)}
  {''.join(discharge_bars)}
  {''.join(unmet_bars)}
  <polyline points="{_line_path(load_points)}" fill="none" stroke="#334e68" stroke-width="2.3" stroke-linejoin="round" />
  <polyline points="{_line_path(pv_points)}" fill="none" stroke="#d99000" stroke-width="2.3" stroke-linejoin="round" />
  <polyline points="{_line_path(soc_points)}" fill="none" stroke="#2f6f73" stroke-width="2.6" stroke-linejoin="round" />
  <line x1="{margin_left}" y1="{top_1 + h_1}" x2="{margin_left + chart_w}" y2="{top_1 + h_1}" stroke="#333" stroke-width="1.4" />
  <line x1="{margin_left}" y1="{top_1}" x2="{margin_left}" y2="{top_1 + h_1}" stroke="#333" stroke-width="1.4" />
  <line x1="{margin_left}" y1="{top_2 + h_2}" x2="{margin_left + chart_w}" y2="{top_2 + h_2}" stroke="#333" stroke-width="1.4" />
  <line x1="{margin_left}" y1="{top_2}" x2="{margin_left}" y2="{top_2 + h_2}" stroke="#333" stroke-width="1.4" />
  {x_labels}
  {power_y_labels}
  {soc_y_labels}
  <text x="{margin_left + chart_w / 2:.1f}" y="{height - 5}" text-anchor="middle" font-size="14" fill="#263238">Hours from event start</text>
  <text x="18" y="{top_1 + h_1 / 2:.1f}" transform="rotate(-90 18 {top_1 + h_1 / 2:.1f})" text-anchor="middle" font-size="14" fill="#263238">Power (kW)</text>
  <text x="18" y="{top_2 + h_2 / 2:.1f}" transform="rotate(-90 18 {top_2 + h_2 / 2:.1f})" text-anchor="middle" font-size="14" fill="#263238">SOC (kWh)</text>
  <text x="{width - 310}" y="92" font-size="13" fill="#263238">Load {total_load:.1f} kWh</text>
  <text x="{width - 310}" y="113" font-size="13" fill="#263238">PV {total_pv:.1f} kWh</text>
  <text x="{width - 310}" y="134" font-size="13" fill="#263238">Unmet {total_unmet:.1f} kWh ({unmet_pct:.1f}%)</text>
  <line x1="{width - 310}" y1="166" x2="{width - 282}" y2="166" stroke="#334e68" stroke-width="3" />
  <text x="{width - 274}" y="170" font-size="12" fill="#263238">Load</text>
  <line x1="{width - 310}" y1="188" x2="{width - 282}" y2="188" stroke="#d99000" stroke-width="3" />
  <text x="{width - 274}" y="192" font-size="12" fill="#263238">Raw disturbed PV</text>
  <line x1="{width - 310}" y1="210" x2="{width - 282}" y2="210" stroke="#2f6f73" stroke-width="3" />
  <text x="{width - 274}" y="214" font-size="12" fill="#263238">Battery SOC</text>
  <rect x="{width - 310}" y="226" width="18" height="12" fill="#c44e52" opacity="0.45" />
  <text x="{width - 274}" y="237" font-size="12" fill="#263238">Unmet load</text>
</svg>
"""
    path = PLOTS_DIR / f"rbc_extreme_event_{_event_label(hours)}_fixed_battery_dispatch.svg"
    path.write_text(svg, encoding="utf-8")


def write_summary_svg(summary: pd.DataFrame) -> None:
    width = 980
    height = 560
    margin_left = 80
    margin_right = 40
    margin_top = 72
    margin_bottom = 70
    chart_w = width - margin_left - margin_right
    chart_h = height - margin_top - margin_bottom
    groups = list(summary["event_label"])
    y_max = float(max(summary["load_kWh"].max(), summary["served_load_kWh"].max() + summary["unmet_load_kWh"].max()) * 1.15)
    bar_group_w = chart_w / len(groups)
    bar_w = bar_group_w * 0.22

    def y(value: float) -> float:
        return _y_scale(value, y_max, margin_top, chart_h)

    bars = []
    labels = []
    for i, row in enumerate(summary.itertuples()):
        center = margin_left + bar_group_w * (i + 0.5)
        values = [
            ("PV", row.pv_kWh, "#d99000", center - bar_w * 1.4),
            ("served", row.served_load_kWh, "#5aa469", center),
            ("unmet", row.unmet_load_kWh, "#c44e52", center + bar_w * 1.4),
        ]
        for _, value, color, x in values:
            y_top = y(float(value))
            bars.append(
                f'<rect x="{x - bar_w / 2:.1f}" y="{y_top:.1f}" width="{bar_w:.1f}" '
                f'height="{margin_top + chart_h - y_top:.1f}" fill="{color}" opacity="0.86" />'
            )
        labels.append(
            f'<text x="{center:.1f}" y="{height - 34}" text-anchor="middle" font-size="13" fill="#263238">{row.event_label}</text>'
        )
        labels.append(
            f'<text x="{center:.1f}" y="{height - 16}" text-anchor="middle" font-size="11" fill="#687076">{row.unmet_load_pct:.1f}% unmet</text>'
        )

    y_ticks = [y_max * i / 5 for i in range(6)]
    y_grid = "\n".join(
        f'<line x1="{margin_left}" y1="{y(t):.1f}" x2="{margin_left + chart_w}" y2="{y(t):.1f}" stroke="#e6e0d6" />'
        for t in y_ticks
    )
    y_labels = "\n".join(
        f'<text x="{margin_left - 10}" y="{y(t) + 4:.1f}" text-anchor="end" font-size="12" fill="#263238">{t:.0f}</text>'
        for t in y_ticks
    )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#fbfaf7" />
  <text x="{margin_left}" y="34" font-size="22" font-weight="700" fill="#263238">Fixed-battery RBC in extracted low-PV events</text>
  <text x="{margin_left}" y="55" font-size="13" fill="#687076">Battery fixed at 215 kWh / 50 kW, initial SOC 50%; raw disturbed PV scenario</text>
  {y_grid}
  {''.join(bars)}
  <line x1="{margin_left}" y1="{margin_top + chart_h}" x2="{margin_left + chart_w}" y2="{margin_top + chart_h}" stroke="#333" stroke-width="1.4" />
  <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + chart_h}" stroke="#333" stroke-width="1.4" />
  {y_labels}
  {''.join(labels)}
  <text x="20" y="{margin_top + chart_h / 2:.1f}" transform="rotate(-90 20 {margin_top + chart_h / 2:.1f})" text-anchor="middle" font-size="14" fill="#263238">Energy (kWh)</text>
  <rect x="{width - 220}" y="82" width="16" height="12" fill="#d99000" opacity="0.86" />
  <text x="{width - 196}" y="93" font-size="13" fill="#263238">PV</text>
  <rect x="{width - 220}" y="106" width="16" height="12" fill="#5aa469" opacity="0.86" />
  <text x="{width - 196}" y="117" font-size="13" fill="#263238">Served load</text>
  <rect x="{width - 220}" y="130" width="16" height="12" fill="#c44e52" opacity="0.86" />
  <text x="{width - 196}" y="141" font-size="13" fill="#263238">Unmet load</text>
</svg>
"""
    SUMMARY_SVG.write_text(svg, encoding="utf-8")


def write_summary_md(summary: pd.DataFrame) -> None:
    lines = [
        "# Fixed-Battery RBC In Extreme Low-PV Events",
        "",
        "Battery parameters are fixed at `215 kWh / 50 kW`, with initial SOC `50%`.",
        "",
        "| Event | Start | End | PV | Load | Served | Unmet | Unmet % | Min SOC | Max continuous unmet |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples():
        lines.append(
            f"| {row.event_label} | {pd.Timestamp(row.start):%Y-%m-%d %H:%M} | "
            f"{pd.Timestamp(row.end):%Y-%m-%d %H:%M} | {row.pv_kWh:.1f} kWh | "
            f"{row.load_kWh:.1f} kWh | {row.served_load_kWh:.1f} kWh | "
            f"{row.unmet_load_kWh:.1f} kWh | {row.unmet_load_pct:.1f}% | "
            f"{row.min_soc_kWh:.1f} kWh | {row.max_continuous_unmet_hours:.1f} h |"
        )
    lines.extend(
        [
            "",
            "Slide figures:",
            "",
            "- `plots/rbc_extreme_events_fixed_battery_summary.svg`",
            "- `plots/rbc_extreme_event_24h_fixed_battery_dispatch.svg`",
            "- `plots/rbc_extreme_event_48h_fixed_battery_dispatch.svg`",
            "- `plots/rbc_extreme_event_72h_fixed_battery_dispatch.svg`",
            "- `plots/rbc_extreme_event_7d_fixed_battery_dispatch.svg`",
            "",
        ]
    )
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not DISPATCH_INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing {DISPATCH_INPUT_CSV}; run build_pv_disturbance_15min.py first.")
    if not EVENTS_CSV.exists():
        raise FileNotFoundError(f"Missing {EVENTS_CSV}; run extract_extreme_pv_events_15min.py first.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DISPATCH_INPUT_CSV, parse_dates=["datetime"])
    events = pd.read_csv(EVENTS_CSV, parse_dates=["start", "end"])
    summary = run_event_dispatches(df, events)
    summary.to_csv(SUMMARY_CSV, index=False)
    write_summary_svg(summary)
    write_summary_md(summary)

    print("Fixed-battery RBC event dispatch complete")
    for row in summary.itertuples():
        print(
            f"{row.event_label}: load {row.load_kWh:.1f} kWh, PV {row.pv_kWh:.1f} kWh, "
            f"unmet {row.unmet_load_kWh:.1f} kWh ({row.unmet_load_pct:.1f}%), "
            f"min SOC {row.min_soc_kWh:.1f} kWh"
        )
    print(f"Saved: {SUMMARY_CSV.relative_to(ROOT)}")
    print(f"Saved: {SUMMARY_MD.relative_to(ROOT)}")
    print(f"Saved: {SUMMARY_SVG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
