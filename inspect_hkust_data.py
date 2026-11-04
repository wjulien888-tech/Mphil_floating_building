"""
Inspect the HKUST Clear Water Bay PV + wind dataset.

The script scans the imported dataset, summarizes each time-series file, and
writes machine-readable and human-readable inventories under output/.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATASET_ROOT = ROOT / "HKUST_clea_bay_PV+wind_data"
TIME_SERIES_ROOT = DATASET_ROOT / "Dataset" / "Time series dataset"
OUT_DIR = ROOT / "output"

INVENTORY_CSV = OUT_DIR / "hkust_data_inventory.csv"
SUMMARY_MD = OUT_DIR / "hkust_data_summary.md"


@dataclass(frozen=True)
class FileSummary:
    relative_path: str
    category: str
    file_type: str
    columns: str
    rows: int
    start_time: str
    end_time: str
    median_interval_min: float | None
    missing_cells: int
    missing_cell_pct: float


def classify_file(path: Path) -> str:
    rel = path.relative_to(TIME_SERIES_ROOT)
    parts = rel.parts
    if "Meteorological dataset" in parts:
        idx = parts.index("Meteorological dataset")
        return f"Meteorological / {parts[idx + 1]}"
    if "PV generation dataset" in parts:
        optimizer = "with optimizer" if "with panel level optimizer" in str(rel) else "without optimizer"
        level = "inverter level" if "Inverter level dataset" in parts else "site level"
        return f"PV generation / {optimizer} / {level}"
    return "Other"


def read_timeseries(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, na_values=["NA", "N/A", ""])
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, na_values=["NA", "N/A", ""])
    raise ValueError(f"Unsupported file type: {path}")


def summarize_file(path: Path) -> FileSummary:
    rel = path.relative_to(DATASET_ROOT).as_posix()
    df = read_timeseries(path)
    columns = [str(c) for c in df.columns]

    if "Time" in df.columns:
        time = pd.to_datetime(df["Time"], errors="coerce")
    else:
        time = pd.Series(pd.NaT, index=df.index)

    valid_time = time.dropna().sort_values()
    if len(valid_time) >= 2:
        diffs = valid_time.diff().dropna().dt.total_seconds() / 60.0
        median_interval = float(diffs.median())
    else:
        median_interval = None

    missing_cells = int(df.isna().sum().sum())
    total_cells = int(df.shape[0] * df.shape[1])
    missing_pct = (missing_cells / total_cells * 100.0) if total_cells else 0.0

    return FileSummary(
        relative_path=rel,
        category=classify_file(path),
        file_type=path.suffix.lower().lstrip("."),
        columns=", ".join(columns),
        rows=int(len(df)),
        start_time="" if valid_time.empty else valid_time.iloc[0].isoformat(sep=" "),
        end_time="" if valid_time.empty else valid_time.iloc[-1].isoformat(sep=" "),
        median_interval_min=median_interval,
        missing_cells=missing_cells,
        missing_cell_pct=round(missing_pct, 4),
    )


def format_interval(value: float | None) -> str:
    if value is None:
        return "N/A"
    if value.is_integer():
        return f"{int(value)} min"
    return f"{value:.2f} min"


def build_markdown(inventory: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("# HKUST Clear Water Bay PV + Wind Data Summary")
    lines.append("")
    lines.append(f"- Dataset root: `{DATASET_ROOT.name}`")
    lines.append(f"- Time-series files inspected: {len(inventory)}")
    lines.append(f"- Total rows scanned: {int(inventory['rows'].sum()):,}")
    lines.append("")

    lines.append("## Category Overview")
    lines.append("")
    grouped = (
        inventory.groupby("category", dropna=False)
        .agg(files=("relative_path", "count"), rows=("rows", "sum"), avg_missing_pct=("missing_cell_pct", "mean"))
        .reset_index()
        .sort_values(["category"])
    )
    lines.append("| Category | Files | Rows | Avg missing cells |")
    lines.append("| --- | ---: | ---: | ---: |")
    for row in grouped.itertuples(index=False):
        lines.append(f"| {row.category} | {row.files} | {int(row.rows):,} | {row.avg_missing_pct:.3f}% |")
    lines.append("")

    lines.append("## Time Coverage By Category")
    lines.append("")
    lines.append("| Category | Earliest | Latest | Typical interval |")
    lines.append("| --- | --- | --- | --- |")
    for category, group in inventory.groupby("category", sort=True):
        starts = pd.to_datetime(group["start_time"], errors="coerce").dropna()
        ends = pd.to_datetime(group["end_time"], errors="coerce").dropna()
        intervals = group["median_interval_min"].dropna()
        interval_text = format_interval(float(intervals.mode().iloc[0])) if not intervals.empty else "N/A"
        start_text = "" if starts.empty else starts.min().isoformat(sep=" ")
        end_text = "" if ends.empty else ends.max().isoformat(sep=" ")
        lines.append(f"| {category} | {start_text} | {end_text} | {interval_text} |")
    lines.append("")

    lines.append("## Candidate Uses")
    lines.append("")
    lines.append("- Calibrate the current PVGIS-based `pv_generation.py` output with measured HKUST PV variability.")
    lines.append("- Build empirical PV disturbance factors from cloudy, rainy, and high-ramp days.")
    lines.append("- Use measured wind speed and direction to estimate wind generation with a small-turbine power curve.")
    lines.append("- Aggregate PV, wind, load, and weather into one hourly dispatch input table.")
    lines.append("")
    lines.append("## Inventory Output")
    lines.append("")
    lines.append(f"- Full file-level inventory: `{INVENTORY_CSV.relative_to(ROOT).as_posix()}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    if not TIME_SERIES_ROOT.exists():
        raise FileNotFoundError(f"Missing dataset folder: {TIME_SERIES_ROOT}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(
        [
            p
            for p in TIME_SERIES_ROOT.rglob("*")
            if p.is_file() and p.suffix.lower() in {".csv", ".xlsx", ".xls"}
        ]
    )
    summaries = [summarize_file(path) for path in files]
    inventory = pd.DataFrame([summary.__dict__ for summary in summaries])
    inventory.to_csv(INVENTORY_CSV, index=False)
    SUMMARY_MD.write_text(build_markdown(inventory), encoding="utf-8")

    print(f"Inspected {len(inventory)} files")
    print(f"Wrote {INVENTORY_CSV.relative_to(ROOT)}")
    print(f"Wrote {SUMMARY_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
