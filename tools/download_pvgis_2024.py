"""Download official PVGIS 6 inputs, preserving responses and request receipts.

2023 is downloaded only for a version-overlap audit; 2024-03-01 is a planning
tail for the last February MPC decision. No model training is performed.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data/pvgis6_2023_2024_20261006"
BASE = "https://photovoltaic-geographic-information-system.ec.europa.eu/api/v6"
# PVGIS 5 used south=0, east=-90, west=90, north=-179. Version 6 uses north=0.
SURFACES = {"south_45": (45, 180), "east_45": (45, 90),
            "west_45": (45, 270), "north_45": (45, 1), "deck": (0, 180)}
START, END = "2023-01-01T00:00:00", "2024-03-01T23:00:00"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parameters(surface):
    tilt, orientation = SURFACES[surface]
    return dict(latitude=22.896, longitude=113.479, surface_tilt=tilt,
                surface_orientation=orientation, start_time=START, end_time=END,
                frequency="Hourly", timezone="UTC", irradiance_source="ERA5",
                horizon_profile="PVGIS", albedo=0.2, verbose=9, csv="data.csv",
                solar_position_model="NOAA", solar_incidence_model="Iqbal")


def download(output=DEFAULT_OUTPUT):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "schema_version": 1, "api_version": "6", "created_utc": datetime.now(timezone.utc).isoformat(),
        "requests": [], "purpose": "2023 overlap audit and 2024 Jan-Feb extension plus planning tail",
        "documentation": BASE + "/redoc", "start_utc": START, "end_utc_inclusive": END,
    }
    for surface in SURFACES:
        params = parameters(surface)
        url = BASE + "/power/broadband?" + urlencode(params)
        filename = f"{surface}.csv"
        target = output / filename
        old = next((r for r in manifest["requests"] if r["surface"] == surface), None)
        if old:
            if old["url"] != url or not target.exists() or digest(target) != old["sha256"]:
                raise ValueError(f"Existing download differs from receipt: {surface}")
            print(f"Verified cached {surface}", flush=True)
            continue
        if target.exists():
            raise FileExistsError(f"Unreceipted file: {target}")
        part = output / (filename + ".part")
        headers = output / (filename + ".headers.txt")
        print(f"Downloading {surface} from PVGIS 6...", flush=True)
        subprocess.run(["curl", "-sSL", "--fail-with-body", "--max-time", "55",
                        "--retry", "2", "--retry-delay", "2", "-D", str(headers),
                        "-o", str(part), url], check=True)
        with part.open() as stream:
            reader = csv.DictReader(stream)
            required = {"Time", "Temperature", "Global Inclined Irradiance Before Reflectivity"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"Unexpected PVGIS CSV columns: {reader.fieldnames}")
            count = sum(1 for _ in reader)
        part.replace(target)
        manifest["requests"].append(dict(surface=surface, parameters=params, url=url,
            file=filename, rows=count, sha256=digest(target), size_bytes=target.stat().st_size,
            retrieved_utc=datetime.now(timezone.utc).isoformat()))
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(manifest_path)
        print(f"Saved {surface}: {count} hourly rows", flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    download(parser.parse_args().output)
