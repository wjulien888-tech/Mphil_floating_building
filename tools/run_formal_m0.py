"""Run formal M0: two-layer LSTM and GRU, PV-only 96-to-1, common seed 42.

Both models use the official external/ol-ems-hkust/main.py entry point. Each
best.pt is selected on validation data and independently evaluated afterwards.
Historical single-layer, three-seed suites remain in their original directories.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

if __package__:
    from .run_pvonly_models import ROOT, run_suite
else:
    from run_pvonly_models import ROOT, run_suite


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=None,
        help="New experiment directory; defaults to a timestamped M0 suite under experiments/.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    destination = args.output or (
        ROOT / "experiments" / f"m0_pvonly_lstm_gru_96to1_{datetime.now():%Y%m%d_%H%M%S_%f}"
    )
    run_suite(destination)


if __name__ == "__main__":
    main()
