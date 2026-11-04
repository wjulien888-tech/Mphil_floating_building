#!/usr/bin/env bash
# Run on the configured training host, not a scheduler submission script.
# Requires an explicit shared M0 forecasting checkpoint and Python environment.
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
: "${M1_M0_CHECKPOINT:?Set M1_M0_CHECKPOINT to the same existing M0 checkpoint for all three runs}"
: "${M1_PYTHON:?Set M1_PYTHON to the Python executable with training dependencies}"
[[ -f "$M1_M0_CHECKPOINT" ]] || { echo 'Starting checkpoint does not exist' >&2; exit 1; }
export MPC_HARDWARE_PROFILE="$ROOT_DIR/configs/battery_50kw_320kwh_24h.json"
export MPC_MILP_SOLVER=gurobi
# Training remains continuous; the MILP solver setting applies to final scenario evaluation.
for ECON_WEIGHT in 5 7.5 10; do
    RUN_TAG="econ1to${ECON_WEIGHT//./p}_b320_seed42"
    "$M1_PYTHON" -u tools/train_m1.py --mode train \
        --checkpoint "$M1_M0_CHECKPOINT" --run-name "$RUN_TAG" \
        --scenario-config configs/m1_s1_s4_soc50_v1_20261025.json \
        --prediction-weight 1 --economic-weight "$ECON_WEIGHT" \
        --selection-metric economic_CNY_per_step \
        --max-epochs 7 --patience 3 --seed 42 --threads 1
done
