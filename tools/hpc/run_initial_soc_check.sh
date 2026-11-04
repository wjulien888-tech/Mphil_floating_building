#!/usr/bin/env bash
# Run in an existing environment with the forecasting, plant and Gurobi dependencies.
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
: "${MPC_PYTHON:?Set MPC_PYTHON to the project Python executable with all dependencies}"
export MPC_HARDWARE_PROFILE="$ROOT_DIR/configs/battery_50kw_320kwh_24h.json"
export MPC_MILP_SOLVER=gurobi
MODE="${1:-benchmark}"
case "$MODE" in
  benchmark) set -- --smoke-steps 100 ;;
  full) set -- ;;
  *) echo 'Usage: run_initial_soc_check.sh benchmark|full' >&2; exit 1 ;;
esac
"$MPC_PYTHON" -u tools/run_m0_scenarios.py \
  --config configs/initial_soc_check_7days_320kwh.json \
  --scenarios S0 --invalid-pv hold-last --no-live "$@"
