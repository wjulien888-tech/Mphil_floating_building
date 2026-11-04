#!/bin/bash
# Run from project root. Never calls sbatch or changes M1 outputs.
set -euo pipefail
TASK_ROOT_M2="$(pwd)"
[[ -f tools/train_m2.py && -f configs/battery_50kw_320kwh_24h.json ]] || { echo 'Run from project root'; exit 2; }
export PYTHONPATH="$TASK_ROOT_M2/external/python-microgrid/src${PYTHONPATH:+:$PYTHONPATH}"
export MPC_HARDWARE_PROFILE="$TASK_ROOT_M2/configs/battery_50kw_320kwh_24h.json"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="$TASK_ROOT_M2/tmp/matplotlib"
mkdir -p "$MPLCONFIGDIR" logs
PY_M2="${M2_PYTHON:-/data/user/jwei484/envs/m1-original-py311/bin/python}"
CKPT_M2="${M2_CHECKPOINT:-$TASK_ROOT_M2/checkpoints/m0_gru_L96_H96_seed42_best.pt}"
[[ -f "$CKPT_M2" ]] || { echo "Missing M0 checkpoint: $CKPT_M2"; exit 2; }
TAG_M2="$(date +%Y%m%d_%H%M%S)_$$"
for W_M2 in 0 1; do
  for MODE_M2 in preflight smoke; do
    "$PY_M2" -u tools/train_m2.py --mode "$MODE_M2" \
      --checkpoint "$CKPT_M2" --scenario-config configs/m1_s1_s4_soc50_v1_20261025.json \
      --run-name "ABcheck_p${W_M2}_${MODE_M2}_${TAG_M2}" \
      --prediction-weight "$W_M2" --economic-weight 5 --ens-weight 5 \
      --voll-cl1 50 --voll-cl2 10 --voll-cl3 2 \
      --selection-metric operational_CNY_per_step --min-delta 0.0001 \
      --max-epochs 7 --patience 3 --learning-rate 1e-4 --gradient-clip 1 \
      --checkpoint-steps 8 --threads 1 --seed 42
  done
done
echo 'Preflight and short smoke finished; no formal job submitted.'
