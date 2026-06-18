#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."


if [[ ! -f data/stability_prepared/saprothub_meta_stability/official.csv ]]; then
  python scripts/prepare_saprothub_splits.py --hf-cache data/stability/saprothub_meta_stability --out-dir data/stability_prepared
fi

source scripts/contextprior_phases/_model_matrix.sh
source scripts/contextprior_phases/_runtime_overrides.sh
append_runtime_overrides

DATA_CONFIG="${DATA_CONFIG:-configs/data/saprothub_meta_stability_official_csv.yaml}"
MODEL_KEYS=($(resolve_model_runs))

for model_key in "${MODEL_KEYS[@]}"; do
  CONFIG_LIST=($(model_config_list "$model_key"))

  CMD=(
    python scripts/run_stability_finetune.py
    --config configs/base.yaml
    --config configs/stability_experiments/stability_main.yaml
    --config "$DATA_CONFIG"
    --config configs/contextprior/stability_dense_f10_s5_t15.yaml
  )
  for cfg in "${CONFIG_LIST[@]}"; do
    CMD+=(--config "$cfg")
  done
  if [[ -n "${EXTRA_CONFIGS:-}" ]]; then
    for cfg in ${EXTRA_CONFIGS}; do
      CMD+=(--config "$cfg")
    done
  fi
  echo "[contextprior:stability] running ${model_key}: ${CMD[*]}"
  "${CMD[@]}"
done
