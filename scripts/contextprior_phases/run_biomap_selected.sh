#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."



TASK_CONFIG="${TASK_CONFIG:-configs/experiments/biomap_selected_stable_variant.yaml}"
source scripts/contextprior_phases/_model_matrix.sh
source scripts/contextprior_phases/_runtime_overrides.sh
append_runtime_overrides
MODEL_KEYS=($(resolve_model_runs))

for model_key in "${MODEL_KEYS[@]}"; do
  CONFIG_LIST=($(model_config_list "$model_key"))

  CMD=(
    python scripts/run_finetune.py
    --config configs/base.yaml
    --config "$TASK_CONFIG"
  )
  for cfg in "${CONFIG_LIST[@]}"; do
    CMD+=(--config "$cfg")
  done
  if [[ -n "${EXTRA_CONFIGS:-}" ]]; then
    for cfg in ${EXTRA_CONFIGS}; do
      CMD+=(--config "$cfg")
    done
  fi
  echo "[contextprior:biomap] running ${model_key}: ${CMD[*]}"
  "${CMD[@]}"
done

python scripts/analyze_biomap_selected.py \
  --runs outputs/runs_biomap_selected \
  --out outputs/stable_variant_prioritization/biomap_selected \
  --allow-empty
