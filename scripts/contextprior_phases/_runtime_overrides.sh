#!/usr/bin/env bash

append_runtime_overrides() {
  if [[ -n "${EPOCHS:-}" ]]; then
    mkdir -p outputs/runtime_configs
    local epoch_cfg="outputs/runtime_configs/epochs_${EPOCHS}.yaml"
    cat > "$epoch_cfg" <<EOF
train:
  epochs: ${EPOCHS}
EOF
    EXTRA_CONFIGS="${EXTRA_CONFIGS:-} ${epoch_cfg}"
    export EXTRA_CONFIGS
  fi
  if [[ -n "${BATCH_SIZE:-}" || -n "${NUM_WORKERS:-}" ]]; then
    mkdir -p outputs/runtime_configs
    local runtime_cfg="outputs/runtime_configs/data_runtime_bs${BATCH_SIZE:-default}_nw${NUM_WORKERS:-default}.yaml"
    {
      echo "data:"
      if [[ -n "${BATCH_SIZE:-}" ]]; then
        echo "  batch_size: ${BATCH_SIZE}"
      fi
      if [[ -n "${NUM_WORKERS:-}" ]]; then
        echo "  num_workers: ${NUM_WORKERS}"
      fi
    } > "$runtime_cfg"
    EXTRA_CONFIGS="${EXTRA_CONFIGS:-} ${runtime_cfg}"
    export EXTRA_CONFIGS
  fi
}
