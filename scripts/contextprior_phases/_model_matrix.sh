#!/usr/bin/env bash


model_config_list() {
  case "$1" in
    ours_650m|contextprior_650m|ours)
      echo "configs/methods/ours.yaml configs/backbones/esm2_650m.yaml configs/scales/xlarge_650m.yaml configs/contextprior/default_graph.yaml configs/train_modes/frozen.yaml"
      ;;
    *)
      echo "Unknown model key: $1" >&2
      return 2
      ;;
  esac
}

resolve_model_runs() {
  if [[ -n "${MODEL_RUNS:-}" ]]; then
    echo "$MODEL_RUNS"
  else
    echo "ours_650m"
  fi
}
