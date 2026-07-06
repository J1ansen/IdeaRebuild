#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
RUNNER="experiments/run_gp2f_prompt_graph.py"
PROMPT_VARIANT="p23_selective_graphite_adapter"
OUT_ROOT="${OUT_ROOT:-outputs/p23_seed12345_hetero_search}"
TRIALS="${TRIALS:-3}"
EPOCHS="${EPOCHS:-300}"

DATASETS=(
  chameleon
  squirrel
  actor
  minesweeper
)

CONFIGS=(
  current:none:none:none:none
  lr5_p5_d0_ctr0:0.0005:0.0005:0.0:0.0
  lr5_p5_d1_ctr0:0.0005:0.0005:0.1:0.0
  lr5_p5_d0_ctr3:0.0005:0.0005:0.0:0.03
  lr5_p5_d1_ctr3:0.0005:0.0005:0.1:0.03
  lr5_p25_d1_ctr3:0.0005:0.00025:0.1:0.03
  lr10_p10_d0_ctr0:0.001:0.001:0.0:0.0
  lr10_p10_d1_ctr3:0.001:0.001:0.1:0.03
)

mkdir -p "$OUT_ROOT/logs"

for dataset in "${DATASETS[@]}"; do
  base_config="configs/gp2f_prompt_p23_selective_graphite_adapter_${dataset}.yaml"
  if [[ ! -f "$base_config" ]]; then
    echo "Missing config: $base_config" >&2
    exit 1
  fi

  for item in "${CONFIGS[@]}"; do
    IFS=: read -r tag lr prompt_lr dropout lambda_ctr <<<"$item"
    output_dir="${OUT_ROOT}/${dataset}/${tag}"
    log_file="${OUT_ROOT}/logs/${dataset}_${tag}.log"

    echo "============================================================"
    echo "Search | ${dataset} | ${tag} | trials=${TRIALS}"
    echo "============================================================"

    cmd=(
      "$PYTHON_BIN" "$RUNNER"
      --config "$base_config"
      --prompt_variant "$PROMPT_VARIANT"
      --seed 12345
      --runs 1
      --trials_per_seed "$TRIALS"
      --epochs "$EPOCHS"
      --output_dir "$output_dir"
      --disable_original_topology_fusion
      --lambda_fus 0.0
    )

    if [[ "$tag" != "current" ]]; then
      cmd+=(--lr "$lr" --prompt_lr "$prompt_lr" --dropout "$dropout")
      if [[ "$lambda_ctr" == "0.0" ]]; then
        cmd+=(--disable_original_contrastive --lambda_ctr 0.0)
      else
        cmd+=(--use_original_contrastive --lambda_ctr "$lambda_ctr" --tau_ctr 0.5)
      fi
    fi

    PYTHONUNBUFFERED=1 "${cmd[@]}" | tee "$log_file"
  done
done

"$PYTHON_BIN" scripts/summarize_p23_seed12345_search.py "$OUT_ROOT"
