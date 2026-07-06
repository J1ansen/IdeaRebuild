#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
RUNNER="experiments/run_gp2f_prompt_graph.py"
PROMPT_VARIANT="p23_selective_graphite_adapter"
CONFIG="configs/gp2f_prompt_p23_selective_graphite_adapter_chameleon.yaml"
OUT_ROOT="${OUT_ROOT:-outputs/p23_seed12345_chameleon_focused_search}"
TRIALS="${TRIALS:-3}"
EPOCHS="${EPOCHS:-300}"

CONFIGS=(
  current:none:none:none:none:on:0.25
  g25_lr5_p5_d0_ctr0:0.0005:0.0005:0.0:0.0:on:0.25
  g25_lr5_p5_d0_ctr1:0.0005:0.0005:0.0:0.01:on:0.25
  g25_lr5_p5_d0_ctr3:0.0005:0.0005:0.0:0.03:on:0.25
  g25_lr5_p5_d0_ctr5:0.0005:0.0005:0.0:0.05:on:0.25
  g25_lr5_p25_d0_ctr1:0.0005:0.00025:0.0:0.01:on:0.25
  g25_lr5_p25_d0_ctr3:0.0005:0.00025:0.0:0.03:on:0.25
  g25_lr5_p75_d0_ctr1:0.0005:0.00075:0.0:0.01:on:0.25
  g25_lr5_p75_d0_ctr3:0.0005:0.00075:0.0:0.03:on:0.25
  g25_lr5_p5_d05_ctr1:0.0005:0.0005:0.05:0.01:on:0.25
  g25_lr5_p5_d05_ctr3:0.0005:0.0005:0.05:0.03:on:0.25
  g25_lr5_p5_d1_ctr3:0.0005:0.0005:0.1:0.03:on:0.25
  g25_lr75_p5_d0_ctr1:0.00075:0.0005:0.0:0.01:on:0.25
  g25_lr75_p5_d0_ctr3:0.00075:0.0005:0.0:0.03:on:0.25
  g25_lr75_p75_d0_ctr1:0.00075:0.00075:0.0:0.01:on:0.25
  g25_lr75_p75_d0_ctr3:0.00075:0.00075:0.0:0.03:on:0.25
  g25_lr10_p5_d0_ctr0:0.001:0.0005:0.0:0.0:on:0.25
  g25_lr10_p10_d0_ctr3:0.001:0.001:0.0:0.03:on:0.25
  g10_lr5_p5_d0_ctr1:0.0005:0.0005:0.0:0.01:on:0.10
  g10_lr5_p5_d0_ctr3:0.0005:0.0005:0.0:0.03:on:0.10
  g10_lr5_p5_d0_ctr5:0.0005:0.0005:0.0:0.05:on:0.10
  g10_lr5_p75_d0_ctr3:0.0005:0.00075:0.0:0.03:on:0.10
  g10_lr5_p5_d05_ctr3:0.0005:0.0005:0.05:0.03:on:0.10
  g35_lr5_p5_d0_ctr1:0.0005:0.0005:0.0:0.01:on:0.35
  g35_lr5_p5_d0_ctr3:0.0005:0.0005:0.0:0.03:on:0.35
  g35_lr5_p75_d0_ctr3:0.0005:0.00075:0.0:0.03:on:0.35
  goff_lr5_p5_d0_ctr0:0.0005:0.0005:0.0:0.0:off:0.0
  goff_lr5_p5_d0_ctr3:0.0005:0.0005:0.0:0.03:off:0.0
)

mkdir -p "$OUT_ROOT/logs"

for item in "${CONFIGS[@]}"; do
  IFS=: read -r tag lr prompt_lr dropout lambda_ctr gate gate_strength <<<"$item"
  output_dir="${OUT_ROOT}/chameleon/${tag}"
  log_file="${OUT_ROOT}/logs/chameleon_${tag}.log"

  echo "============================================================"
  echo "Chameleon focused search | ${tag} | trials=${TRIALS}"
  echo "============================================================"

  cmd=(
    "$PYTHON_BIN" "$RUNNER"
    --config "$CONFIG"
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
    if [[ "$gate" == "off" ]]; then
      cmd+=(--disable_p23_node_feature_edge_gate)
    else
      cmd+=(--enable_p23_node_feature_edge_gate --p23_node_feature_edge_gate_strength "$gate_strength")
    fi
  fi

  PYTHONUNBUFFERED=1 "${cmd[@]}" | tee "$log_file"
  "$PYTHON_BIN" scripts/summarize_p23_seed12345_search.py "$OUT_ROOT"
done

"$PYTHON_BIN" scripts/summarize_p23_seed12345_search.py "$OUT_ROOT"
