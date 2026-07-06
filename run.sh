#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
RUNNER="experiments/run_gp2f_prompt_graph.py"
PROMPT_VARIANT="p23_selective_graphite_adapter"
RUN_LOG_DIR="${RUN_LOG_DIR:-outputs/run_logs}"

DATASETS=(
  cora
  citeseer
  chameleon
  squirrel
  actor
  minesweeper
)

echo "Running Cora -> target experiments with ${PROMPT_VARIANT}"
echo "Python: ${PYTHON_BIN}"
echo

mkdir -p "$RUN_LOG_DIR"

print_dataset_result() {
  local dataset="$1"
  local summary_json="$2"

  if [[ ! -f "$summary_json" ]]; then
    echo "Summary file not found for ${dataset}: ${summary_json}" >&2
    return 1
  fi

  "$PYTHON_BIN" - "$dataset" "$summary_json" <<'PY'
import json
import sys
from pathlib import Path

dataset = sys.argv[1]
summary_path = Path(sys.argv[2])
summary = json.loads(summary_path.read_text(encoding="utf-8"))

def value(key: str) -> str:
    item = summary.get(key, "NA")
    return str(item)

print("------------------------------------------------------------")
print(f"Result | Cora -> {dataset}")
print(f"Best Test Acc:       {value('best_test_acc_mean_std')}")
print(f"Best Test Macro-F1:  {value('best_test_macro_f1_mean_std')}")
print(f"Best Test AUROC:     {value('best_test_auroc_mean_std')}")
print(f"Best Test AUPRC:     {value('best_test_auprc_mean_std')}")
print(f"Final Test Acc:      {value('final_test_acc_mean_std')}")
print(f"Final Test Macro-F1: {value('final_test_macro_f1_mean_std')}")
print(f"Runs:                {value('num_runs')}")
print(f"Summary:             {summary_path}")
print("------------------------------------------------------------")
PY
}

for dataset in "${DATASETS[@]}"; do
  config="configs/gp2f_prompt_p23_selective_graphite_adapter_${dataset}.yaml"

  if [[ ! -f "$config" ]]; then
    echo "Missing config: $config" >&2
    exit 1
  fi

  echo "============================================================"
  echo "Cora -> ${dataset}"
  echo "Config: ${config}"
  echo "============================================================"

  log_file="${RUN_LOG_DIR}/${dataset}_$(date +%Y%m%d_%H%M%S).log"

  PYTHONUNBUFFERED=1 "$PYTHON_BIN" "$RUNNER" \
    --config "$config" \
    --prompt_variant "$PROMPT_VARIANT" | tee "$log_file"

  summary_json="$(grep -E 'Saved summary to .*/summary\.json$' "$log_file" | tail -n 1 | sed 's/^Saved summary to //' || true)"
  if [[ -z "$summary_json" ]]; then
    echo "Could not locate summary.json from log: ${log_file}" >&2
    exit 1
  fi

  print_dataset_result "$dataset" "$summary_json"

  echo
done

echo "All Cora -> target experiments finished."
