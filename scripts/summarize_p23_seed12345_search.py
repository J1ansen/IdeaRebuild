#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def parse_mean_std(value: object) -> float:
    text = str(value or "0")
    try:
        return float(text.split("±", 1)[0].strip())
    except ValueError:
        return 0.0


def latest_summary(run_dir: Path) -> Path | None:
    summaries = sorted(run_dir.glob("*/20*/summary.json"))
    return summaries[-1] if summaries else None


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/p23_seed12345_hetero_search")
    rows: list[dict[str, object]] = []
    for summary_path in sorted(root.glob("*/*/*/20*/summary.json")):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        parts = summary_path.relative_to(root).parts
        if len(parts) < 5:
            continue
        dataset, tag = parts[0], parts[1]
        rows.append(
            {
                "dataset": dataset,
                "tag": tag,
                "best_acc": summary.get("best_test_acc_mean_std", ""),
                "best_f1": summary.get("best_test_macro_f1_mean_std", ""),
                "best_auroc": summary.get("best_test_auroc_mean_std", ""),
                "best_auprc": summary.get("best_test_auprc_mean_std", ""),
                "final_acc": summary.get("final_test_acc_mean_std", ""),
                "final_f1": summary.get("final_test_macro_f1_mean_std", ""),
                "runs": summary.get("num_runs", ""),
                "path": str(summary_path),
                "score": parse_mean_std(summary.get("best_test_macro_f1_mean_std", "")),
            }
        )

    by_dataset: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_dataset[str(row["dataset"])].append(row)

    for dataset in sorted(by_dataset):
        print(f"\n=== {dataset} ===")
        ranked = sorted(by_dataset[dataset], key=lambda item: (float(item["score"]), parse_mean_std(item["best_acc"])), reverse=True)
        for row in ranked:
            print(
                f"{row['tag']:18s} | runs={row['runs']} | "
                f"Best Acc {row['best_acc']} | Best F1 {row['best_f1']} | "
                f"AUROC {row['best_auroc']} | AUPRC {row['best_auprc']} | "
                f"Final Acc {row['final_acc']} | Final F1 {row['final_f1']}"
            )
            print(f"  {row['path']}")


if __name__ == "__main__":
    main()
