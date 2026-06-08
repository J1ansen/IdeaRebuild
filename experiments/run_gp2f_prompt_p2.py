"""Run GP2F P2 prompt-aware adapted-branch experiments."""

from __future__ import annotations

import sys

from experiments.run_gp2f_prompt_graph import main as _graph_main


def main() -> None:
    if "--config" not in sys.argv:
        sys.argv[1:1] = ["--config", "configs/gp2f_prompt_p2.yaml"]
    _graph_main()


if __name__ == "__main__":
    main()
