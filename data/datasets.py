"""Node dataset loading with explicit cache checks.

The baseline reuses existing PyG cache directories and must not silently
download datasets during experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.datasets import Actor, HeterophilousGraphDataset, Planetoid, WikipediaNetwork


DATASET_ALIASES: dict[str, str] = {
    "actor": "Actor",
    "Actor": "Actor",
    "chameleon": "chameleon",
    "Chameleon": "chameleon",
    "squirrel": "squirrel",
    "Squirrel": "squirrel",
    "cora": "Cora",
    "Cora": "Cora",
    "citeseer": "CiteSeer",
    "CiteSeer": "CiteSeer",
    "pubmed": "PubMed",
    "PubMed": "PubMed",
    "minesweeper": "minesweeper",
    "Minesweeper": "minesweeper",
}


@dataclass(frozen=True)
class LoadedNodeDataset:
    name: str
    data: Data
    num_features: int
    num_classes: int


def resolve_dataset_name(name: str) -> str:
    key = name.strip()
    if key in DATASET_ALIASES:
        return DATASET_ALIASES[key]
    raise ValueError(
        f"Unsupported dataset {name!r}. Supported names: "
        "Actor, chameleon, squirrel, minesweeper, Cora, CiteSeer, PubMed."
    )


def _require_processed(path: Path, *, download_if_missing: bool) -> None:
    if path.is_dir() and any(path.iterdir()):
        return
    if download_if_missing:
        return
    raise FileNotFoundError(
        f"Processed dataset cache not found: {path}. "
        "Set download_if_missing=true only when intentional."
    )


def load_node_dataset(
    name: str,
    root: str | Path,
    *,
    download_if_missing: bool = False,
) -> LoadedNodeDataset:
    """Load a supported node dataset from an existing PyG cache."""

    canonical = resolve_dataset_name(name)
    root_path = Path(root).expanduser()
    transform = T.NormalizeFeatures()

    if canonical in {"Cora", "CiteSeer", "PubMed"}:
        _require_processed(root_path / "Planetoid" / canonical / "processed", download_if_missing=download_if_missing)
        dataset = Planetoid(root=str(root_path / "Planetoid"), name=canonical, transform=transform)
    elif canonical == "Actor":
        _require_processed(root_path / "Actor" / "processed", download_if_missing=download_if_missing)
        dataset = Actor(root=str(root_path / "Actor"), transform=transform)
    elif canonical in {"chameleon", "squirrel"}:
        processed = root_path / "WikipediaNetwork" / canonical / "geom_gcn" / "processed"
        _require_processed(processed, download_if_missing=download_if_missing)
        dataset = WikipediaNetwork(
            root=str(root_path / "WikipediaNetwork"),
            name=canonical,
            transform=transform,
        )
    elif canonical == "minesweeper":
        processed = root_path / "HeterophilousGraphDataset" / canonical / "processed"
        _require_processed(processed, download_if_missing=download_if_missing)
        dataset = HeterophilousGraphDataset(
            root=str(root_path / "HeterophilousGraphDataset"),
            name=canonical,
            transform=transform,
        )
    else:
        raise AssertionError(f"Unhandled canonical dataset: {canonical}")

    data = dataset[0]
    return LoadedNodeDataset(
        name=canonical,
        data=data,
        num_features=int(dataset.num_features),
        num_classes=int(dataset.num_classes),
    )
