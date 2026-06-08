from __future__ import annotations

from experiments.run_gp2f_baseline import PRESETS, _format_mean_std, _resolve_run_seeds
from experiments.run_gp2f_baseline import InputAligner
from data.datasets import resolve_dataset_name
import torch


def test_format_mean_std_single_run() -> None:
    assert _format_mean_std([0.1234]) == "12.34+-0.00"


def test_format_mean_std_multi_run() -> None:
    assert _format_mean_std([0.10, 0.20]) == "15.00+-7.07"


def test_resolve_run_seeds_from_runs() -> None:
    seeds = _resolve_run_seeds({"experiment": {"seed": 5, "runs": 3}})
    assert seeds == [5, 6, 7]


def test_resolve_run_seeds_explicit_list() -> None:
    seeds = _resolve_run_seeds({"experiment": {"seed": 5, "runs": 3, "seeds": [1, 9]}})
    assert seeds == [1, 9]


def test_identity_if_same_dim_aligner_preserves_features() -> None:
    aligner = InputAligner(4, 4, style="identity_if_same_dim")
    x = torch.randn(3, 4)
    assert torch.allclose(aligner(x), x)


def test_identity_if_same_dim_aligner_projects_when_dims_differ() -> None:
    aligner = InputAligner(4, 7, style="identity_if_same_dim")
    x = torch.randn(3, 4)
    assert aligner(x).shape == (3, 7)
    assert any(parameter.requires_grad for parameter in aligner.parameters())


def test_official_projector_aligner_shape() -> None:
    aligner = InputAligner(4, 7, style="official_projector")
    x = torch.randn(3, 4)
    assert aligner(x).shape == (3, 7)


def test_resolve_actor_accepts_canonical_name() -> None:
    assert resolve_dataset_name("Actor") == "Actor"


def test_presets_isolate_ctr_and_fus_losses() -> None:
    assert PRESETS["B0"]["loss"]["lambda_ctr"] == 0.0
    assert PRESETS["B0"]["loss"]["lambda_fus"] == 0.0

    assert PRESETS["B1"]["loss"]["use_original_contrastive"] is True
    assert PRESETS["B1"]["loss"]["use_original_topology_fusion"] is False
    assert PRESETS["B1"]["loss"]["lambda_ctr"] > 0.0
    assert PRESETS["B1"]["loss"]["lambda_fus"] == 0.0

    assert PRESETS["B2"]["loss"]["use_original_contrastive"] is False
    assert PRESETS["B2"]["loss"]["use_original_topology_fusion"] is True
    assert PRESETS["B2"]["loss"]["lambda_ctr"] == 0.0
    assert PRESETS["B2"]["loss"]["lambda_fus"] > 0.0

    assert PRESETS["B3"]["loss"]["use_original_contrastive"] is True
    assert PRESETS["B3"]["loss"]["use_original_topology_fusion"] is True
    assert PRESETS["B3"]["loss"]["lambda_ctr"] > 0.0
    assert PRESETS["B3"]["loss"]["lambda_fus"] > 0.0
