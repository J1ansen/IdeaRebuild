from experiments.run_gp2f_prompt_graph import _resolve_shot_setting


def test_auto_shot_by_dataset_keeps_homophily_at_five_shot():
    data_cfg = {
        "shots": 5,
        "homophily_shots": 5,
        "auto_shot_by_dataset": True,
        "heterophily_shot_ratio": 0.10,
        "heterophily_datasets": ["Actor", "Squirrel", "Chameleon"],
    }
    shots, ratio, mode = _resolve_shot_setting(data_cfg, {}, "Cora")
    assert shots == 5
    assert ratio is None
    assert mode == "homophily_5shot"


def test_auto_shot_by_dataset_uses_ten_percent_for_heterophily():
    data_cfg = {
        "shots": 5,
        "homophily_shots": 5,
        "auto_shot_by_dataset": True,
        "heterophily_shot_ratio": 0.10,
        "heterophily_datasets": ["Actor", "Squirrel", "Chameleon"],
    }
    shots, ratio, mode = _resolve_shot_setting(data_cfg, {}, "Squirrel")
    assert shots == 5
    assert ratio == 0.10
    assert mode == "heterophily_10pct"


def test_explicit_shot_ratio_is_preserved_when_auto_disabled():
    shots, ratio, mode = _resolve_shot_setting(
        {"shots": 5, "shot_ratio": 0.05, "auto_shot_by_dataset": False},
        {},
        "Actor",
    )
    assert shots == 5
    assert ratio == 0.05
    assert mode == "explicit_ratio"
