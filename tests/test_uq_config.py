"""Which tissues a ``ConductivityUQConfig`` perturbs, and what it refuses at construction."""

from __future__ import annotations

import pytest

from cunibs.uq.conductivity.config import DEFAULT_TISSUE_COV, ConductivityUQConfig

MESH_TAGS = (1, 2, 3, 5, 7, 8)


def test_tissue_cov_alone_selects_the_perturbed_tissues():
    cfg = ConductivityUQConfig(tissue_cov={2: 0.15, 3: 0.05, 7: 0.35, 8: 0.35})
    assert cfg.tags_for(MESH_TAGS) == (2, 3, 7, 8)


def test_no_selection_perturbs_every_tissue_in_the_mesh():
    assert ConductivityUQConfig().tags_for((5, 1, 3)) == (1, 3, 5)


def test_explicit_perturbed_tags_take_default_covs_for_unnamed_tissues():
    cfg = ConductivityUQConfig(tissue_cov={2: 0.4}, perturbed_tags=(3, 2))
    assert cfg.tags_for(MESH_TAGS) == (2, 3)
    assert cfg.cov_for(2) == 0.4
    assert cfg.cov_for(3) == DEFAULT_TISSUE_COV[3]


def test_labels_and_tags_name_the_same_tissue():
    by_label = ConductivityUQConfig(tissue_cov={"gray_matter": 0.2, "csf": 0.1})
    by_tag = ConductivityUQConfig(tissue_cov={2: 0.2, 3: 0.1})
    assert by_label == by_tag
    assert ConductivityUQConfig(perturbed_tags=("cortical_bone",)).perturbed_tags == (7,)


def test_a_cov_for_a_tissue_held_fixed_is_rejected():
    with pytest.raises(ValueError, match="holds fixed"):
        ConductivityUQConfig(tissue_cov={2: 0.2, 3: 0.1}, perturbed_tags=(2,))


def test_a_tissue_missing_from_the_mesh_is_rejected():
    with pytest.raises(ValueError, match="holds none of their elements"):
        ConductivityUQConfig(tissue_cov={9: 0.1}).tags_for(MESH_TAGS)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_samples": 0},
        {"n_samples": 2.5},
        {"distribution": "uniform"},
        {"tissue_cov": {2: -0.1}},
        {"tissue_cov": {2: float("nan")}},
        {"tissue_cov": {99: 0.1}},
        {"tissue_cov": {"grey_matter": 0.1}},
        {"tissue_cov": {2: 0.1, "gray_matter": 0.2}},
        {"perturbed_tags": ()},
        {"perturbed_tags": (2, 2)},
    ],
)
def test_invalid_configs_raise(kwargs):
    with pytest.raises(ValueError):
        ConductivityUQConfig(**kwargs)


def test_a_single_draw_and_a_zero_cov_stay_valid():
    cfg = ConductivityUQConfig(n_samples=1, tissue_cov={2: 0.0})
    assert cfg.n_samples == 1 and cfg.cov_for(2) == 0.0
