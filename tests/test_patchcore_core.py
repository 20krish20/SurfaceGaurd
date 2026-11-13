from __future__ import annotations

import numpy as np
import pytest
import torch

from surfaceguard.models.patchcore.core import (
    PatchCoreEngine,
    PatchCoreState,
    extract_feature_map,
    feature_map_to_patches,
    kcenter_greedy_coreset,
    l2_normalize,
    random_coreset,
)


def test_feature_map_to_patches_keeps_spatial_order() -> None:
    fmap = torch.arange(2 * 3 * 2 * 2, dtype=torch.float32).reshape(2, 3, 2, 2)  # [B,C,H,W]
    patches = feature_map_to_patches(fmap)
    assert patches.shape == (2, 4, 3)
    # Patch (row=1, col=0) of image 1 is the channel vector fmap[1, :, 1, 0].
    assert torch.equal(patches[1, 2], fmap[1, :, 1, 0])


def test_l2_normalize_gives_unit_vectors() -> None:
    x = torch.randn(5, 7)
    norms = l2_normalize(x).norm(dim=-1)
    assert torch.allclose(norms, torch.ones(5), atol=1e-6)
    assert torch.all(torch.isfinite(l2_normalize(torch.zeros(2, 3))))


@pytest.mark.parametrize("fn", [random_coreset, kcenter_greedy_coreset])
def test_coreset_size_uniqueness_and_determinism(fn) -> None:
    x = np.random.default_rng(0).normal(size=(200, 8)).astype(np.float32)
    idx = fn(x, fraction=0.1, seed=7)
    assert len(idx) == 20
    assert len(np.unique(idx)) == 20
    assert idx.min() >= 0 and idx.max() < 200
    np.testing.assert_array_equal(idx, fn(x, fraction=0.1, seed=7))


@pytest.mark.parametrize("fn", [random_coreset, kcenter_greedy_coreset])
def test_coreset_keeps_everything_when_fraction_is_one(fn) -> None:
    x = np.zeros((10, 2), dtype=np.float32)
    np.testing.assert_array_equal(np.sort(fn(x, fraction=1.0)), np.arange(10))


def test_kcenter_covers_isolated_outliers() -> None:
    # 100 points in a tight cluster plus 3 far-away points: k-center must pick all outliers.
    rng = np.random.default_rng(0)
    cluster = rng.normal(scale=0.01, size=(100, 2)).astype(np.float32)
    outliers = np.array([[10, 10], [-10, 10], [10, -10]], dtype=np.float32)
    x = np.vstack([cluster, outliers])
    idx = kcenter_greedy_coreset(x, fraction=0.04, seed=0)  # 4 centres
    assert {100, 101, 102} <= set(idx.tolist())


def test_extract_feature_map_rejects_unknown_layer() -> None:
    with pytest.raises(ValueError, match="no layer"):
        extract_feature_map(torch.nn.Identity(), torch.zeros(1, 3, 8, 8), "layer9")


def _engine(memory: np.ndarray, image_score: str = "max", topk: int = 3) -> PatchCoreEngine:
    state = PatchCoreState(
        backbone="resnet18",
        layer="layer2",
        image_size=64,
        nn_k=1,
        image_score=image_score,
        topk=topk,
        threshold=0.0,
        memory=memory,
    )
    return PatchCoreEngine(state, device=torch.device("cpu"))


@pytest.mark.usefixtures("tiny_backbone")
def test_engine_scores_memory_images_as_normal() -> None:
    # Build the memory bank from the very image we score: every patch has an exact match.
    torch.manual_seed(1)
    img = torch.randn(1, 3, 64, 64)
    probe = _engine(np.zeros((1, 128), dtype=np.float32))
    fmap = extract_feature_map(probe.backbone, img, "layer2")
    memory = l2_normalize(feature_map_to_patches(fmap))[0].numpy()

    engine = _engine(memory)
    scores, heat = engine.batch_score(torch.cat([img, torch.randn(1, 3, 64, 64)]))
    assert heat.shape == (2, 64, 64) and heat.dtype == np.float32
    assert scores[0] == pytest.approx(0.0, abs=1e-3)
    assert scores[1] > scores[0]


@pytest.mark.usefixtures("tiny_backbone")
def test_topk_mean_score_is_at_most_max_score() -> None:
    memory = l2_normalize(torch.randn(50, 128)).numpy()
    x = torch.randn(2, 3, 64, 64)
    max_scores, _ = _engine(memory, "max").batch_score(x)
    topk_scores, _ = _engine(memory, "topk_mean", topk=5).batch_score(x)
    assert np.all(topk_scores <= max_scores + 1e-6)
