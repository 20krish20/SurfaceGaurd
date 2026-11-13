from __future__ import annotations

import math

import numpy as np
import pytest

from surfaceguard.eval.metrics import (
    _pro_for_threshold,
    au_pro,
    best_f1_at_threshold,
    image_level_auroc,
    pixel_level_auroc,
)


def test_image_auroc_perfect_and_inverted() -> None:
    y = np.array([0, 0, 1, 1])
    assert image_level_auroc(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert image_level_auroc(y, np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(0.0)


def test_image_auroc_is_nan_with_one_class() -> None:
    assert math.isnan(image_level_auroc(np.array([1, 1, 1]), np.array([0.1, 0.5, 0.9])))


def test_pixel_auroc() -> None:
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:4, 2:4] = 1
    assert pixel_level_auroc([mask], [mask.astype(np.float32)]) == pytest.approx(1.0)
    assert math.isnan(pixel_level_auroc([np.zeros((4, 4))], [np.random.rand(4, 4)]))


def test_best_f1_finds_separating_threshold() -> None:
    y = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.9])
    best = best_f1_at_threshold(y, scores)
    assert best["f1"] == pytest.approx(1.0)
    assert 0.3 < best["threshold"] <= 0.7


def test_pro_averages_over_connected_regions() -> None:
    gt = np.zeros((10, 10), dtype=np.uint8)
    gt[0:2, 0:2] = 1  # region A (4 px)
    gt[6:8, 6:8] = 1  # region B (4 px)
    pred = np.zeros_like(gt)
    pred[0:2, 0:2] = 1  # A fully covered, B missed
    assert _pro_for_threshold(gt, pred) == pytest.approx(0.5)
    assert _pro_for_threshold(np.zeros_like(gt), pred) == 0.0


def _blobs(rng: np.random.Generator, n: int = 3, size: int = 32):
    masks = []
    for _ in range(n):
        m = np.zeros((size, size), dtype=np.uint8)
        r, c = rng.integers(2, size - 8, size=2)
        m[r : r + 6, c : c + 6] = 1
        masks.append(m)
    return masks


def test_au_pro_rewards_good_localisation() -> None:
    rng = np.random.default_rng(0)
    masks = _blobs(rng)
    # Defect pixels are hot, background is mild noise (a realistic "good" heatmap).
    good_maps = [(0.6 * m + 0.4 * rng.random(m.shape)).astype(np.float32) for m in masks]
    noise_maps = [rng.random(m.shape).astype(np.float32) for m in masks]

    good = au_pro(masks, good_maps)
    random = au_pro(masks, noise_maps)
    assert 0.9 <= good <= 1.0
    assert 0.0 <= random < 0.5
    assert good > random


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known issue: au_pro() integrates only up to the largest *sampled* FPR <= max_fpr instead of "
        "interpolating PRO at max_fpr, so a perfect binary heatmap scores 0.0 (and spiky heatmaps are "
        "under-estimated). Remove this marker once the curve is interpolated at max_fpr."
    ),
)
def test_au_pro_perfect_binary_heatmap_scores_one() -> None:
    masks = _blobs(np.random.default_rng(0))
    assert au_pro(masks, [m.astype(np.float32) for m in masks]) == pytest.approx(1.0, abs=1e-3)


def test_au_pro_constant_heatmap_scores_zero() -> None:
    m = np.zeros((8, 8), dtype=np.uint8)
    m[0:2, 0:2] = 1
    assert au_pro([m], [np.ones((8, 8), dtype=np.float32)]) == 0.0
