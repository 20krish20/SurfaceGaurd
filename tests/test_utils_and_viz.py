from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from surfaceguard.data.transforms import build_image_transform, build_mask_transform
from surfaceguard.eval.visualize import normalize_heatmap, overlay_heatmap, threshold_mask
from surfaceguard.utils.config import load_yaml, resolve_path
from surfaceguard.utils.io import load_npz, save_npz
from surfaceguard.utils.provenance import stable_cfg_hash

REPO_ROOT = Path(__file__).resolve().parents[1]


# --- provenance / config / io -------------------------------------------------

def test_cfg_hash_is_order_independent_and_value_sensitive() -> None:
    a = {"data": {"root": "x", "image_size": 256}, "seed": 1}
    b = {"seed": 1, "data": {"image_size": 256, "root": "x"}}
    assert stable_cfg_hash(a) == stable_cfg_hash(b)
    assert len(stable_cfg_hash(a)) == 12
    assert stable_cfg_hash(a) != stable_cfg_hash({**a, "seed": 2})


def test_npz_round_trip(tmp_path: Path) -> None:
    mem = np.random.default_rng(0).random((5, 4)).astype(np.float32)
    path = tmp_path / "nested" / "memory.npz"
    save_npz(path, memory=mem)
    np.testing.assert_array_equal(load_npz(path)["memory"], mem)


def test_resolve_path(tmp_path: Path) -> None:
    assert resolve_path(tmp_path, "a/b") == (tmp_path / "a" / "b").resolve()
    assert resolve_path(tmp_path, "/abs/p") == Path("/abs/p")


@pytest.mark.parametrize("name", ["patchcore_mvtec.yaml", "winclip_mvtec.yaml", "app.yaml"])
def test_shipped_configs_parse(name: str) -> None:
    cfg = load_yaml(REPO_ROOT / "configs" / name)
    assert isinstance(cfg, dict) and cfg


def test_patchcore_config_has_required_keys() -> None:
    cfg = load_yaml(REPO_ROOT / "configs" / "patchcore_mvtec.yaml")
    assert {"root", "image_size"} <= set(cfg["data"])
    assert {"backbone", "layer", "coreset_fraction"} <= set(cfg["patchcore"])


# --- transforms -----------------------------------------------------------------

def test_image_transform_shapes_and_ranges() -> None:
    from PIL import Image

    img = Image.fromarray(np.full((40, 30, 3), 255, dtype=np.uint8))
    raw = build_image_transform(32, normalize="none")(img)
    assert raw.shape == (3, 32, 32) and torch.allclose(raw, torch.ones_like(raw))
    assert build_image_transform(32, normalize="clip")(img).shape == (3, 32, 32)
    with pytest.raises(ValueError):
        build_image_transform(32, normalize="bogus")


def test_mask_transform_stays_binary() -> None:
    from PIL import Image

    m = np.zeros((50, 50), dtype=np.uint8)
    m[10:30, 10:30] = 255
    out = build_mask_transform(32)(Image.fromarray(m))
    assert out.shape == (1, 32, 32)
    assert set(torch.unique(out).tolist()) == {0.0, 1.0}


# --- visualisation -------------------------------------------------------------

def test_normalize_heatmap() -> None:
    out = normalize_heatmap(np.array([[2.0, 4.0], [6.0, 10.0]]))
    assert out.min() == 0.0 and out.max() == 1.0
    assert np.all(normalize_heatmap(np.full((3, 3), 7.0)) == 0.0)


def test_overlay_matches_input_image() -> None:
    img = np.zeros((20, 30, 3), dtype=np.uint8)
    out = overlay_heatmap(img, np.random.rand(20, 30))
    assert out.shape == img.shape and out.dtype == np.uint8


@pytest.mark.parametrize("mode", ["otsu", "p95", "p99", "manual"])
def test_threshold_mask_marks_the_hot_spot(mode: str) -> None:
    hm = np.zeros((20, 20), dtype=np.float32)
    hm[4:9, 4:9] = 1.0  # 25 of 400 px (> 5%, so the p95 cut-off lands on the hot spot)
    mask = threshold_mask(hm, mode=mode)
    assert mask.dtype == np.uint8 and set(np.unique(mask).tolist()) <= {0, 255}
    assert mask[6, 6] == 255 and mask[15, 15] == 0


def test_threshold_mask_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError):
        threshold_mask(np.random.rand(4, 4), mode="nope")
