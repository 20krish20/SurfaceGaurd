from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from surfaceguard.data.mvtec import (
    MVTec_AD_15_CATEGORIES,
    MVTecADDataset,
    all_categories,
    discover_categories,
    mvtec_category_names,
    resolve_mvtec_categories,
)
from surfaceguard.data.transforms import build_image_transform, build_mask_transform


def test_train_split_contains_only_normal_images(toy_root: Path) -> None:
    ds = MVTecADDataset(toy_root, "toy", split="train")
    assert len(ds) == 8
    assert all(s.label == 0 and s.defect_type == "good" and s.mask_path is None for s in ds.samples)


def test_test_split_labels_and_masks(toy_root: Path) -> None:
    ds = MVTecADDataset(toy_root, "toy", split="test")
    good = [s for s in ds.samples if s.label == 0]
    bad = [s for s in ds.samples if s.label == 1]
    assert len(good) == 6 and len(bad) == 6
    assert {s.defect_type for s in bad} == {"scratch"}
    # Every anomalous image is paired with its "<stem>_mask.png" ground truth.
    for s in bad:
        assert s.mask_path is not None and s.mask_path.name == f"{s.image_path.stem}_mask.png"


def test_mask_with_same_filename_is_also_found(tmp_path: Path) -> None:
    cat = tmp_path / "widget"
    for sub in ("train/good", "test/good", "test/dent", "ground_truth/dent"):
        (cat / sub).mkdir(parents=True)
    img = Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8))
    img.save(cat / "test" / "dent" / "000.png")
    img.save(cat / "ground_truth" / "dent" / "000.png")  # no "_mask" suffix

    ds = MVTecADDataset(tmp_path, "widget", split="test")
    (sample,) = ds.samples
    assert sample.mask_path == cat / "ground_truth" / "dent" / "000.png"


def test_getitem_with_transforms_returns_tensors(toy_root: Path) -> None:
    ds = MVTecADDataset(
        toy_root,
        "toy",
        split="test",
        # Native 256px: nearest-neighbour downsizing could erase the 2px synthetic scratches.
        image_transform=build_image_transform(256),
        mask_transform=build_mask_transform(256),
    )
    items = [ds[i] for i in range(len(ds))]
    for item in items:
        assert item["image"].shape == (3, 256, 256)
        assert item["mask"].shape == (1, 256, 256)
        assert set(torch.unique(item["mask"]).tolist()) <= {0.0, 1.0}
    defective = [it for it in items if it["label"] == 1]
    assert all(it["has_mask"] and it["mask"].sum() > 0 for it in defective)
    normal = [it for it in items if it["label"] == 0]
    assert all(not it["has_mask"] and it["mask"].sum() == 0 for it in normal)


def test_missing_category_lists_available_ones(toy_root: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Available categories.*toy"):
        MVTecADDataset(toy_root, "bottle", split="train")


def test_empty_root_points_to_toy_dataset_helper(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="make_synth_mvtec"):
        MVTecADDataset(tmp_path / "missing", "bottle", split="train")


def test_invalid_split_is_rejected(toy_root: Path) -> None:
    with pytest.raises(ValueError, match="split"):
        MVTecADDataset(toy_root, "toy", split="val")


def test_category_resolution(toy_root: Path, tmp_path: Path) -> None:
    assert mvtec_category_names(toy_root) == ["toy"]
    assert mvtec_category_names(tmp_path / "missing") == []
    # Requested categories are filtered to those present on disk.
    assert resolve_mvtec_categories(toy_root, ["toy", "bottle"]) == ["toy"]
    # No dataset on disk yet: fall back to the full MVTec AD list.
    assert resolve_mvtec_categories(tmp_path / "missing") == list(MVTec_AD_15_CATEGORIES)


def test_discovery_finds_any_mvtec_style_category(tmp_path: Path, make_toy) -> None:
    make_toy(tmp_path, "battery_2170", n_train=2, n_test_good=1, n_test_bad=1)
    (tmp_path / "test_only_cat" / "test" / "good").mkdir(parents=True)  # eval-only category
    (tmp_path / ".cache").mkdir()  # stray folders without the layout are ignored
    (tmp_path / "license").mkdir()
    (tmp_path / "README.txt").write_text("not a category")

    assert discover_categories(tmp_path) == ["battery_2170", "test_only_cat"]
    assert all_categories(tmp_path) == ["battery_2170", "test_only_cat"]
    # "all" with no explicit list now means "everything on disk".
    assert resolve_mvtec_categories(tmp_path) == ["battery_2170", "test_only_cat"]


def test_all_categories_falls_back_to_mvtec_names(tmp_path: Path) -> None:
    assert discover_categories(tmp_path / "missing") == []
    assert all_categories(tmp_path / "missing") == list(MVTec_AD_15_CATEGORIES)
