"""End-to-end smoke test: train PatchCore -> load checkpoint -> evaluate -> infer.

Runs on the synthetic "toy" dataset with a tiny random backbone, so it finishes in
seconds on a CPU-only CI runner and never touches the network.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from surfaceguard.eval.runner import eval_method_for_category
from surfaceguard.infer.runner import infer_single_image
from surfaceguard.models.patchcore.api import PatchCoreModel
from surfaceguard.train.patchcore_trainer import train_patchcore_for_category
from surfaceguard.utils.io import load_npz

pytestmark = pytest.mark.usefixtures("tiny_backbone")


@pytest.fixture
def trained(patchcore_cfg):
    cfg = patchcore_cfg()
    ckpt_dir = Path(train_patchcore_for_category(cfg, category="toy"))
    return cfg, ckpt_dir


def test_training_writes_a_complete_checkpoint(trained) -> None:
    cfg, ckpt_dir = trained
    assert ckpt_dir == Path(cfg["outputs"]["patchcore_dir"]) / "toy"

    meta = yaml.safe_load((ckpt_dir / "meta.yaml").read_text(encoding="utf-8"))
    memory = load_npz(ckpt_dir / "memory.npz")["memory"]

    assert meta["category"] == "toy"
    assert meta["num_train_images"] == 8
    # 8 images x 64 sampled patches, then a 25% coreset.
    assert meta["num_memory_patches"] == memory.shape[0] == int(8 * 64 * 0.25)
    assert meta["patch_dim"] == memory.shape[1] == 128  # resnet18 layer2 channels
    assert meta["coreset_method"] == "kcenter"  # "auto" picks k-center for small pools
    assert math.isfinite(meta["threshold"]) and meta["threshold"] >= meta["train_score_mean"]
    assert len(meta["config_hash"]) == 12
    assert memory.dtype == np.float32


def test_training_is_reproducible(patchcore_cfg, tmp_path) -> None:
    a = train_patchcore_for_category(patchcore_cfg(), category="toy")
    mem_a = load_npz(Path(a) / "memory.npz")["memory"]
    b_cfg = patchcore_cfg(outputs={"patchcore_dir": str(tmp_path / "run_b")})
    mem_b = load_npz(Path(train_patchcore_for_category(b_cfg, category="toy")) / "memory.npz")["memory"]
    np.testing.assert_allclose(mem_a, mem_b, rtol=1e-5, atol=1e-6)


def test_random_coreset_option(patchcore_cfg) -> None:
    cfg = patchcore_cfg(patchcore={"coreset_method": "random"})
    meta = yaml.safe_load((Path(train_patchcore_for_category(cfg, "toy")) / "meta.yaml").read_text())
    assert meta["coreset_method"] == "random"


def test_loaded_model_predicts(trained, toy_root: Path) -> None:
    _, ckpt_dir = trained
    model = PatchCoreModel.load(ckpt_dir)
    assert model.category == "toy"

    images = [cv2.imread(str(p)) for p in sorted((toy_root / "toy" / "test" / "good").glob("*.png"))[:2]]
    preds = model.predict_batch(images)
    assert len(preds) == 2
    for p in preds:
        assert p["label"] in {"normal", "defective"}
        assert math.isfinite(p["score"])
        assert p["heatmap"].shape == (64, 64)
    assert model.predict_batch([]) == []


def test_missing_checkpoint_gives_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Train first"):
        PatchCoreModel.load(tmp_path / "nope")


def test_evaluation_report(trained) -> None:
    cfg, _ = trained
    report_path = Path(eval_method_for_category(cfg, method="patchcore", category="toy"))
    assert report_path == Path(cfg["outputs"]["dir"]) / "reports" / "patchcore_toy_report.json"

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["method"] == "patchcore" and report["category"] == "toy"
    assert report["num_test"] == 12 and report["num_pixel_gt"] == 6
    for key in ("image_auroc", "pixel_auroc", "au_pro"):
        assert 0.0 <= report[key] <= 1.0, key
    assert 0.0 <= report["best_f1"]["f1"] <= 1.0


def test_single_image_inference_matches_input_size(trained, toy_root: Path, tmp_path: Path) -> None:
    cfg, _ = trained
    src = next((toy_root / "toy" / "test" / "scratch").glob("*.png"))
    # Use a non-square image to check the heatmap is resized back to the input.
    big = cv2.resize(cv2.imread(str(src)), (120, 90))
    img_path = tmp_path / "query.png"
    cv2.imwrite(str(img_path), big)

    out = infer_single_image("patchcore", "toy", img_path, outputs_dir=Path(cfg["outputs"]["dir"]))
    assert out["heatmap"].shape == (90, 120)
    assert out["overlay_bgr"].shape == (90, 120, 3)
    assert out["mask_uint8"].shape == (90, 120)
    assert out["label"] in {"normal", "defective"}


def test_unknown_method_is_rejected(toy_root: Path, tmp_path: Path) -> None:
    img = next((toy_root / "toy" / "test" / "good").glob("*.png"))
    with pytest.raises(ValueError, match="Unknown method"):
        infer_single_image("magic", "toy", img, outputs_dir=tmp_path)


def test_unreadable_image_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Could not read image"):
        infer_single_image("patchcore", "toy", tmp_path / "missing.png", outputs_dir=tmp_path)
