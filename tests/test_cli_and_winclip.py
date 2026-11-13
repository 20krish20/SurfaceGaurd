from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

from conftest import REPO_ROOT, load_tool
from surfaceguard.data.mvtec import MVTec_AD_15_CATEGORIES
from surfaceguard.models.winclip.api import WinCLIPModel


def test_toy_dataset_cli(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "tools" / "make_synth_mvtec.py"), "--root", str(tmp_path), "--category", "toy"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "wrote" in result.stdout
    cat = tmp_path / "toy"
    assert len(list((cat / "train" / "good").glob("*.png"))) == 20
    assert len(list((cat / "test" / "scratch").glob("*.png"))) == 10
    assert len(list((cat / "ground_truth" / "scratch").glob("*_mask.png"))) == 10


def _eval_args(categories: str = "all", category: str = "") -> argparse.Namespace:
    return argparse.Namespace(categories=categories, category=category)


def test_eval_cli_category_parsing(toy_root: Path, tmp_path: Path) -> None:
    parse = load_tool("eval")._parse_categories
    # Comma lists are de-duplicated and filtered to categories present on disk.
    assert parse(_eval_args("toy, toy ,bottle"), str(toy_root)) == ["toy"]
    # Deprecated --category still works.
    assert parse(_eval_args("all", category="toy"), str(toy_root)) == ["toy"]
    # With no dataset on disk, "all" means the 15 MVTec AD categories.
    assert parse(_eval_args("all"), str(tmp_path / "none")) == list(MVTec_AD_15_CATEGORIES)


def test_eval_cli_all_uses_categories_on_disk(toy_root: Path) -> None:
    # Regression: "all" used to resolve to the 15 MVTec AD names even for other datasets.
    parse = load_tool("eval")._parse_categories
    assert parse(_eval_args("all"), str(toy_root)) == ["toy"]


def test_benchmark_all_resolves_categories_from_disk(toy_root: Path, tmp_path: Path) -> None:
    as_list = load_tool("benchmark_all")._as_list
    assert as_list("all", str(toy_root)) == ["toy"]
    assert as_list("", str(toy_root)) == ["toy"]
    assert as_list("a, b", str(toy_root)) == ["a", "b"]
    assert as_list("all", str(tmp_path / "none")) == list(MVTec_AD_15_CATEGORIES)


class _FakeWinCLIPEngine:
    def __init__(self, score: float) -> None:
        self._score = score

    def score(self, image_bgr: np.ndarray):
        return self._score, np.zeros(image_bgr.shape[:2], dtype=np.float32)


def test_winclip_labels_follow_logit_gap(tmp_path: Path) -> None:
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    defective = WinCLIPModel("bottle", tmp_path, _FakeWinCLIPEngine(0.7)).predict(img)
    normal = WinCLIPModel("bottle", tmp_path, _FakeWinCLIPEngine(-0.2)).predict(img)
    assert defective["label"] == "defective" and defective["score"] == 0.7
    assert normal["label"] == "normal"
    assert normal["heatmap"].shape == (8, 8)
