"""Shared pytest fixtures.

The suite is hermetic: it never downloads model weights or datasets. PatchCore tests
swap the pretrained Wide-ResNet50 backbone for a small, randomly initialised ResNet-18,
and every dataset is generated on the fly with ``tools/make_synth_mvtec.py``.
"""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable, Dict

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"


def load_tool(name: str):
    """Import a script from ``tools/`` (it is not a package) as a module."""
    path = TOOLS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"sg_tools_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def make_toy() -> Callable[..., Path]:
    """Return the toy-dataset generator from ``tools/make_synth_mvtec.py``."""
    return load_tool("make_synth_mvtec").make_toy_mvtec


@pytest.fixture(scope="session")
def toy_root(tmp_path_factory: pytest.TempPathFactory, make_toy) -> Path:
    """A small MVTec-style dataset: data/<root>/toy/{train,test,ground_truth}."""
    root = tmp_path_factory.mktemp("mvtec_ad")
    make_toy(root, "toy", n_train=8, n_test_good=6, n_test_bad=6)
    return root


def _tiny_backbone(_name: str) -> torch.nn.Module:
    # Seeded so feature extraction (and therefore every score) is reproducible.
    torch.manual_seed(0)
    from torchvision.models import resnet18

    model = resnet18(weights=None)
    model.eval()
    return model


@pytest.fixture
def tiny_backbone(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the pretrained backbone everywhere PatchCore builds one (no downloads)."""
    import surfaceguard.models.patchcore.core as core
    import surfaceguard.train.patchcore_trainer as trainer

    monkeypatch.setattr(core, "_build_backbone", _tiny_backbone)
    # The trainer imports the function by name, so patch its reference too.
    monkeypatch.setattr(trainer, "_build_backbone", _tiny_backbone)
    # Engines cache backbones per (name, device); start each test with a clean cache.
    monkeypatch.setattr(core, "_SHARED_BACKBONES", {})


@pytest.fixture
def patchcore_cfg(tmp_path: Path, toy_root: Path) -> Callable[..., Dict[str, Any]]:
    """Factory for a fast PatchCore config that writes outputs under ``tmp_path``."""
    base: Dict[str, Any] = {
        "data": {"dataset": "mvtec_ad", "root": str(toy_root), "image_size": 64, "normalize": "imagenet"},
        "patchcore": {
            "backbone": "resnet18",
            "layer": "layer2",
            "coreset_fraction": 0.25,
            "coreset_method": "auto",
            "patches_per_image": 64,
            "max_patch_samples": 5000,
            "nn_k": 1,
            "image_score": "max",
            "topk": 10,
        },
        "train": {"batch_size": 4, "num_workers": 0, "seed": 42},
        "outputs": {"dir": str(tmp_path / "outputs"), "patchcore_dir": str(tmp_path / "outputs" / "patchcore")},
    }

    def _make(**overrides: Dict[str, Any]) -> Dict[str, Any]:
        cfg = copy.deepcopy(base)
        for section, values in overrides.items():
            cfg.setdefault(section, {}).update(values)
        return cfg

    return _make
