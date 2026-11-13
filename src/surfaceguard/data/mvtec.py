from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

MVTec_AD_15_CATEGORIES: Tuple[str, ...] = (
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
)


def _list_images(folder: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    if not folder.exists():
        return []
    paths = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(paths)


@dataclass(frozen=True)
class MVTecSample:
    image_path: Path
    mask_path: Optional[Path]
    label: int  # 0 normal, 1 anomalous
    defect_type: str


class MVTecADDataset(Dataset):
    """
    MVTec AD dataset wrapper supporting:
    - train split: only normal images (train/good)
    - test split: normal + anomalous, with pixel masks for anomalous images
    """

    def __init__(
        self,
        root: str | Path,
        category: str,
        split: str,
        image_transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
    ) -> None:
        self.root = Path(root)
        self.category = category
        self.split = split
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        self.category_dir = self.root / category
        if not self.category_dir.exists():
            available = mvtec_category_names(self.root)
            hint = ""
            if available:
                hint = f" Available categories under {self.root}: {', '.join(available[:20])}" + (
                    " ..." if len(available) > 20 else ""
                )
            else:
                hint = (
                    f" Dataset root not found or empty: {self.root}. "
                    f"Place MVTec AD at data/mvtec_ad/<category>/... or create a toy dataset via "
                    f"`python tools\\make_synth_mvtec.py --category toy`."
                )
            raise FileNotFoundError(f"Category not found: {self.category_dir}.{hint}")

        if split not in {"train", "test"}:
            raise ValueError("split must be train or test")

        self.samples: List[MVTecSample] = self._build_samples()

    def _build_samples(self) -> List[MVTecSample]:
        if self.split == "train":
            imgs = _list_images(self.category_dir / "train" / "good")
            return [MVTecSample(p, None, 0, "good") for p in imgs]

        # test split
        test_dir = self.category_dir / "test"
        gt_dir = self.category_dir / "ground_truth"
        samples: List[MVTecSample] = []

        for defect_dir in sorted([d for d in test_dir.iterdir() if d.is_dir()]):
            defect_type = defect_dir.name
            img_paths = _list_images(defect_dir)
            if defect_type == "good":
                samples.extend([MVTecSample(p, None, 0, "good") for p in img_paths])
            else:
                # ground truth masks mirror filenames (sometimes *_mask.png)
                mask_base = gt_dir / defect_type
                for p in img_paths:
                    # MVTec mask file is typically same stem + "_mask.png"
                    cand1 = mask_base / f"{p.stem}_mask.png"
                    cand2 = mask_base / p.name
                    mask_path = cand1 if cand1.exists() else (cand2 if cand2.exists() else None)
                    samples.append(MVTecSample(p, mask_path, 1, defect_type))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = Image.open(s.image_path).convert("RGB")
        mask = None
        if s.mask_path is not None and s.mask_path.exists():
            mask = Image.open(s.mask_path).convert("L")

        if self.image_transform is not None:
            img_t = self.image_transform(img)
        else:
            img_t = img

        has_mask = mask is not None
        if mask is not None:
            if self.mask_transform is not None:
                mask_t = self.mask_transform(mask)
            else:
                mask_t = mask
            mask_t = (mask_t > 0.5).float()
        else:
            # keep collate-friendly output even for normal images
            if hasattr(img_t, "shape"):
                h, w = int(img_t.shape[-2]), int(img_t.shape[-1])
            else:
                h, w = img.size[1], img.size[0]
            mask_t = torch.zeros((1, h, w), dtype=torch.float32)

        return {
            "image": img_t,
            "mask": mask_t,
            "has_mask": has_mask,
            "label": s.label,
            "defect_type": s.defect_type,
            "image_path": str(s.image_path),
            "mask_path": str(s.mask_path) if s.mask_path is not None else "",
        }


def mvtec_category_names(root: str | Path) -> List[str]:
    root_p = Path(root)
    if not root_p.exists():
        return []
    return sorted([d.name for d in root_p.iterdir() if d.is_dir()])


def discover_categories(root: str | Path) -> List[str]:
    """
    Category folders under ``root`` that follow the MVTec layout (contain ``train/good`` or ``test``).
    Works for any dataset arranged that way, not just MVTec AD (e.g. the toy set or battery CT slices).
    Stray folders without that layout (caches, licence folders, ...) are ignored.
    """
    root_p = Path(root)
    if not root_p.is_dir():
        return []
    return sorted(
        d.name
        for d in root_p.iterdir()
        if d.is_dir() and ((d / "train" / "good").is_dir() or (d / "test").is_dir())
    )


def all_categories(root: str | Path) -> List[str]:
    """What "all" means: every category found on disk, or the 15 MVTec AD names if none are there yet."""
    return discover_categories(root) or list(MVTec_AD_15_CATEGORIES)


def resolve_mvtec_categories(root: str | Path, categories: Optional[List[str]] = None) -> List[str]:
    requested = [c.strip() for c in (categories or []) if c and c.strip()]
    if not requested:
        return all_categories(root)

    available = set(mvtec_category_names(root))
    if not available:
        return requested

    resolved = [c for c in requested if c in available]
    if resolved:
        return resolved
    return requested

