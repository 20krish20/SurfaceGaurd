from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict

import numpy as np
import torch

if TYPE_CHECKING:  # type-only import: keeps open_clip out of module import time
    from surfaceguard.models.winclip.core import WinCLIPEngine


@dataclass
class WinCLIPModel:
    category: str
    cache_dir: Path
    engine: "WinCLIPEngine"

    @classmethod
    def load(
        cls,
        cache_dir: Path,
        category: str,
        window_size: int = 224,
        window_stride: int = 112,
        blur_sigma: float = 0.0,
        enable_int8: bool = False,
    ) -> "WinCLIPModel":
        if not category or category.strip().lower() == "auto":
            category = "object"
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        from surfaceguard.models.winclip.core import WinCLIPState, WinCLIPEngine

        # Defaults match configs/winclip_mvtec.yaml; can be made configurable later via app.
        state = WinCLIPState(
            model="ViT-B-32",
            pretrained="laion2b_s34b_b79k",
            window_size=int(window_size),
            window_stride=int(window_stride),
            normal_prompts=[
                "a photo of a clean {category}",
                "a photo of a normal {category}",
                "a high quality photo of a pristine {category}",
            ],
            anomalous_prompts=[
                "a photo of a defective {category}",
                "a photo of a damaged {category}",
                "a photo of an anomalous {category}",
            ],
            image_size=224,
            blur_sigma=float(blur_sigma),
        )
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        engine = WinCLIPEngine(state, device=device, category=category, enable_int8=enable_int8)
        return cls(category=category, cache_dir=cache_dir, engine=engine)

    def predict(self, image_bgr: np.ndarray) -> Dict:
        score, heat = self.engine.score(image_bgr)
        # score is a logit-gap (anom - normal); >0 leans anomalous.
        label = "defective" if score > 0.0 else "normal"
        return {"label": label, "score": float(score), "heatmap": heat}

