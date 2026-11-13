from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from surfaceguard.data.mvtec import all_categories
from surfaceguard.eval.runner import eval_method_for_category
from surfaceguard.utils.config import load_yaml
from surfaceguard.utils.io import ensure_dir
from surfaceguard.utils.provenance import get_git_commit_short, stable_cfg_hash

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=str, help="Path to YAML config")
    p.add_argument("--method", required=True, choices=["patchcore", "winclip", "both"])
    p.add_argument("--categories", default="all", type=str, help="Comma list or 'all'")
    return p.parse_args()


def _as_list(s: str, dataset_root: str) -> List[str]:
    s = (s or "").strip()
    if not s or s.lower() == "all":
        return all_categories(dataset_root)
    return [x.strip() for x in s.split(",") if x.strip()]


def main() -> None:
    args = parse_args()
    cfg: Dict[str, Any] = load_yaml(args.config)

    outputs_dir = Path(cfg.get("outputs", {}).get("dir", "outputs"))
    reports_dir = ensure_dir(outputs_dir / "reports")

    categories = _as_list(args.categories, dataset_root=str(cfg["data"]["root"]))
    methods = ["patchcore", "winclip"] if args.method == "both" else [args.method]

    results: List[Dict[str, Any]] = []
    for method in methods:
        for cat in categories:
            path = eval_method_for_category(cfg, method=method, category=cat)
            report = json.loads(Path(path).read_text(encoding="utf-8"))
            results.append(report)

    summary = {
        "config_path": str(Path(args.config)),
        "config_hash": stable_cfg_hash(cfg),
        "git_commit": get_git_commit_short(),
        "methods": methods,
        "categories": categories,
        "num_reports": len(results),
        "reports": results,
    }

    out_path = reports_dir / f"benchmark_{'_'.join(methods)}_all_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(str(out_path))


if __name__ == "__main__":
    main()

