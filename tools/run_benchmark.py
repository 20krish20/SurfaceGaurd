from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from surfaceguard.data.mvtec import all_categories, resolve_mvtec_categories
from surfaceguard.eval.runner import eval_method_for_category
from surfaceguard.train.patchcore_trainer import train_patchcore_for_category
from surfaceguard.utils.config import load_yaml
from surfaceguard.utils.io import ensure_dir
from surfaceguard.utils.repro import write_run_manifest


def _parse_csv(v: str) -> List[str]:
    return [x.strip() for x in v.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run multi-category benchmark and aggregate reports.")
    p.add_argument("--categories", type=str, default="all")
    p.add_argument("--all_categories", action="store_true")
    p.add_argument("--methods", type=str, default="patchcore,winclip")
    p.add_argument("--patchcore_config", type=str, default="configs/patchcore_mvtec.yaml")
    p.add_argument("--winclip_config", type=str, default="configs/winclip_mvtec.yaml")
    p.add_argument("--train_patchcore_if_missing", action="store_true")
    p.add_argument("--out_dir", type=str, default="outputs/reports")
    return p.parse_args()


def _read_report(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _aggregate_numeric(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    keys = ["image_auroc", "pixel_auroc", "au_pro"]
    out: Dict[str, float] = {}
    for k in keys:
        vals = [float(r[k]) for r in rows if k in r and r[k] == r[k]]
        if vals:
            out[f"{k}_mean"] = float(sum(vals) / len(vals))
    f1_vals = [float(r.get("best_f1", {}).get("f1")) for r in rows if "best_f1" in r and "f1" in r["best_f1"]]
    f1_vals = [v for v in f1_vals if v == v]
    if f1_vals:
        out["best_f1_mean"] = float(sum(f1_vals) / len(f1_vals))
    return out


def main() -> None:
    args = parse_args()
    methods = _parse_csv(args.methods)

    pc_cfg = load_yaml(args.patchcore_config)
    wc_cfg = load_yaml(args.winclip_config)

    if args.all_categories or (args.categories or "").strip().lower() == "all":
        categories = all_categories(pc_cfg["data"]["root"])
    else:
        categories = resolve_mvtec_categories(pc_cfg["data"]["root"], _parse_csv(args.categories))

    if not categories:
        raise ValueError("No categories selected. Provide --categories or use --all_categories.")

    rows: List[Dict[str, Any]] = []
    for method in methods:
        cfg = pc_cfg if method == "patchcore" else wc_cfg
        outputs_dir = Path(cfg.get("outputs", {}).get("dir", "outputs"))
        for category in categories:
            if method == "patchcore":
                ckpt_dir = outputs_dir / "patchcore" / category
                if not ckpt_dir.exists():
                    if args.train_patchcore_if_missing:
                        print(f"[train] patchcore category={category}")
                        train_patchcore_for_category(cfg, category=category)
                    else:
                        print(f"[skip] missing checkpoint for category={category} at {ckpt_dir}")
                        continue
            print(f"[eval] method={method} category={category}")
            report_path = eval_method_for_category(cfg, method=method, category=category)
            row = _read_report(report_path)
            row["report_path"] = report_path
            rows.append(row)

    by_method: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_method.setdefault(r["method"], []).append(r)

    aggregate = {m: _aggregate_numeric(rs) for m, rs in by_method.items()}
    out = {
        "kind": "benchmark_summary",
        "methods": methods,
        "categories_requested": categories,
        "num_reports": len(rows),
        "aggregate": aggregate,
        "reports": rows,
    }

    out_dir = ensure_dir(args.out_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"benchmark_summary_{stamp}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    manifest_path = write_run_manifest(
        output_root=Path(args.out_dir).parent,
        run_name="benchmark",
        args=vars(args),
        config_paths=[args.patchcore_config, args.winclip_config],
        extra={"summary_path": str(out_path)},
    )
    print(f"summary={out_path}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
