from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from surfaceguard.data.mvtec import all_categories, resolve_mvtec_categories
from surfaceguard.utils.config import load_yaml
from surfaceguard.eval.runner import eval_method_for_category


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--method", required=True, choices=["patchcore", "winclip"])
    p.add_argument("--category", default="", type=str, help="Single category (deprecated; use --categories).")
    p.add_argument("--categories", default="all", type=str, help="Comma list, or 'all' for every category folder under the dataset root.")
    p.add_argument("--out_summary", default="", type=str, help="Optional output path for aggregated summary JSON.")
    return p.parse_args()


def _parse_categories(args: argparse.Namespace, dataset_root: str) -> List[str]:
    raw = (args.categories or "").strip().lower()
    if args.category and args.category.strip() and (raw == "all" or not raw):
        return resolve_mvtec_categories(dataset_root, [args.category.strip()])
    if raw == "all" or not raw:
        return all_categories(dataset_root)
    parsed = [x.strip() for x in args.categories.split(",") if x.strip()]
    if args.category and args.category.strip():
        parsed.append(args.category.strip())
    deduped = list(dict.fromkeys(parsed))
    return resolve_mvtec_categories(dataset_root, deduped)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    categories = _parse_categories(args, dataset_root=str(cfg["data"]["root"]))
    if not categories:
        raise ValueError("No categories resolved for evaluation.")

    reports = []
    for cat in categories:
        report_path = eval_method_for_category(cfg, method=args.method, category=cat)
        reports.append(json.loads(Path(report_path).read_text(encoding="utf-8")))
        print(f"report={report_path}")

    summary = {
        "kind": "eval_summary",
        "method": args.method,
        "categories": categories,
        "num_reports": len(reports),
        "reports": reports,
    }
    if args.out_summary:
        out_path = Path(args.out_summary)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"summary={out_path}")


if __name__ == "__main__":
    main()

