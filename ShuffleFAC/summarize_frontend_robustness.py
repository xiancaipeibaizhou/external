#!/usr/bin/env python3
"""Summarize non-SN cross-front-end recording-level robustness results."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


INCLUDED_HEADS = ["mean", "attention", "bigru", "mil_linear_softmax"]
RAW_FIELDS = [
    "dataset",
    "seed",
    "frontend",
    "head",
    "test_macro_f1",
    "test_acc",
    "test_macro_precision",
    "test_macro_recall",
    "embed_dim_raw",
    "adapter_dim",
    "clips_per_recording",
    "total_params",
    "trainable_params",
    "source_summary_csv",
    "source_run_dir",
    "is_smoke",
]
SUMMARY_FIELDS = [
    "frontend",
    "head",
    "num_seeds",
    "mean_test_macro_f1",
    "std_test_macro_f1",
    "mean_test_acc",
    "std_test_acc",
    "mean_test_macro_precision",
    "mean_test_macro_recall",
]
BEST_FIELDS = ["frontend", "best_head", "mean_test_macro_f1", "std_test_macro_f1", "num_seeds"]
FRONTEND_ORDER = ["shufflefac", "resnet18", "mobilenet_v2", "panns_cnn14"]
HEAD_ORDER = {head: idx for idx, head in enumerate(INCLUDED_HEADS)}
SMOKE_KEYWORDS = ["smoke", "debug", "_smoke"]


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize non-SN front-end robustness baselines.")
    parser.add_argument("--results_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset", default="DeepShip")
    parser.add_argument("--exclude_heads", default="sn_decoupled")
    parser.add_argument("--include_smoke", action="store_true")
    return parser.parse_args()


def split_arg(value):
    return {item.strip() for item in str(value).split(",") if item.strip()}


def read_csv_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f)


def first_present(row, names):
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip() != "":
            return value
    return None


def to_float(value):
    if value is None or str(value).strip() == "":
        return math.nan
    return float(value)


def to_int_or_empty(value):
    if value is None or str(value).strip() == "":
        return ""
    return int(float(value))


def mean(values):
    values = [float(v) for v in values if math.isfinite(float(v))]
    if not values:
        return math.nan
    return sum(values) / len(values)


def sample_std(values):
    values = [float(v) for v in values if math.isfinite(float(v))]
    if len(values) < 2:
        return 0.0
    avg = sum(values) / len(values)
    return math.sqrt(sum((v - avg) ** 2 for v in values) / (len(values) - 1))


def frontend_sort_key(frontend):
    try:
        return (FRONTEND_ORDER.index(frontend), frontend)
    except ValueError:
        return (len(FRONTEND_ORDER), frontend)


def row_preference(row):
    score = 0
    if not row.get("is_smoke"):
        score += 100
    if "frontend_robustness" in str(row.get("source_summary_csv", "")).lower():
        score += 10
    return score


def path_has_smoke_keyword(path: Path):
    for part in path.parts:
        lower = part.lower()
        if any(keyword in lower for keyword in SMOKE_KEYWORDS):
            return True
    return False


def is_one(value):
    if value is None or str(value).strip() == "":
        return False
    try:
        return int(float(value)) == 1
    except ValueError:
        return False


def detect_smoke(row, summary_path: Path):
    source_smoke = path_has_smoke_keyword(summary_path)
    if source_smoke:
        return True
    if is_one(row.get("epochs")):
        return True
    if is_one(row.get("best_epoch")) and "frontend_robustness_smoke" in str(summary_path).lower():
        return True
    for key in ["frontend_ckpt", "output_dir"]:
        value = row.get(key)
        if value is not None and any(keyword in str(value).lower() for keyword in SMOKE_KEYWORDS):
            return True
    return False


def normalize_row(row, summary_path: Path, dataset: str):
    head = str(row.get("head", "")).strip()
    frontend = str(row.get("frontend", "")).strip()
    if not head or not frontend:
        return None
    row_dataset = str(row.get("dataset", dataset) or dataset).strip()
    if row_dataset and row_dataset != dataset:
        return None
    test_macro_f1 = first_present(row, ["test_macro_f1", "Macro-F1", "test_f1"])
    test_acc = first_present(row, ["test_acc", "ACC", "test_accuracy", "accuracy"])
    test_macro_precision = first_present(row, ["test_macro_precision", "Macro-Precision", "test_precision"])
    test_macro_recall = first_present(row, ["test_macro_recall", "Macro-Recall", "test_recall"])
    normalized = {
        "dataset": row_dataset or dataset,
        "seed": to_int_or_empty(row.get("seed")),
        "frontend": frontend,
        "head": head,
        "test_macro_f1": to_float(test_macro_f1),
        "test_acc": to_float(test_acc),
        "test_macro_precision": to_float(test_macro_precision),
        "test_macro_recall": to_float(test_macro_recall),
        "embed_dim_raw": to_int_or_empty(first_present(row, ["embed_dim_raw", "embed_dim"])),
        "adapter_dim": to_int_or_empty(row.get("adapter_dim")),
        "clips_per_recording": to_int_or_empty(row.get("clips_per_recording")),
        "total_params": to_int_or_empty(row.get("total_params")),
        "trainable_params": to_int_or_empty(row.get("trainable_params")),
        "source_summary_csv": str(summary_path),
        "source_run_dir": str(summary_path.parent),
        "is_smoke": detect_smoke(row, summary_path),
    }
    return normalized


def collect_all_rows(results_root: Path, dataset: str, exclude_heads: set):
    rows = []
    allowed_heads = set(INCLUDED_HEADS) - set(exclude_heads)
    for summary_path in sorted(results_root.rglob("summary.csv")):
        for row in read_csv_rows(summary_path):
            normalized = normalize_row(row, summary_path, dataset)
            if normalized is None:
                continue
            if normalized["head"] not in allowed_heads:
                continue
            if normalized["head"] in exclude_heads:
                continue
            rows.append(normalized)
    rows.sort(key=lambda r: (frontend_sort_key(r["frontend"]), HEAD_ORDER.get(r["head"], 99), r["seed"], r["source_summary_csv"]))
    return rows


def dedupe_rows(rows):
    best_by_key = {}
    for row in rows:
        key = (row["frontend"], row["head"], row["seed"])
        prev = best_by_key.get(key)
        if prev is None or row_preference(row) > row_preference(prev) or (
            row_preference(row) == row_preference(prev) and row["source_summary_csv"] < prev["source_summary_csv"]
        ):
            best_by_key[key] = row
    filtered = [{field: row[field] for field in RAW_FIELDS} for row in best_by_key.values()]
    filtered.sort(key=lambda r: (frontend_sort_key(r["frontend"]), HEAD_ORDER.get(r["head"], 99), r["seed"]))
    return filtered


def summarize(raw_rows):
    groups = defaultdict(list)
    for row in raw_rows:
        groups[(row["frontend"], row["head"])].append(row)

    summary_rows = []
    for (frontend, head), rows in groups.items():
        f1 = [row["test_macro_f1"] for row in rows]
        acc = [row["test_acc"] for row in rows]
        precision = [row["test_macro_precision"] for row in rows]
        recall = [row["test_macro_recall"] for row in rows]
        seeds = {row["seed"] for row in rows if row["seed"] != ""}
        summary_rows.append(
            {
                "frontend": frontend,
                "head": head,
                "num_seeds": len(seeds),
                "mean_test_macro_f1": mean(f1),
                "std_test_macro_f1": sample_std(f1),
                "mean_test_acc": mean(acc),
                "std_test_acc": sample_std(acc),
                "mean_test_macro_precision": mean(precision),
                "mean_test_macro_recall": mean(recall),
            }
        )
    summary_rows.sort(key=lambda r: (frontend_sort_key(r["frontend"]), HEAD_ORDER.get(r["head"], 99)))
    return summary_rows


def best_by_frontend(summary_rows):
    groups = defaultdict(list)
    for row in summary_rows:
        groups[row["frontend"]].append(row)
    best_rows = []
    for frontend, rows in groups.items():
        best = max(rows, key=lambda r: (r["mean_test_macro_f1"], -HEAD_ORDER.get(r["head"], 99)))
        best_rows.append(
            {
                "frontend": frontend,
                "best_head": best["head"],
                "mean_test_macro_f1": best["mean_test_macro_f1"],
                "std_test_macro_f1": best["std_test_macro_f1"],
                "num_seeds": best["num_seeds"],
            }
        )
    best_rows.sort(key=lambda r: frontend_sort_key(r["frontend"]))
    return best_rows


def incomplete_frontends(filtered_rows, all_rows):
    observed_frontends = {row["frontend"] for row in all_rows}
    formal_seeds = defaultdict(set)
    for row in filtered_rows:
        if row["seed"] != "":
            formal_seeds[row["frontend"]].add(row["seed"])
    out = []
    for frontend in sorted(observed_frontends, key=frontend_sort_key):
        seeds = formal_seeds.get(frontend, set())
        if len(seeds) < 3:
            out.append({"frontend": frontend, "num_seeds": len(seeds), "observed_seeds": sorted(seeds)})
    return out


def write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    exclude_heads = split_arg(args.exclude_heads)

    raw_all_rows = collect_all_rows(results_root, args.dataset, exclude_heads)
    candidate_rows = raw_all_rows if args.include_smoke else [row for row in raw_all_rows if not row["is_smoke"]]
    raw_rows = dedupe_rows(candidate_rows)
    summary_rows = summarize(raw_rows)
    best_rows = best_by_frontend(summary_rows)
    incomplete = incomplete_frontends(raw_rows, raw_all_rows)
    excluded_smoke_rows = [row for row in raw_all_rows if row["is_smoke"]]

    write_csv(output_dir / "cross_frontend_non_sn_raw_all.csv", raw_all_rows, RAW_FIELDS)
    write_csv(output_dir / "cross_frontend_non_sn_raw.csv", raw_rows, RAW_FIELDS)
    write_csv(output_dir / "cross_frontend_non_sn_summary.csv", summary_rows, SUMMARY_FIELDS)
    write_csv(output_dir / "best_non_sn_by_frontend.csv", best_rows, BEST_FIELDS)
    with (output_dir / "cross_frontend_non_sn_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset": args.dataset,
                "results_root": str(results_root),
                "exclude_heads": sorted(exclude_heads),
                "include_smoke": bool(args.include_smoke),
                "included_heads": [head for head in INCLUDED_HEADS if head not in exclude_heads],
                "num_raw_all_rows": len(raw_all_rows),
                "num_filtered_rows": len(raw_rows),
                "num_excluded_smoke_rows": len(excluded_smoke_rows) if not args.include_smoke else 0,
                "incomplete_frontends": incomplete,
                "raw_rows": raw_rows,
                "summary": summary_rows,
                "best_non_sn_by_frontend": best_rows,
                "excluded_smoke_rows": excluded_smoke_rows if not args.include_smoke else [],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"raw all rows: {len(raw_all_rows)}")
    print(f"filtered rows: {len(raw_rows)}")
    if not args.include_smoke:
        print(f"excluded smoke rows: {len(excluded_smoke_rows)}")
    print(f"summary rows: {len(summary_rows)}")
    for row in best_rows:
        print(
            f"{row['frontend']}: {row['best_head']} "
            f"{row['mean_test_macro_f1']:.4f} +/- {row['std_test_macro_f1']:.4f} "
            f"(n={row['num_seeds']})"
        )


if __name__ == "__main__":
    main()
