#!/usr/bin/env python3
"""Train TorchVision clip-level front-ends on strict recording-level caches.

The model is optimized with clip-level cross entropy, but checkpoints are
selected by validation recording-level Macro-F1 after mean-probability pooling.
This keeps the front-end selection aligned with downstream recording-level
aggregation experiments.
"""

import argparse
import atexit
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset

from model.frontends_torchvision import build_clip_frontend


SPLITS = ["train", "val", "test"]


class TeeStream:
    """Mirror stdout/stderr to console and output_dir/run.log."""

    def __init__(self, *streams):
        self.streams = streams
        self.encoding = getattr(streams[0], "encoding", "utf-8") if streams else "utf-8"
        self.errors = getattr(streams[0], "errors", "replace") if streams else "replace"

    def write(self, data):
        for stream in self.streams:
            stream.write(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return bool(self.streams and self.streams[0].isatty())

    def __getattr__(self, name):
        return getattr(self.streams[0], name)


def install_run_log(output_dir: Path):
    """Capture stdout/stderr in output_dir/run.log while preserving console output."""

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"
    log_file = log_path.open("w", encoding="utf-8")
    sys.stdout = TeeStream(sys.__stdout__, log_file)
    sys.stderr = TeeStream(sys.__stderr__, log_file)

    def close_log():
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            log_file.close()

    atexit.register(close_log)
    print(f"run log: {log_path}", flush=True)
    print(f"command: {' '.join(sys.argv)}", flush=True)


def torch_load(path: Path, map_location="cpu"):
    """Load torch checkpoints across PyTorch versions."""

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def resolve_path(path: str, root: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_cache_paths(config: dict, root: Path) -> dict:
    cache_paths = config.get("cache_paths")
    if not isinstance(cache_paths, dict):
        raise ValueError("model_config must contain cache_paths")
    out = {}
    for split in SPLITS:
        if split not in cache_paths:
            raise ValueError(f"cache_paths missing split: {split}")
        out[split] = resolve_path(cache_paths[split], root)
    return out


def validate_source_cache(cache_path: Path, split_name: str):
    payload = torch_load(cache_path, map_location="cpu")
    metadata = payload.get("metadata", {})
    protocol = metadata.get("protocol")
    actual_split = payload.get("split_name")
    if protocol != "recording_level":
        raise ValueError(f"{cache_path} is not strict recording-level cache. metadata.protocol={protocol!r}")
    if actual_split and actual_split != split_name:
        raise ValueError(f"{cache_path} split mismatch: expected {split_name}, got {actual_split}")
    for key in ["features", "labels", "entries"]:
        if key not in payload:
            raise ValueError(f"{cache_path} missing key: {key}")
    features = payload["features"]
    if not torch.is_tensor(features) or features.ndim != 4:
        raise ValueError(f"{cache_path} features must be a 4-D tensor [N, 1, F, T], got {type(features)} {getattr(features, 'shape', None)}")
    if int(features.size(1)) != 1:
        raise ValueError(f"{cache_path} features must have one input channel, got shape {tuple(features.shape)}")
    return payload


def recording_id_from_entry(entry: dict, fallback_index: int):
    rid = entry.get("recording_id")
    if rid:
        return str(rid)
    path = entry.get("path")
    return Path(path).stem if path else str(fallback_index)


class ClipCacheDataset(Dataset):
    """Clip-level dataset backed by a strict recording-level feature cache."""

    def __init__(self, cache_payload: dict):
        self.features = cache_payload["features"].float()
        self.labels = cache_payload["labels"].long()
        self.entries = cache_payload.get("entries", [])
        if self.features.ndim != 4:
            raise ValueError(f"Expected [N, 1, F, T] features, got {tuple(self.features.shape)}")
        if len(self.entries) != int(self.labels.numel()):
            raise ValueError(f"entries/labels length mismatch: {len(self.entries)} vs {int(self.labels.numel())}")
        self.recording_ids = []
        self.segment_indices = []
        for index, entry in enumerate(self.entries):
            self.recording_ids.append(recording_id_from_entry(entry, index))
            self.segment_indices.append(int(entry.get("segment_index", -1)))

    def __len__(self):
        return int(self.labels.numel())

    def __getitem__(self, index):
        return (
            self.features[index],
            self.labels[index],
            self.recording_ids[index],
            torch.tensor(self.segment_indices[index], dtype=torch.long),
        )


def validate_recording_level_no_leakage(datasets: dict):
    sets = {split: set(dataset.recording_ids) for split, dataset in datasets.items()}
    overlaps = {}
    for left, right in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = sorted(sets[left].intersection(sets[right]))
        overlaps[f"{left}-{right}"] = overlap
        if overlap:
            raise ValueError(f"Recording-level leakage detected for {left}-{right}: {overlap[:20]}")
    return overlaps


def class_names_from_config_or_cache(config: dict, cache_payloads: dict):
    split_meta = config.get("split_metadata", {})
    mapping = split_meta.get("class_mapping")
    if not isinstance(mapping, dict):
        for payload in cache_payloads.values():
            metadata = payload.get("metadata", {})
            mapping = metadata.get("class_mapping")
            if isinstance(mapping, dict):
                break
    if isinstance(mapping, dict):
        try:
            return [name for name, _idx in sorted(mapping.items(), key=lambda item: int(item[1]))]
        except Exception:
            return None
    return None


def metrics_from_arrays(y_true, y_pred):
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device, collect_predictions=False):
    model.eval()
    total_loss = 0.0
    total_seen = 0
    y_true = []
    logits_all = []
    recording_ids = []
    segment_indices = []
    for features, labels, rids, seg_idx in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(features)
        loss = criterion(logits, labels)
        total_loss += float(loss.detach().cpu()) * int(labels.size(0))
        total_seen += int(labels.size(0))
        y_true.extend(labels.detach().cpu().numpy().tolist())
        logits_all.append(logits.detach().cpu())
        recording_ids.extend([str(rid) for rid in rids])
        if torch.is_tensor(seg_idx):
            segment_indices.extend([int(v) for v in seg_idx.detach().cpu().numpy().tolist()])
        else:
            segment_indices.extend([int(v) for v in seg_idx])

    logits_np = torch.cat(logits_all, dim=0).numpy()
    prob_np = softmax_np(logits_np)
    y_true_np = np.asarray(y_true, dtype=np.int64)
    pred_np = prob_np.argmax(axis=1).astype(np.int64)
    clip_metrics = metrics_from_arrays(y_true_np, pred_np)
    clip_metrics["loss"] = total_loss / max(total_seen, 1)
    recording_metrics, recording_payload = recording_mean_prob_metrics(
        y_true_np,
        prob_np,
        recording_ids,
    )
    out = {
        "clip_metrics": clip_metrics,
        "recording_metrics": recording_metrics,
    }
    if collect_predictions:
        out.update(
            {
                "y_true": y_true_np,
                "y_pred": pred_np,
                "y_prob": prob_np,
                "recording_ids": recording_ids,
                "segment_indices": segment_indices,
                "recording_payload": recording_payload,
            }
        )
    return out


def softmax_np(logits: np.ndarray):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def recording_mean_prob_metrics(y_true, y_prob, recording_ids):
    groups = defaultdict(lambda: {"labels": [], "probs": []})
    for label, prob, rid in zip(y_true, y_prob, recording_ids):
        groups[str(rid)]["labels"].append(int(label))
        groups[str(rid)]["probs"].append(np.asarray(prob, dtype=np.float64))

    rec_ids = []
    rec_true = []
    rec_prob = []
    rec_pred = []
    for rid in sorted(groups.keys()):
        labels = groups[rid]["labels"]
        label_set = sorted(set(labels))
        if len(label_set) != 1:
            raise ValueError(f"Recording {rid} has inconsistent labels during aggregation: {label_set}")
        mean_prob = np.stack(groups[rid]["probs"], axis=0).mean(axis=0)
        rec_ids.append(rid)
        rec_true.append(label_set[0])
        rec_prob.append(mean_prob)
        rec_pred.append(int(mean_prob.argmax()))

    rec_true_np = np.asarray(rec_true, dtype=np.int64)
    rec_pred_np = np.asarray(rec_pred, dtype=np.int64)
    metrics = metrics_from_arrays(rec_true_np, rec_pred_np)
    payload = {
        "recording_ids": rec_ids,
        "y_true": rec_true_np,
        "y_pred": rec_pred_np,
        "y_prob": np.stack(rec_prob, axis=0) if rec_prob else np.zeros((0, 0), dtype=np.float64),
    }
    return metrics, payload


def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip: float):
    model.train()
    total_loss = 0.0
    total_seen = 0
    y_true = []
    logits_all = []
    for features, labels, _rids, _seg_idx in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * int(labels.size(0))
        total_seen += int(labels.size(0))
        y_true.extend(labels.detach().cpu().numpy().tolist())
        logits_all.append(logits.detach().cpu())

    logits_np = torch.cat(logits_all, dim=0).numpy()
    prob_np = softmax_np(logits_np)
    pred_np = prob_np.argmax(axis=1).astype(np.int64)
    metrics = metrics_from_arrays(np.asarray(y_true, dtype=np.int64), pred_np)
    metrics["loss"] = total_loss / max(total_seen, 1)
    return metrics


def count_params(model):
    return {
        "total_params": int(sum(p.numel() for p in model.parameters())),
        "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    }


def write_epoch_csv(path: Path, rows: list):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_metrics_txt(path: Path, metrics: dict):
    with path.open("w", encoding="utf-8") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")


def save_clip_predictions(path: Path, eval_payload: dict, num_classes: int):
    fields = ["recording_id", "segment_index", "true_label", "pred_label"]
    fields += [f"prob_class_{idx}" for idx in range(num_classes)]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for idx, rid in enumerate(eval_payload["recording_ids"]):
            row = {
                "recording_id": rid,
                "segment_index": int(eval_payload["segment_indices"][idx]),
                "true_label": int(eval_payload["y_true"][idx]),
                "pred_label": int(eval_payload["y_pred"][idx]),
            }
            for class_idx in range(num_classes):
                row[f"prob_class_{class_idx}"] = float(eval_payload["y_prob"][idx, class_idx])
            writer.writerow(row)


def save_recording_predictions(path: Path, recording_payload: dict, num_classes: int):
    fields = ["recording_id", "true_label", "pred_label"]
    fields += [f"prob_class_{idx}" for idx in range(num_classes)]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for idx, rid in enumerate(recording_payload["recording_ids"]):
            row = {
                "recording_id": rid,
                "true_label": int(recording_payload["y_true"][idx]),
                "pred_label": int(recording_payload["y_pred"][idx]),
            }
            for class_idx in range(num_classes):
                row[f"prob_class_{class_idx}"] = float(recording_payload["y_prob"][idx, class_idx])
            writer.writerow(row)


def infer_num_classes(datasets: dict, config: dict):
    labels = torch.cat([dataset.labels for dataset in datasets.values()])
    from_labels = int(labels.max().item()) + 1
    cnn_cfg = config.get("cnn_cfg_used") or config.get("CNN") or {}
    from_config = cnn_cfg.get("n_class")
    if from_config is not None:
        return int(from_config)
    return from_labels


def parse_args():
    parser = argparse.ArgumentParser(description="Train TorchVision clip-level front-ends from ShuffleFAC recording-level caches.")
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--frontend", choices=["resnet18", "mobilenet_v2"], required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.0, help="Reserved for future front-end classifiers; current classifiers are linear.")
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--eval_recording_aggregation", choices=["mean_prob"], default="mean_prob")
    parser.add_argument("--pretrained", choices=["none", "imagenet"], default="none")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path.cwd()
    output_dir = resolve_path(args.output_dir, root)
    output_dir.mkdir(parents=True, exist_ok=True)
    install_run_log(output_dir)
    set_seed(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.", flush=True)
        device = torch.device("cpu")

    model_config_path = resolve_path(args.model_config, root)
    config = load_json(model_config_path)
    cache_paths = resolve_cache_paths(config, root)
    cache_payloads = {split: validate_source_cache(cache_paths[split], split) for split in SPLITS}
    datasets = {split: ClipCacheDataset(cache_payloads[split]) for split in SPLITS}
    overlaps = validate_recording_level_no_leakage(datasets)
    class_names = class_names_from_config_or_cache(config, cache_payloads)
    num_classes = infer_num_classes(datasets, config)
    input_shape = list(datasets["train"].features.shape[1:])

    if args.pretrained != "none":
        raise ValueError("ImageNet pretraining is reserved for a future task; use --pretrained none.")
    model, embed_dim = build_clip_frontend(args.frontend, num_classes, pretrained=args.pretrained)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    param_summary = {**count_params(model), "frontend": args.frontend, "embed_dim": int(embed_dim)}
    write_json(output_dir / "params_summary.json", param_summary)

    pin_memory = device.type == "cuda"
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory),
        "val": DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory),
        "test": DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory),
    }

    best_path = output_dir / "best.pt"
    best_val_recording_macro_f1 = -1.0
    best_val_clip_macro_f1 = -1.0
    best_epoch = -1
    stale = 0
    epoch_rows = []

    print(f"frontend: {args.frontend}", flush=True)
    print(f"num_classes: {num_classes}", flush=True)
    print(f"input_shape: {input_shape}", flush=True)
    print(f"recordings train/val/test: {len(set(datasets['train'].recording_ids))}/{len(set(datasets['val'].recording_ids))}/{len(set(datasets['test'].recording_ids))}", flush=True)
    print(f"clips train/val/test: {len(datasets['train'])}/{len(datasets['val'])}/{len(datasets['test'])}", flush=True)

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, loaders["train"], criterion, optimizer, device, args.grad_clip)
        val_eval = evaluate(model, loaders["val"], criterion, device, collect_predictions=False)
        val_clip = val_eval["clip_metrics"]
        val_recording = val_eval["recording_metrics"]
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_clip_acc": train_metrics["acc"],
            "train_clip_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_clip["loss"],
            "val_clip_acc": val_clip["acc"],
            "val_clip_macro_f1": val_clip["macro_f1"],
            "val_recording_acc": val_recording["acc"],
            "val_recording_macro_f1": val_recording["macro_f1"],
            "val_recording_macro_precision": val_recording["macro_precision"],
            "val_recording_macro_recall": val_recording["macro_recall"],
        }
        epoch_rows.append(row)
        print(
            f"{epoch},{row['train_loss']:.6f},{row['train_clip_acc']:.6f},{row['train_clip_macro_f1']:.6f},"
            f"{row['val_loss']:.6f},{row['val_clip_acc']:.6f},{row['val_clip_macro_f1']:.6f},"
            f"{row['val_recording_acc']:.6f},{row['val_recording_macro_f1']:.6f}",
            flush=True,
        )
        if math.isfinite(val_recording["macro_f1"]) and val_recording["macro_f1"] > best_val_recording_macro_f1:
            best_val_recording_macro_f1 = val_recording["macro_f1"]
            best_val_clip_macro_f1 = val_clip["macro_f1"]
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "frontend": args.frontend,
                    "model_state": model.state_dict(),
                    "num_classes": num_classes,
                    "embed_dim": int(embed_dim),
                    "input_shape": input_shape,
                    "best_epoch": best_epoch,
                    "best_val_recording_macro_f1": best_val_recording_macro_f1,
                    "best_val_clip_macro_f1": best_val_clip_macro_f1,
                    "model_config": str(model_config_path),
                    "cache_paths": {split: str(path) for split, path in cache_paths.items()},
                    "class_names": class_names,
                    "args": vars(args),
                },
                best_path,
            )
        else:
            stale += 1
            if args.patience > 0 and stale >= args.patience:
                print(f"Early stopping at epoch {epoch}", flush=True)
                break

    write_epoch_csv(output_dir / "epoch_metrics.csv", epoch_rows)
    checkpoint = torch_load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_eval = evaluate(model, loaders["test"], criterion, device, collect_predictions=True)
    test_clip = test_eval["clip_metrics"]
    test_recording = test_eval["recording_metrics"]

    metrics_payload = {
        "test_clip_acc": test_clip["acc"],
        "test_clip_macro_f1": test_clip["macro_f1"],
        "test_clip_macro_precision": test_clip["macro_precision"],
        "test_clip_macro_recall": test_clip["macro_recall"],
        "test_recording_acc": test_recording["acc"],
        "test_recording_macro_f1": test_recording["macro_f1"],
        "test_recording_macro_precision": test_recording["macro_precision"],
        "test_recording_macro_recall": test_recording["macro_recall"],
    }
    write_json(output_dir / "metrics.json", metrics_payload)
    write_metrics_txt(output_dir / "metrics.txt", metrics_payload)
    save_clip_predictions(output_dir / "test_predictions_clip_level.csv", test_eval, num_classes)
    save_recording_predictions(output_dir / "test_predictions_recording_level.csv", test_eval["recording_payload"], num_classes)

    run_config = {
        "frontend": args.frontend,
        "dataset": args.dataset,
        "seed": args.seed,
        "num_classes": num_classes,
        "embed_dim": int(embed_dim),
        "input_shape": input_shape,
        "source_model_config": str(model_config_path),
        "cache_paths": {split: str(path) for split, path in cache_paths.items()},
        "recording_overlap": overlaps,
        "class_names": class_names,
        "best_epoch": best_epoch,
        "best_val_recording_macro_f1": best_val_recording_macro_f1,
        "best_val_clip_macro_f1": best_val_clip_macro_f1,
        "test_recording_metrics": test_recording,
        "test_clip_metrics": test_clip,
        **param_summary,
    }
    write_json(output_dir / "model_config.json", run_config)
    print(json.dumps({"metrics": metrics_payload, "best_epoch": best_epoch}, indent=2), flush=True)


if __name__ == "__main__":
    main()
