#!/usr/bin/env python3
"""Fine-tune official PANNs Cnn14_16k on DeepShip waveform clips.

This script trains only clip-level PANNs front-end parameters plus a DeepShip
classifier. It reads raw waveform segments through strict recording-level
ShuffleFAC cache entries and selects checkpoints by validation recording-level
mean-probability Macro-F1.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset

from run_frontend_robustness import (
    OfficialPannsCnn14_16kFrontend,
    install_run_log,
    load_model_config,
    load_waveform_segment_from_entry,
    resolve_cache_paths,
    resolve_panns_repo_and_ckpt,
    resolve_path,
    set_seed,
    torch_load,
    validate_source_cache,
    write_json,
)


SPLITS = ["train", "val", "test"]
FINETUNE_ADAPTATION = {
    "classifier_only": "classifier_only_linear_probe",
    "fc1_only": "fc1_only_domain_adaptation",
    "block6_fc1": "block6_fc1_domain_adaptation",
}
FINETUNE_PRETRAINING = {
    "classifier_only": "AudioSet",
    "fc1_only": "AudioSet+DeepShip",
    "block6_fc1": "AudioSet+DeepShip",
}


class WaveformClipCacheDataset(Dataset):
    """Read waveform clips referenced by a strict recording-level cache."""

    def __init__(
        self,
        cache_payload: dict,
        source_cache_path: Path,
        root: Path,
        target_sample_rate: int,
        segment_length: float,
        max_clips: int = 0,
    ):
        self.entries = list(cache_payload.get("entries", []))
        self.labels = cache_payload["labels"].long()
        self.source_cache_path = Path(source_cache_path)
        self.root = Path(root)
        self.target_sample_rate = int(target_sample_rate)
        self.segment_length = float(segment_length)
        if len(self.entries) != int(self.labels.numel()):
            raise ValueError(
                f"entries/labels length mismatch for {source_cache_path}: "
                f"entries={len(self.entries)} labels={int(self.labels.numel())}"
            )
        if int(max_clips) > 0:
            keep = int(max_clips)
            self.entries = self.entries[:keep]
            self.labels = self.labels[:keep]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        waveform = load_waveform_segment_from_entry(
            entry,
            source_cache_path=self.source_cache_path,
            root=self.root,
            target_sample_rate=self.target_sample_rate,
            segment_length=self.segment_length,
        )
        label = int(self.labels[index])
        recording_id = str(entry.get("recording_id") or Path(entry.get("path", str(index))).stem)
        segment_index = int(entry.get("segment_index", index))
        return waveform, label, recording_id, segment_index

    def recording_ids(self):
        return {str(entry.get("recording_id") or Path(entry.get("path", "")).stem) for entry in self.entries}


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune official PANNs Cnn14_16k on waveform clips.")
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--dataset", default="DeepShip")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--panns_repo_dir", default=None)
    parser.add_argument("--panns_ckpt", default=None)
    parser.add_argument("--panns_sample_rate", type=int, default=16000)
    parser.add_argument("--segment_length", type=float, default=3.0)
    parser.add_argument("--finetune_mode", choices=["classifier_only", "fc1_only", "block6_fc1"], default="fc1_only")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr_classifier", type=float, default=1e-3)
    parser.add_argument("--lr_fc1", type=float, default=1e-4)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--smoke_max_clips_per_split", type=int, default=0)
    return parser.parse_args()


def cache_recording_ids(payload: dict):
    ids = set()
    for index, entry in enumerate(payload.get("entries", [])):
        ids.add(str(entry.get("recording_id") or Path(entry.get("path", str(index))).stem))
    return ids


def validate_recording_overlap(cache_payloads: dict):
    sets = {split: cache_recording_ids(payload) for split, payload in cache_payloads.items()}
    overlaps = {}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = sorted(sets[a].intersection(sets[b]))
        overlaps[f"{a}-{b}"] = overlap
        if overlap:
            raise ValueError(f"Recording-level leakage detected for {a}-{b}: {overlap[:10]}")
    return overlaps


def infer_class_names(config: dict, num_classes: int):
    split_meta = config.get("split_metadata", {}) if isinstance(config.get("split_metadata"), dict) else {}
    mapping = split_meta.get("class_mapping")
    if not isinstance(mapping, dict) or not mapping:
        return None
    class_names = [None] * int(num_classes)
    for name, idx in mapping.items():
        idx = int(idx)
        if 0 <= idx < len(class_names):
            class_names[idx] = str(name)
    return class_names if any(item is not None for item in class_names) else None


def metrics_from_arrays(y_true, y_pred):
    if len(y_true) == 0:
        return {"acc": math.nan, "macro_f1": math.nan, "macro_precision": math.nan, "macro_recall": math.nan}
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def recording_metrics_from_clips(clip_records):
    grouped = defaultdict(lambda: {"probs": [], "labels": [], "segments": []})
    for record in clip_records:
        group = grouped[record["recording_id"]]
        group["probs"].append(record["probs"])
        group["labels"].append(int(record["label"]))
        group["segments"].append(int(record["segment_index"]))

    y_true = []
    y_pred = []
    recording_records = []
    for recording_id, group in sorted(grouped.items()):
        label_set = sorted(set(group["labels"]))
        if len(label_set) != 1:
            raise ValueError(f"Recording {recording_id} has inconsistent labels: {label_set}")
        probs = np.stack(group["probs"], axis=0).mean(axis=0)
        pred = int(np.argmax(probs))
        label = int(label_set[0])
        y_true.append(label)
        y_pred.append(pred)
        recording_records.append(
            {
                "recording_id": recording_id,
                "label": label,
                "pred": pred,
                "probs": probs,
                "num_clips": len(group["probs"]),
                "segment_indices": sorted(group["segments"]),
            }
        )
    metrics = metrics_from_arrays(y_true, y_pred)
    return metrics, recording_records


class PannsDeepShipClassifier(nn.Module):
    def __init__(self, panns_model: nn.Module, num_classes: int):
        super().__init__()
        self.panns_model = panns_model
        self.classifier = nn.Linear(2048, int(num_classes))

    def forward(self, waveform):
        output = self.panns_model(waveform)
        embedding = output["embedding"]
        return self.classifier(embedding)


def configure_trainable(model: PannsDeepShipClassifier, finetune_mode: str):
    if finetune_mode not in FINETUNE_ADAPTATION:
        raise ValueError(f"Unsupported finetune_mode: {finetune_mode}")

    for param in model.panns_model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    if finetune_mode == "classifier_only":
        return

    if not hasattr(model.panns_model, "fc1"):
        raise AttributeError("Official Cnn14_16k model has no fc1 layer")
    for param in model.panns_model.fc1.parameters():
        param.requires_grad = True

    if finetune_mode == "fc1_only":
        return

    if finetune_mode != "block6_fc1":
        raise ValueError(f"Unsupported finetune_mode: {finetune_mode}")
    if not hasattr(model.panns_model, "conv_block6"):
        raise AttributeError("Official Cnn14_16k model has no conv_block6 layer")
    for param in model.panns_model.conv_block6.parameters():
        param.requires_grad = True


def set_panns_training_mode(panns_model, classifier, finetune_mode):
    if finetune_mode not in FINETUNE_ADAPTATION:
        raise ValueError(f"Unsupported finetune_mode: {finetune_mode}")

    panns_model.eval()
    classifier.train()
    if finetune_mode == "classifier_only":
        return
    if finetune_mode == "fc1_only":
        if not hasattr(panns_model, "fc1"):
            raise AttributeError("Official Cnn14_16k model has no fc1 layer")
        panns_model.fc1.train()
        return
    if not hasattr(panns_model, "conv_block6"):
        raise AttributeError("Official Cnn14_16k model has no conv_block6 layer")
    if not hasattr(panns_model, "fc1"):
        raise AttributeError("Official Cnn14_16k model has no fc1 layer")
    panns_model.conv_block6.train()
    panns_model.fc1.train()


def early_conv_blocks_eval(panns_model):
    block_names = ["conv_block1", "conv_block2", "conv_block3", "conv_block4", "conv_block5"]
    return all(not getattr(panns_model, name).training for name in block_names if hasattr(panns_model, name))


def print_training_mode_status(model, finetune_mode, prefix="training mode"):
    print(
        f"{prefix}: finetune_mode={finetune_mode} "
        f"fc1.training={model.panns_model.fc1.training} "
        f"classifier.training={model.classifier.training} "
        f"early_conv_blocks_eval={early_conv_blocks_eval(model.panns_model)}",
        flush=True,
    )


def build_optimizer(model: PannsDeepShipClassifier, args):
    if args.finetune_mode not in FINETUNE_ADAPTATION:
        raise ValueError(f"Unsupported finetune_mode: {args.finetune_mode}")

    param_groups = []
    classifier_params = [p for p in model.classifier.parameters() if p.requires_grad]
    if classifier_params:
        param_groups.append({"params": classifier_params, "lr": args.lr_classifier, "name": "classifier"})
    if args.finetune_mode in ("fc1_only", "block6_fc1"):
        fc1_params = [p for p in model.panns_model.fc1.parameters() if p.requires_grad]
        if fc1_params:
            param_groups.append({"params": fc1_params, "lr": args.lr_fc1, "name": "fc1"})
    if args.finetune_mode == "block6_fc1":
        block6_params = [p for p in model.panns_model.conv_block6.parameters() if p.requires_grad]
        if block6_params:
            param_groups.append({"params": block6_params, "lr": args.lr_backbone, "name": "conv_block6"})
    if not param_groups:
        raise ValueError("No trainable parameters were selected for the optimizer.")
    return torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    grad_clip=0.0,
    collect_predictions=False,
    finetune_mode="fc1_only",
    log_mode_status=False,
):
    train = optimizer is not None
    if train:
        set_panns_training_mode(model.panns_model, model.classifier, finetune_mode)
        if log_mode_status:
            print_training_mode_status(model, finetune_mode)
    else:
        model.panns_model.eval()
        model.classifier.eval()
    total_loss = 0.0
    total_n = 0
    y_true = []
    y_pred = []
    clip_records = []

    for waveform, labels, recording_ids, segment_indices in loader:
        waveform = waveform.float().to(device, non_blocking=True)
        labels = labels.long().to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            logits = model(waveform)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
                optimizer.step()

        batch = int(labels.size(0))
        total_loss += float(loss.detach().cpu()) * batch
        total_n += batch
        probs = torch.softmax(logits.detach(), dim=-1).cpu().numpy()
        preds = probs.argmax(axis=1)
        labels_np = labels.detach().cpu().numpy()
        y_true.extend(labels_np.tolist())
        y_pred.extend(preds.tolist())
        if collect_predictions or not train:
            for i in range(batch):
                clip_records.append(
                    {
                        "recording_id": str(recording_ids[i]),
                        "segment_index": int(segment_indices[i]),
                        "label": int(labels_np[i]),
                        "pred": int(preds[i]),
                        "probs": probs[i],
                    }
                )

    clip_metrics = metrics_from_arrays(y_true, y_pred)
    clip_metrics["loss"] = total_loss / max(total_n, 1)
    recording_metrics, recording_records = recording_metrics_from_clips(clip_records) if clip_records else ({}, [])
    return clip_metrics, recording_metrics, clip_records, recording_records


def save_epoch_metrics(path: Path, rows: list):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_clip_predictions(path: Path, clip_records, class_names, num_classes):
    fieldnames = ["recording_id", "segment_index", "label", "label_name", "pred", "pred_name"] + [
        f"prob_{i}" for i in range(num_classes)
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in clip_records:
            row = {
                "recording_id": record["recording_id"],
                "segment_index": record["segment_index"],
                "label": record["label"],
                "label_name": class_names[record["label"]] if class_names else "",
                "pred": record["pred"],
                "pred_name": class_names[record["pred"]] if class_names else "",
            }
            for i, value in enumerate(record["probs"]):
                row[f"prob_{i}"] = float(value)
            writer.writerow(row)


def write_recording_predictions(path: Path, recording_records, class_names, num_classes):
    fieldnames = ["recording_id", "num_clips", "segment_indices", "label", "label_name", "pred", "pred_name"] + [
        f"prob_{i}" for i in range(num_classes)
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in recording_records:
            row = {
                "recording_id": record["recording_id"],
                "num_clips": record["num_clips"],
                "segment_indices": " ".join(str(x) for x in record["segment_indices"]),
                "label": record["label"],
                "label_name": class_names[record["label"]] if class_names else "",
                "pred": record["pred"],
                "pred_name": class_names[record["pred"]] if class_names else "",
            }
            for i, value in enumerate(record["probs"]):
                row[f"prob_{i}"] = float(value)
            writer.writerow(row)


def write_metrics_txt(path: Path, metrics: dict):
    lines = [f"{key}: {value}" for key, value in sorted(metrics.items())]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_loaders(cache_payloads, cache_paths, root, args):
    datasets = {
        split: WaveformClipCacheDataset(
            cache_payloads[split],
            cache_paths[split],
            root,
            target_sample_rate=args.panns_sample_rate,
            segment_length=args.segment_length,
            max_clips=args.smoke_max_clips_per_split,
        )
        for split in SPLITS
    }
    pin_memory = str(args.device).startswith("cuda") and torch.cuda.is_available()
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        ),
    }
    return datasets, loaders


def main():
    args = parse_args()
    root = Path.cwd()
    output_dir = resolve_path(args.output_dir, root)
    output_dir.mkdir(parents=True, exist_ok=True)
    install_run_log(output_dir)
    set_seed(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    model_config_path = resolve_path(args.model_config, root)
    config = load_model_config(model_config_path)
    cache_paths = resolve_cache_paths(config, root)
    cache_payloads = {split: validate_source_cache(cache_paths[split], split) for split in SPLITS}
    recording_overlap = validate_recording_overlap(cache_payloads)
    datasets, loaders = make_loaders(cache_payloads, cache_paths, root, args)

    all_labels = torch.cat([datasets[split].labels for split in SPLITS])
    num_classes = int(all_labels.max().item()) + 1
    class_names = infer_class_names(config, num_classes)

    repo_dir, ckpt_path = resolve_panns_repo_and_ckpt(args, root)
    wrapper = OfficialPannsCnn14_16kFrontend(
        repo_dir=repo_dir,
        ckpt_path=ckpt_path,
        device=device,
        sample_rate=args.panns_sample_rate,
    )
    model = PannsDeepShipClassifier(wrapper.model, num_classes=num_classes).to(device)
    configure_trainable(model, args.finetune_mode)
    adaptation = FINETUNE_ADAPTATION[args.finetune_mode]
    pretraining = FINETUNE_PRETRAINING[args.finetune_mode]

    trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]
    print(f"finetune_mode: {args.finetune_mode}", flush=True)
    print("trainable parameters:", flush=True)
    for name in trainable_names:
        print(f"  {name}", flush=True)
    (output_dir / "trainable_params.txt").write_text("\n".join(trainable_names) + "\n", encoding="utf-8")

    params_summary = {
        "total_params": int(sum(p.numel() for p in model.parameters())),
        "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "panns_total_params": int(sum(p.numel() for p in model.panns_model.parameters())),
        "panns_trainable_params": int(sum(p.numel() for p in model.panns_model.parameters() if p.requires_grad)),
        "classifier_params": int(sum(p.numel() for p in model.classifier.parameters())),
        "trainable_parameter_names": trainable_names,
    }
    write_json(output_dir / "params_summary.json", params_summary)

    run_config = {
        "script": "train_panns_frontend.py",
        "dataset": args.dataset,
        "model_config": str(model_config_path),
        "cache_paths": {k: str(v) for k, v in cache_paths.items()},
        "panns_repo_dir": str(repo_dir),
        "panns_ckpt": str(ckpt_path),
        "num_classes": num_classes,
        "class_names": class_names,
        "finetune_mode": args.finetune_mode,
        "adaptation": adaptation,
        "pretraining": pretraining,
        "recording_overlap": recording_overlap,
        "smoke_max_clips_per_split": int(args.smoke_max_clips_per_split),
        "args": vars(args),
    }
    write_json(output_dir / "model_config.json", run_config)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, args)
    best_path = output_dir / "best.pt"
    best_val_recording_macro_f1 = -1.0
    best_val_clip_macro_f1 = -1.0
    best_epoch = -1
    stale = 0
    epoch_rows = []

    for epoch in range(1, args.epochs + 1):
        train_clip, _train_recording, _train_clips, _train_recs = run_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            optimizer=optimizer,
            grad_clip=args.grad_clip,
            collect_predictions=False,
            finetune_mode=args.finetune_mode,
            log_mode_status=True,
        )
        val_clip, val_recording, _val_clips, _val_recs = run_epoch(
            model,
            loaders["val"],
            criterion,
            device,
            optimizer=None,
            collect_predictions=True,
            finetune_mode=args.finetune_mode,
        )
        row = {
            "epoch": epoch,
            "train_clip_loss": train_clip["loss"],
            "train_clip_acc": train_clip["acc"],
            "train_clip_macro_f1": train_clip["macro_f1"],
            "val_clip_loss": val_clip["loss"],
            "val_clip_acc": val_clip["acc"],
            "val_clip_macro_f1": val_clip["macro_f1"],
            "val_recording_acc": val_recording["acc"],
            "val_recording_macro_f1": val_recording["macro_f1"],
            "val_recording_macro_precision": val_recording["macro_precision"],
            "val_recording_macro_recall": val_recording["macro_recall"],
        }
        epoch_rows.append(row)
        print(
            f"epoch={epoch} train_loss={row['train_clip_loss']:.6f} "
            f"val_clip_f1={row['val_clip_macro_f1']:.6f} "
            f"val_recording_f1={row['val_recording_macro_f1']:.6f}",
            flush=True,
        )

        score = val_recording["macro_f1"]
        if math.isfinite(score) and score > best_val_recording_macro_f1:
            best_val_recording_macro_f1 = score
            best_val_clip_macro_f1 = val_clip["macro_f1"]
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "frontend": "panns_cnn14_finetuned",
                    "base_frontend": "panns_cnn14",
                    "pretraining": pretraining,
                    "finetune_mode": args.finetune_mode,
                    "adaptation": adaptation,
                    "model_state": model.panns_model.state_dict(),
                    "classifier_state": model.classifier.state_dict(),
                    "num_classes": num_classes,
                    "embed_dim": 2048,
                    "sample_rate": int(args.panns_sample_rate),
                    "segment_length": float(args.segment_length),
                    "base_panns_ckpt": str(ckpt_path),
                    "best_epoch": best_epoch,
                    "best_val_recording_macro_f1": best_val_recording_macro_f1,
                    "best_val_clip_macro_f1": best_val_clip_macro_f1,
                    "model_config": str(model_config_path),
                    "cache_paths": {k: str(v) for k, v in cache_paths.items()},
                    "class_names": class_names,
                    "args": vars(args),
                },
                best_path,
            )
        else:
            stale += 1
            if args.patience > 0 and stale >= args.patience:
                print(f"early stopping at epoch {epoch}", flush=True)
                break

    save_epoch_metrics(output_dir / "epoch_metrics.csv", epoch_rows)

    if not best_path.exists():
        raise RuntimeError("No best checkpoint was saved")
    best = torch_load(best_path, map_location=device)
    model.panns_model.load_state_dict(best["model_state"], strict=True)
    model.classifier.load_state_dict(best["classifier_state"], strict=True)

    test_clip, test_recording, test_clip_records, test_recording_records = run_epoch(
        model,
        loaders["test"],
        criterion,
        device,
        optimizer=None,
        collect_predictions=True,
        finetune_mode=args.finetune_mode,
    )

    write_clip_predictions(output_dir / "test_predictions_clip_level.csv", test_clip_records, class_names, num_classes)
    write_recording_predictions(
        output_dir / "test_predictions_recording_level.csv", test_recording_records, class_names, num_classes
    )

    metrics = {
        "test_clip_acc": test_clip["acc"],
        "test_clip_macro_f1": test_clip["macro_f1"],
        "test_clip_macro_precision": test_clip["macro_precision"],
        "test_clip_macro_recall": test_clip["macro_recall"],
        "test_recording_acc": test_recording["acc"],
        "test_recording_macro_f1": test_recording["macro_f1"],
        "test_recording_macro_precision": test_recording["macro_precision"],
        "test_recording_macro_recall": test_recording["macro_recall"],
        "best_epoch": best_epoch,
        "best_val_recording_macro_f1": best_val_recording_macro_f1,
        "best_val_clip_macro_f1": best_val_clip_macro_f1,
        "finetune_mode": args.finetune_mode,
        "adaptation": adaptation,
        "pretraining": pretraining,
        "smoke_max_clips_per_split": int(args.smoke_max_clips_per_split),
    }
    write_json(output_dir / "metrics.json", metrics)
    write_metrics_txt(output_dir / "metrics.txt", metrics)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
