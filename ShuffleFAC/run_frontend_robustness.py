#!/usr/bin/env python3
"""Cross-front-end recording-level robustness experiments.

This script is intentionally additive: it does not change the existing
ShuffleFAC training/evaluation entry points. It reuses strict recording-level
clip caches produced by run_deepship.py, converts each clip to a front-end
embedding, stores recording bags as [R, S, D], and trains a unified set of
recording-level heads on those cached embeddings.
"""

import argparse
import atexit
import csv
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset

from model.shuffleFAC import shuffleFAC


FRONTENDS = ["shufflefac", "resnet18", "mobilenet_v2", "panns_cnn14"]
HEADS = ["mean", "attention", "sn_decoupled", "sn_expd_warmup5", "bigru", "mil_linear_softmax"]
SPLITS = ["train", "val", "test"]
TORCHVISION_TRAINED_FRONTENDS = {"resnet18", "mobilenet_v2"}
EXP_D_EDGE_MODE = "threshold_similarity"
EXP_D_SIM_THRESHOLD = 0.8
EXP_D_SIGNAL_TOP_K = 4
EXP_D_TOPK_WARMUP_EPOCHS = 5
EMBEDDING_CACHE_SCHEMA = "recording_all_clip_embeddings_v2"


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
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


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


def checkpoint_state(checkpoint):
    if isinstance(checkpoint, dict):
        state = checkpoint.get("model_state")
        if state is None:
            state = checkpoint.get("state_dict")
        if state is None:
            state = checkpoint
    else:
        state = checkpoint
    if not isinstance(state, dict):
        raise ValueError("Checkpoint does not contain a state dict")
    return {str(k).removeprefix("module."): v for k, v in state.items()}


def load_state_flexible(model: nn.Module, ckpt_path: Optional[Path], output_dir: Path, prefix: str):
    """Load optional checkpoints without making random-init front-ends unusable."""

    warning_path = output_dir / f"{prefix}_load_warnings.txt"
    if ckpt_path is None:
        warning_path.write_text("No checkpoint provided; using current initialization.\n", encoding="utf-8")
        return None
    checkpoint = torch_load(ckpt_path, map_location="cpu")
    state = checkpoint_state(checkpoint)
    try:
        model.load_state_dict(state, strict=True)
        warning_path.write_text("strict=True load succeeded.\n", encoding="utf-8")
    except RuntimeError as exc:
        result = model.load_state_dict(state, strict=False)
        warning_path.write_text(
            "strict=True load failed; strict=False fallback was used.\n\n"
            f"strict_error:\n{exc}\n\n"
            f"missing_keys:\n{list(result.missing_keys)}\n\n"
            f"unexpected_keys:\n{list(result.unexpected_keys)}\n",
            encoding="utf-8",
        )
    return checkpoint


def load_model_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def infer_shufflefac_ckpt(config: dict, model_config_path: Path, root: Path, explicit: Optional[str]) -> Optional[Path]:
    if explicit:
        return resolve_path(explicit, root)
    for key in ["Encoder checkpoint", "encoder_ckpt"]:
        if config.get(key):
            return resolve_path(config[key], root)
    training = config.get("training", {}) if isinstance(config.get("training"), dict) else {}
    if training.get("ckpt_path"):
        return resolve_path(training["ckpt_path"], root)
    candidate = model_config_path.parent / "best.pt"
    return candidate if candidate.exists() else None


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
    return payload



def resolve_audio_path_from_entry(entry, source_cache_path: Path, root: Path) -> Path:
    """Resolve the raw audio path referenced by a strict recording-level cache entry."""

    candidate_paths = []
    path_fields = ["wav_path", "audio_path", "path", "file_path"]
    for field in path_fields:
        raw_value = entry.get(field)
        if raw_value is None or str(raw_value).strip() == "":
            continue
        raw_path = Path(str(raw_value)).expanduser()
        if raw_path.is_absolute():
            candidate_paths.append(raw_path)
            if raw_path.exists():
                return raw_path
            continue
        for base in [root, source_cache_path.parent, source_cache_path.parent.parent]:
            candidate = (base / raw_path).expanduser()
            candidate_paths.append(candidate)
            if candidate.exists():
                return candidate

    raise FileNotFoundError(
        "Could not resolve audio path from cache entry. "
        f"entry.keys()={list(entry.keys())}; "
        f"source_cache_path={source_cache_path}; "
        f"candidate_paths={[str(path) for path in candidate_paths]}"
    )


def infer_segment_start_info(entry, segment_length: float):
    """Infer segment start from seconds, samples, or segment_index fallback."""

    for key in ["start_sec", "offset_sec", "start_time", "start"]:
        value = entry.get(key)
        if value is not None and str(value).strip() != "":
            return "sec", float(value)
    for key in ["start_sample", "offset_sample"]:
        value = entry.get(key)
        if value is not None and str(value).strip() != "":
            return "sample", int(value)
    segment_index = int(entry.get("segment_index", 0))
    return "sec", float(segment_index) * float(segment_length)


def load_waveform_segment_from_entry(
    entry,
    source_cache_path: Path,
    root: Path,
    target_sample_rate: int = 16000,
    segment_length: float = 3.0,
):
    """Load a fixed-length mono waveform segment referenced by a cache entry."""

    audio_path = resolve_audio_path_from_entry(entry, source_cache_path, root)
    waveform, sr = torchaudio.load(audio_path)
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0)
    elif waveform.ndim != 1:
        raise ValueError(f"Expected waveform shape [C, T] or [T], got {tuple(waveform.shape)} from {audio_path}")

    start_kind, start_value = infer_segment_start_info(entry, segment_length)
    if start_kind == "sec":
        start_sample = int(round(float(start_value) * int(sr)))
    elif start_kind == "sample":
        start_sample = int(start_value)
    else:
        raise ValueError(f"Unsupported segment start kind: {start_kind!r}")
    start_sample = max(start_sample, 0)

    num_samples = int(round(float(segment_length) * int(sr)))
    segment = waveform[start_sample : start_sample + num_samples]
    if segment.numel() < num_samples:
        segment = F.pad(segment, (0, num_samples - int(segment.numel())))
    elif segment.numel() > num_samples:
        segment = segment[:num_samples]

    target_sample_rate = int(target_sample_rate)
    if int(sr) != target_sample_rate:
        segment = torchaudio.functional.resample(segment, int(sr), target_sample_rate)

    target_samples = int(round(float(segment_length) * target_sample_rate))
    if segment.numel() < target_samples:
        segment = F.pad(segment, (0, target_samples - int(segment.numel())))
    elif segment.numel() > target_samples:
        segment = segment[:target_samples]
    return segment.float()


def debug_first_waveform_segment(
    model_config_path: Path,
    split_name: str,
    root: Path,
    target_sample_rate: int,
    segment_length: float,
):
    """Print basic diagnostics for the first waveform segment in a source cache."""

    config = load_model_config(model_config_path)
    cache_paths = resolve_cache_paths(config, root)
    if split_name not in cache_paths:
        raise ValueError(f"Unknown split {split_name!r}; available={sorted(cache_paths)}")
    source_cache_path = cache_paths[split_name]
    payload = validate_source_cache(source_cache_path, split_name)
    entries = payload.get("entries", [])
    if not entries:
        raise ValueError(f"{source_cache_path} has no entries")
    entry = entries[0]
    audio_path = resolve_audio_path_from_entry(entry, source_cache_path, root)
    segment = load_waveform_segment_from_entry(
        entry,
        source_cache_path,
        root,
        target_sample_rate=target_sample_rate,
        segment_length=segment_length,
    )
    print(f"resolved audio path: {audio_path}", flush=True)
    print(f"entry keys: {list(entry.keys())}", flush=True)
    print(f"segment_index: {entry.get('segment_index')}", flush=True)
    print(f"output shape: {tuple(segment.shape)}", flush=True)
    print(f"output dtype: {segment.dtype}", flush=True)
    print(
        f"min/max/mean: {float(segment.min()):.8f} / {float(segment.max()):.8f} / {float(segment.mean()):.8f}",
        flush=True,
    )


def resolve_panns_repo_and_ckpt(args, root: Path):
    """Resolve the official PANNs repo and Cnn14_16k checkpoint paths."""

    if args.panns_repo_dir:
        repo_dir = resolve_path(args.panns_repo_dir, root)
    else:
        repo_dir = Path(__file__).resolve().parents[1] / "audioset_tagging_cnn-master"
    if not repo_dir.exists():
        raise FileNotFoundError(f"PANNs repo_dir does not exist: {repo_dir}")

    if args.panns_ckpt:
        ckpt_path = resolve_path(args.panns_ckpt, root)
    else:
        direct = repo_dir / "Cnn14_16k_mAP=0.438.pth"
        if direct.exists():
            ckpt_path = direct
        else:
            matches = sorted(repo_dir.glob("Cnn14_16k_mAP=0.438*"))
            ckpt_path = matches[0] if matches else direct
    if not ckpt_path.exists():
        raise FileNotFoundError(f"PANNs checkpoint does not exist: {ckpt_path}")
    return repo_dir, ckpt_path

def infer_num_classes_from_config(config: dict):
    """Infer class count for random smoke-test front-ends only."""

    for key in ["cnn_cfg_used", "CNN"]:
        cfg = config.get(key)
        if isinstance(cfg, dict) and cfg.get("n_class") is not None:
            return int(cfg["n_class"])
    split_meta = config.get("split_metadata", {})
    mapping = split_meta.get("class_mapping")
    if isinstance(mapping, dict) and mapping:
        return max(int(v) for v in mapping.values()) + 1
    raise ValueError("Could not infer num_classes from model_config")


def checkpoint_cache_metadata(ckpt_path: Optional[str], checkpoint: Optional[dict]):
    """Return checkpoint fields that should invalidate stale embedding caches."""

    if not ckpt_path:
        return None
    path = Path(ckpt_path).expanduser().resolve()
    stat = path.stat()
    out = {
        "resolved_path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime": float(stat.st_mtime),
    }
    if isinstance(checkpoint, dict):
        out["best_epoch"] = checkpoint.get("best_epoch", checkpoint.get("epoch"))
        out["best_val_recording_macro_f1"] = checkpoint.get("best_val_recording_macro_f1")
    return out


class ShuffleFACFrontend(nn.Module):
    """Use a trained ShuffleFAC classifier as a frozen embedding extractor."""

    def __init__(self, ckpt_path: Path, device: torch.device):
        super().__init__()
        checkpoint = torch_load(ckpt_path, map_location=device)
        if not isinstance(checkpoint, dict) or "cnn_cfg" not in checkpoint:
            raise ValueError("ShuffleFAC front-end requires a checkpoint with cnn_cfg and model_state")
        self.checkpoint = checkpoint
        self.cnn_cfg = checkpoint["cnn_cfg"]
        self.model = shuffleFAC(**self.cnn_cfg).to(device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.embed_dim = int(self.cnn_cfg["nb_filters"][-1])
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, x):
        x = x.transpose(2, 3)
        return self.model.cnn(x).view(x.size(0), -1)


class TorchVisionTrainedFrontend(nn.Module):
    """Wrap a train_clip_frontend.py checkpoint and expose forward_features()."""

    def __init__(self, ckpt_path: Path, device: torch.device, expected_frontend: str):
        super().__init__()
        from model.frontends_torchvision import build_clip_frontend

        checkpoint = torch_load(ckpt_path, map_location=device)
        if not isinstance(checkpoint, dict):
            raise ValueError(f"TorchVision front-end checkpoint must be a dict: {ckpt_path}")
        frontend = checkpoint.get("frontend")
        if frontend != expected_frontend:
            raise ValueError(
                f"frontend checkpoint mismatch: CLI requested {expected_frontend!r}, "
                f"checkpoint contains {frontend!r}"
            )
        num_classes = int(checkpoint["num_classes"])
        model, model_embed_dim = build_clip_frontend(frontend, num_classes, pretrained="none")
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.to(device).eval()
        for param in model.parameters():
            param.requires_grad = False

        self.model = model
        self.embed_dim = int(checkpoint.get("embed_dim", model_embed_dim))
        if self.embed_dim != int(model_embed_dim):
            raise ValueError(
                f"embed_dim mismatch for {ckpt_path}: checkpoint={self.embed_dim}, model={model_embed_dim}"
            )
        self.frontend = frontend
        self.input_shape = checkpoint.get("input_shape")
        self.checkpoint = checkpoint
        self.frontend_source = "torchvision_trained"
        self.pretraining = "none"
        self.weights = "trained_on_train_recordings"

    @torch.no_grad()
    def forward(self, x):
        return self.model.forward_features(x)


class TorchVisionRandomFrontend(nn.Module):
    """Random front-end wrapper allowed only for explicit smoke tests."""

    def __init__(self, frontend: str, num_classes: int, device: torch.device):
        super().__init__()
        from model.frontends_torchvision import build_clip_frontend

        model, embed_dim = build_clip_frontend(frontend, num_classes, pretrained="none")
        model.to(device).eval()
        for param in model.parameters():
            param.requires_grad = False
        self.model = model
        self.frontend = frontend
        self.embed_dim = int(embed_dim)
        self.input_shape = None
        self.checkpoint = None
        self.frontend_source = "torchvision_random"
        self.pretraining = "random_smoke_test"
        self.weights = "random_smoke_test"

    @torch.no_grad()
    def forward(self, x):
        return self.model.forward_features(x)


class OfficialPannsCnn14_16kFrontend(nn.Module):
    """Official PANNs Cnn14_16k AudioSet front-end for waveform embeddings."""

    def __init__(
        self,
        repo_dir: Path,
        ckpt_path: Path,
        device: torch.device,
        sample_rate: int = 16000,
    ):
        super().__init__()
        repo_dir = Path(repo_dir)
        ckpt_path = Path(ckpt_path)
        pytorch_dir = repo_dir / "pytorch"
        if str(pytorch_dir) not in sys.path:
            sys.path.insert(0, str(pytorch_dir))
        from models import Cnn14_16k

        model = Cnn14_16k(
            sample_rate=int(sample_rate),
            window_size=512,
            hop_size=160,
            mel_bins=64,
            fmin=50,
            fmax=8000,
            classes_num=527,
        )
        ckpt = torch_load(ckpt_path, map_location=device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            state = ckpt["model"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
        state = {str(k).removeprefix("module."): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        for param in model.parameters():
            param.requires_grad = False

        self.model = model
        self.embed_dim = 2048
        self.frontend_source = "official_audioset_tagging_cnn"
        self.pretraining = "AudioSet"
        self.weights = "Cnn14_16k_mAP=0.438"
        self.input_type = "waveform"
        self.sample_rate = int(sample_rate)
        self.panns_repo_dir = str(repo_dir)
        self.panns_ckpt = str(ckpt_path)
        self.checkpoint = ckpt if isinstance(ckpt, dict) else None

    @torch.no_grad()
    def forward(self, waveform):
        output = self.model(waveform)
        return output["embedding"]


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = F.relu(out + self.shortcut(x), inplace=True)
        return out


class ResNet18Frontend(nn.Module):
    """Torchvision-free ResNet18-style spectrogram encoder."""

    def __init__(self, in_channels=1, base_width=64):
        super().__init__()
        self.in_planes = base_width
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_width, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(base_width, 2, stride=1)
        self.layer2 = self._make_layer(base_width * 2, 2, stride=2)
        self.layer3 = self._make_layer(base_width * 4, 2, stride=2)
        self.layer4 = self._make_layer(base_width * 8, 2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.embed_dim = base_width * 8

    def _make_layer(self, planes, blocks, stride):
        layers = [BasicBlock(self.in_planes, planes, stride)]
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_planes, planes, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.pool(x).flatten(1)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        hidden = int(round(inp * expand_ratio))
        self.use_res_connect = stride == 1 and inp == oup
        layers = []
        if expand_ratio != 1:
            layers.extend([nn.Conv2d(inp, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.ReLU6(inplace=True)])
        layers.extend(
            [
                nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1, groups=hidden, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden, oup, 1, bias=False),
                nn.BatchNorm2d(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2Frontend(nn.Module):
    """PANNs-style MobileNetV2 encoder compatible with MobileNetV2_mAP checkpoints."""

    def __init__(self, in_channels=1, embed_dim=1024):
        super().__init__()
        if int(embed_dim) != 1024:
            raise ValueError("PANNs-style mobilenet_v2 front-end uses a fixed 1024-D embedding")
        self.embed_dim = 1024
        self.bn0 = nn.BatchNorm2d(64)
        input_channel = 32
        last_channel = 1280
        settings = [[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2], [6, 64, 4, 2], [6, 96, 3, 2], [6, 160, 3, 1], [6, 320, 1, 1]]

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, 1, 1, bias=False),
                nn.AvgPool2d(stride),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True),
            )

        def conv_1x1_bn(inp, oup):
            return nn.Sequential(nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), nn.ReLU6(inplace=True))

        layers = [conv_bn(in_channels, input_channel, 2)]
        for t, c, n, s in settings:
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(InvertedResidual(input_channel, c, stride, t))
                input_channel = c
        layers.append(conv_1x1_bn(input_channel, last_channel))
        self.features = nn.Sequential(*layers)
        self.fc1 = nn.Linear(last_channel, self.embed_dim, bias=True)
        self.fc_audioset = nn.Linear(self.embed_dim, 527, bias=True)

    def forward(self, x):
        # Source caches are [B,1,F,T]; PANNs MobileNetV2 expects [B,1,T,64].
        x = x.transpose(2, 3)
        if x.size(-1) != 64:
            x = F.interpolate(x, size=(x.size(2), 64), mode="bilinear", align_corners=False)
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        x = self.features(x)
        x = torch.mean(x, dim=3)
        x = torch.max(x, dim=2).values + torch.mean(x, dim=2)
        return F.relu(self.fc1(x), inplace=True)

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, pool=(2, 2)):
        x = self.net(x)
        # ShuffleFAC caches can have short time axes, so avoid pooling past size 1.
        h, w = x.shape[-2:]
        kernel = (min(int(pool[0]), int(h)), min(int(pool[1]), int(w)))
        return F.avg_pool2d(x, kernel_size=kernel)


class PannsCnn14Frontend(nn.Module):
    """Cnn14-style log-mel encoder without torchlibrosa front-end dependency."""

    def __init__(self, in_channels=1, embed_dim=2048):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.bn0 = nn.BatchNorm2d(in_channels)
        self.block1 = ConvBlock(in_channels, 64)
        self.block2 = ConvBlock(64, 128)
        self.block3 = ConvBlock(128, 256)
        self.block4 = ConvBlock(256, 512)
        self.block5 = ConvBlock(512, 1024)
        self.block6 = ConvBlock(1024, 2048)
        self.fc = nn.Linear(2048, self.embed_dim)

    def forward(self, x):
        x = self.bn0(x)
        x = self.block1(x, (2, 2))
        x = self.block2(x, (2, 2))
        x = self.block3(x, (2, 2))
        x = self.block4(x, (2, 2))
        x = self.block5(x, (2, 2))
        x = self.block6(x, (1, 1))
        x = torch.mean(x, dim=3)
        x = torch.max(x, dim=2).values + torch.mean(x, dim=2)
        return F.relu(self.fc(x), inplace=True)


def build_frontend(name: str, args, config: dict, model_config_path: Path, root: Path, device: torch.device):
    name = str(name)
    frontend_ckpt = resolve_path(args.frontend_ckpt, root) if args.frontend_ckpt else None
    if name == "shufflefac":
        ckpt = infer_shufflefac_ckpt(config, model_config_path, root, args.frontend_ckpt)
        if ckpt is None or not ckpt.exists():
            raise FileNotFoundError("shufflefac front-end requires --frontend_ckpt or best.pt next to model_config")
        model = ShuffleFACFrontend(ckpt, device=device)
        return model.to(device).eval(), model.embed_dim, str(ckpt)

    if name in TORCHVISION_TRAINED_FRONTENDS:
        if frontend_ckpt is None:
            if not args.allow_random_frontend_for_smoke_test:
                raise ValueError(
                    f"frontend={name} requires --frontend_ckpt from train_clip_frontend.py. "
                    "Use --allow_random_frontend_for_smoke_test only for smoke tests."
                )
            num_classes = infer_num_classes_from_config(config)
            model = TorchVisionRandomFrontend(name, num_classes=num_classes, device=device)
            return model.to(device).eval(), model.embed_dim, None
        model = TorchVisionTrainedFrontend(frontend_ckpt, device=device, expected_frontend=name)
        return model.to(device).eval(), model.embed_dim, str(frontend_ckpt)

    if name == "panns_cnn14":
        repo_dir, ckpt_path = resolve_panns_repo_and_ckpt(args, root)
        model = OfficialPannsCnn14_16kFrontend(
            repo_dir=repo_dir,
            ckpt_path=ckpt_path,
            device=device,
            sample_rate=args.panns_sample_rate,
        )
        return model.to(device).eval(), model.embed_dim, str(ckpt_path)

    raise ValueError(f"Unsupported front-end: {name}")


def group_recordings(payload: dict):
    features = payload["features"]
    labels = payload["labels"].long()
    entries = payload.get("entries", [])
    grouped = defaultdict(list)
    for index, entry in enumerate(entries):
        rid = entry.get("recording_id") or Path(entry.get("path", str(index))).stem
        grouped[str(rid)].append(index)
    recordings = []
    for rid, indices in grouped.items():
        indices = sorted(indices, key=lambda i: int(entries[i].get("segment_index", i)) if i < len(entries) else i)
        label_set = {int(labels[i]) for i in indices}
        if len(label_set) != 1:
            raise ValueError(f"Recording {rid} has inconsistent labels: {sorted(label_set)}")
        recordings.append((rid, indices, label_set.pop()))
    recordings.sort(key=lambda item: item[0])
    return features, entries, recordings


def fixed_bag_indices(indices, clips_per_recording: int):
    n = len(indices)
    s = int(clips_per_recording)
    if n <= 0:
        raise ValueError("Empty recording")
    if n >= s:
        positions = np.linspace(0, n - 1, num=s)
        return [indices[int(round(pos))] for pos in positions]
    return [indices[i % n] for i in range(s)]


def frontend_cache_key(
    frontend: str,
    source_cache: Path,
    clips_per_recording: int,
    ckpt_path: Optional[str],
    embed_dim: int,
    checkpoint: Optional[dict] = None,
    smoke_max_recordings_per_split: int = 0,
):
    payload = {
        "cache_schema": EMBEDDING_CACHE_SCHEMA,
        "frontend": frontend,
        "source_cache": str(source_cache.resolve()),
        "clips_per_recording": int(clips_per_recording),
        "frontend_ckpt": str(ckpt_path) if ckpt_path else None,
        "frontend_ckpt_metadata": checkpoint_cache_metadata(ckpt_path, checkpoint),
        "embed_dim": int(embed_dim),
        "smoke_max_recordings_per_split": int(smoke_max_recordings_per_split),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def panns_frontend_cache_key(
    source_cache: Path,
    clips_per_recording: int,
    ckpt_path: Path,
    embed_dim: int,
    args,
    checkpoint=None,
):
    resolved_ckpt = Path(ckpt_path).expanduser().resolve()
    stat = resolved_ckpt.stat()
    payload = {
        "cache_schema": EMBEDDING_CACHE_SCHEMA,
        "frontend": "panns_cnn14",
        "source_cache": str(source_cache.resolve()),
        "clips_per_recording": int(clips_per_recording),
        "panns_ckpt": str(resolved_ckpt),
        "panns_ckpt_metadata": {
            "resolved_path": str(resolved_ckpt),
            "size_bytes": int(stat.st_size),
            "mtime": float(stat.st_mtime),
        },
        "sample_rate": int(args.panns_sample_rate),
        "segment_length": float(args.segment_length),
        "input_type": "waveform",
        "embed_dim": int(embed_dim),
        "weights": "Cnn14_16k_mAP=0.438",
        "smoke_max_recordings_per_split": int(args.smoke_max_recordings_per_split),
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


@torch.no_grad()
def encode_recording_cache(split_name: str, source_cache_path: Path, frontend, frontend_name: str, frontend_ckpt: Optional[str], embed_dim: int, args, device):
    out_dir = Path(args.embedding_cache_dir) if args.embedding_cache_dir else Path(args.output_dir) / "embedding_cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    key = frontend_cache_key(
        frontend_name,
        source_cache_path,
        args.clips_per_recording,
        frontend_ckpt,
        embed_dim,
        getattr(frontend, "checkpoint", None),
        smoke_max_recordings_per_split=args.smoke_max_recordings_per_split,
    )
    out_path = out_dir / f"{frontend_name}_{split_name}_S{args.clips_per_recording}_{key[:12]}.pt"
    if out_path.exists() and not args.rebuild_embedding_cache:
        payload = torch_load(out_path, map_location="cpu")
        if payload.get("cache_key") == key:
            return payload, out_path, True

    source = validate_source_cache(source_cache_path, split_name)
    clip_features, source_entries, recordings = group_recordings(source)
    if int(args.smoke_max_recordings_per_split) > 0:
        recordings = recordings[: int(args.smoke_max_recordings_per_split)]
    expected_input_shape = getattr(frontend, "input_shape", None)
    if expected_input_shape is not None:
        expected_input_shape = [int(v) for v in expected_input_shape]
        actual_input_shape = [int(v) for v in clip_features.shape[1:]]
        if actual_input_shape != expected_input_shape:
            raise ValueError(
                f"Front-end checkpoint input_shape mismatch for {split_name}: "
                f"checkpoint={expected_input_shape}, source_cache={actual_input_shape}"
            )

    labels = []
    recording_ids = []
    selected_indices_all = []
    recording_source_indices = []
    all_embeddings = []
    batch_clips = []
    batch_meta = []
    frontend.eval()

    def flush_batch():
        if not batch_clips:
            return
        clips = torch.stack(batch_clips, dim=0).float().to(device, non_blocking=True)
        emb = frontend(clips).detach().cpu().float()
        for row, (rec_idx, clip_pos) in enumerate(batch_meta):
            all_embeddings[rec_idx][clip_pos] = emb[row]
        batch_clips.clear()
        batch_meta.clear()

    for rec_idx, (rid, indices, label) in enumerate(recordings):
        picked = fixed_bag_indices(indices, args.clips_per_recording)
        labels.append(int(label))
        recording_ids.append(str(rid))
        selected_indices_all.append([int(i) for i in picked])
        recording_source_indices.append([int(i) for i in indices])
        all_embeddings.append([None] * len(indices))
        for clip_pos, clip_index in enumerate(indices):
            batch_clips.append(clip_features[clip_index].float())
            batch_meta.append((rec_idx, clip_pos))
            if len(batch_clips) >= args.frontend_batch_size:
                flush_batch()
    flush_batch()

    flat_embeddings = []
    recording_embedding_indices = []
    fixed_bags = []
    cursor = 0
    for rec_idx, (_rid, indices, _label) in enumerate(recordings):
        rec_embs = all_embeddings[rec_idx]
        if any(item is None for item in rec_embs):
            raise RuntimeError(f"Missing encoded clip embeddings for recording index {rec_idx}")
        rec_indices = list(range(cursor, cursor + len(rec_embs)))
        recording_embedding_indices.append(rec_indices)
        flat_embeddings.extend(rec_embs)
        cursor += len(rec_embs)

        source_index_to_pos = {int(source_index): pos for pos, source_index in enumerate(indices)}
        fixed_bags.append(
            torch.stack([rec_embs[source_index_to_pos[int(source_index)]] for source_index in selected_indices_all[rec_idx]], dim=0)
        )

    if fixed_bags:
        feature_tensor = torch.stack(fixed_bags, dim=0).float()
        clip_embedding_tensor = torch.stack(flat_embeddings, dim=0).float()
    else:
        feature_tensor = torch.empty((0, int(args.clips_per_recording), int(embed_dim)), dtype=torch.float32)
        clip_embedding_tensor = torch.empty((0, int(embed_dim)), dtype=torch.float32)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    payload = {
        "cache_schema": EMBEDDING_CACHE_SCHEMA,
        "cache_key": key,
        "frontend": frontend_name,
        "frontend_ckpt": frontend_ckpt,
        "source_cache": str(source_cache_path),
        "split_name": split_name,
        "features": feature_tensor,
        "clip_embeddings": clip_embedding_tensor,
        "labels": label_tensor,
        "recording_ids": recording_ids,
        "recording_embedding_indices": recording_embedding_indices,
        "recording_source_indices": recording_source_indices,
        "selected_source_indices": selected_indices_all,
        "clips_per_recording": int(args.clips_per_recording),
        "embed_dim": int(feature_tensor.size(-1)) if feature_tensor.ndim == 3 else int(embed_dim),
        "metadata": source.get("metadata", {}),
        "source_split_name": source.get("split_name"),
        "sampling_protocol": {
            "train": "random_recording_clip_bag_from_all_embeddings",
            "eval": "deterministic_multisample_recording_clip_bag_from_all_embeddings",
        },
        "smoke_max_recordings_per_split": int(args.smoke_max_recordings_per_split),
    }
    torch.save(payload, out_path)
    return payload, out_path, False


@torch.no_grad()
def encode_recording_cache_panns(
    split_name: str,
    source_cache_path: Path,
    frontend,
    frontend_name: str,
    frontend_ckpt: Optional[str],
    embed_dim: int,
    args,
    device,
    root: Path,
):
    out_dir = Path(args.embedding_cache_dir) if args.embedding_cache_dir else Path(args.output_dir) / "embedding_cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(frontend_ckpt) if frontend_ckpt else Path(getattr(frontend, "panns_ckpt"))
    key = panns_frontend_cache_key(
        source_cache_path,
        args.clips_per_recording,
        ckpt_path,
        embed_dim,
        args,
        getattr(frontend, "checkpoint", None),
    )
    out_path = out_dir / f"panns_cnn14_{split_name}_S{args.clips_per_recording}_{key[:12]}.pt"
    if out_path.exists() and not args.rebuild_embedding_cache:
        payload = torch_load(out_path, map_location="cpu")
        if payload.get("cache_key") == key:
            return payload, out_path, True

    source = validate_source_cache(source_cache_path, split_name)
    _features, entries, recordings = group_recordings(source)
    if int(args.smoke_max_recordings_per_split) > 0:
        recordings = recordings[: int(args.smoke_max_recordings_per_split)]

    labels = []
    recording_ids = []
    selected_indices_all = []
    recording_source_indices = []
    all_embeddings = []
    batch_waveforms = []
    batch_meta = []
    frontend.eval()

    def flush_batch():
        if not batch_waveforms:
            return
        waveforms = torch.stack(batch_waveforms, dim=0).float().to(device, non_blocking=True)
        emb = frontend(waveforms).detach().cpu().float()
        if emb.ndim != 2 or int(emb.size(-1)) != int(embed_dim):
            raise ValueError(f"Expected PANNs embedding [B, {embed_dim}], got {tuple(emb.shape)}")
        for row, (rec_idx, clip_pos) in enumerate(batch_meta):
            all_embeddings[rec_idx][clip_pos] = emb[row]
        batch_waveforms.clear()
        batch_meta.clear()

    for rec_idx, (rid, indices, label) in enumerate(recordings):
        picked = fixed_bag_indices(indices, args.clips_per_recording)
        labels.append(int(label))
        recording_ids.append(str(rid))
        selected_indices_all.append([int(i) for i in picked])
        recording_source_indices.append([int(i) for i in indices])
        all_embeddings.append([None] * len(indices))
        for clip_pos, clip_index in enumerate(indices):
            waveform = load_waveform_segment_from_entry(
                entries[clip_index],
                source_cache_path=source_cache_path,
                root=root,
                target_sample_rate=args.panns_sample_rate,
                segment_length=args.segment_length,
            )
            batch_waveforms.append(waveform)
            batch_meta.append((rec_idx, clip_pos))
            if len(batch_waveforms) >= args.frontend_batch_size:
                flush_batch()
    flush_batch()

    flat_embeddings = []
    recording_embedding_indices = []
    fixed_bags = []
    cursor = 0
    for rec_idx, (_rid, indices, _label) in enumerate(recordings):
        rec_embs = all_embeddings[rec_idx]
        if any(item is None for item in rec_embs):
            raise RuntimeError(f"Missing encoded PANNs clip embeddings for recording index {rec_idx}")
        rec_indices = list(range(cursor, cursor + len(rec_embs)))
        recording_embedding_indices.append(rec_indices)
        flat_embeddings.extend(rec_embs)
        cursor += len(rec_embs)

        source_index_to_pos = {int(source_index): pos for pos, source_index in enumerate(indices)}
        fixed_bags.append(
            torch.stack([rec_embs[source_index_to_pos[int(source_index)]] for source_index in selected_indices_all[rec_idx]], dim=0)
        )

    if fixed_bags:
        feature_tensor = torch.stack(fixed_bags, dim=0).float()
        clip_embedding_tensor = torch.stack(flat_embeddings, dim=0).float()
    else:
        feature_tensor = torch.empty((0, int(args.clips_per_recording), int(embed_dim)), dtype=torch.float32)
        clip_embedding_tensor = torch.empty((0, int(embed_dim)), dtype=torch.float32)
    label_tensor = torch.tensor(labels, dtype=torch.long)
    payload = {
        "cache_schema": EMBEDDING_CACHE_SCHEMA,
        "cache_key": key,
        "frontend": "panns_cnn14",
        "frontend_source": "official_audioset_tagging_cnn",
        "frontend_ckpt": str(frontend_ckpt),
        "source_cache": str(source_cache_path),
        "split_name": split_name,
        "features": feature_tensor,
        "clip_embeddings": clip_embedding_tensor,
        "labels": label_tensor,
        "recording_ids": recording_ids,
        "recording_embedding_indices": recording_embedding_indices,
        "recording_source_indices": recording_source_indices,
        "selected_source_indices": selected_indices_all,
        "clips_per_recording": int(args.clips_per_recording),
        "embed_dim": 2048,
        "embed_dim_raw": 2048,
        "input_type": "waveform",
        "sample_rate": int(args.panns_sample_rate),
        "segment_length": float(args.segment_length),
        "weights": "Cnn14_16k_mAP=0.438",
        "pretraining": "AudioSet",
        "metadata": source.get("metadata", {}),
        "source_split_name": source.get("split_name"),
        "sampling_protocol": {
            "train": "random_recording_clip_bag_from_all_embeddings",
            "eval": "deterministic_multisample_recording_clip_bag_from_all_embeddings",
        },
        "smoke_max_recordings_per_split": int(args.smoke_max_recordings_per_split),
    }
    torch.save(payload, out_path)
    return payload, out_path, False


def validate_recording_overlap(cache_payloads: dict):
    sets = {split: set(payload.get("recording_ids", [])) for split, payload in cache_payloads.items()}
    overlaps = {}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = sorted(sets[a].intersection(sets[b]))
        overlaps[f"{a}-{b}"] = overlap
        if overlap:
            raise ValueError(f"Recording-level leakage detected for {a}-{b}: {overlap[:10]}")
    return overlaps


class EmbeddingBagDataset(Dataset):
    """Reads precomputed recording bags with shape [S, D]."""

    def __init__(self, cache_payload: dict):
        self.features = cache_payload["features"].float()
        self.labels = cache_payload["labels"].long()
        self.recording_ids = [str(x) for x in cache_payload.get("recording_ids", [])]
        if self.features.ndim != 3:
            raise ValueError(f"Expected [R, S, D] features, got {tuple(self.features.shape)}")

    def __len__(self):
        return int(self.labels.numel())

    def __getitem__(self, index):
        rid = self.recording_ids[index] if index < len(self.recording_ids) else str(index)
        return self.features[index], self.labels[index], rid


class DynamicEmbeddingBagDataset(Dataset):
    """Samples recording bags from all cached per-clip embeddings."""

    def __init__(self, cache_payload: dict, clips_per_recording: int, train: bool = False, seed: int = 42):
        clip_embeddings = cache_payload.get("clip_embeddings")
        rec_indices = cache_payload.get("recording_embedding_indices")
        if clip_embeddings is None or rec_indices is None:
            raise ValueError("sn_expd_warmup5 requires clip_embeddings and recording_embedding_indices in the cache")
        self.features = clip_embeddings.float()
        self.labels = cache_payload["labels"].long()
        self.recording_ids = [str(x) for x in cache_payload.get("recording_ids", [])]
        self.recording_embedding_indices = [[int(i) for i in indices] for indices in rec_indices]
        self.clips_per_recording = int(clips_per_recording)
        self.train = bool(train)
        self.rng = random.Random(int(seed))
        if len(self.recording_embedding_indices) != int(self.labels.numel()):
            raise ValueError("recording_embedding_indices/labels length mismatch")

    def __len__(self):
        return int(self.labels.numel())

    def _sample_indices(self, indices, eval_sample_id: int = 0, eval_samples: int = 1):
        s = self.clips_per_recording
        n = len(indices)
        if n <= 0:
            raise ValueError("Empty recording bag")
        if self.train:
            if n >= s:
                return sorted(self.rng.sample(indices, s))
            return [self.rng.choice(indices) for _ in range(s)]

        eval_samples = max(int(eval_samples), 1)
        eval_sample_id = int(eval_sample_id) % eval_samples
        if n >= s:
            if eval_samples > 1:
                phase = (eval_sample_id + 0.5) / eval_samples
                positions = (np.arange(s, dtype=np.float64) + phase) * n / s - 0.5
                positions = np.clip(np.rint(positions), 0, n - 1).astype(int)
            else:
                positions = np.rint(np.linspace(0, n - 1, num=s)).astype(int)
            return [indices[int(pos)] for pos in positions]
        return [indices[(i + eval_sample_id) % n] for i in range(s)]

    def get_eval_item(self, index, eval_sample_id: int = 0, eval_samples: int = 1):
        picked = self._sample_indices(self.recording_embedding_indices[index], eval_sample_id, eval_samples)
        rid = self.recording_ids[index] if index < len(self.recording_ids) else str(index)
        return self.features[picked].float(), self.labels[index], rid

    def __getitem__(self, index):
        if not self.train:
            return self.get_eval_item(index)
        picked = self._sample_indices(self.recording_embedding_indices[index])
        rid = self.recording_ids[index] if index < len(self.recording_ids) else str(index)
        return self.features[picked].float(), self.labels[index], rid


class EmbeddingAdapter(nn.Module):
    """Project raw front-end embeddings to a shared head dimension."""

    def __init__(self, in_dim, adapter_dim=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, adapter_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(adapter_dim),
        )

    def forward(self, x):
        return self.net(x)


class MeanHead(nn.Module):
    def __init__(self, in_dim, num_classes, adapter_dim=256, dropout=0.1):
        super().__init__()
        self.adapter = EmbeddingAdapter(in_dim, adapter_dim=adapter_dim, dropout=dropout)
        self.classifier = nn.Linear(adapter_dim, num_classes)

    def forward(self, x, return_parts=False):
        z = self.adapter(x)
        pooled = z.mean(dim=1)
        logits = self.classifier(pooled)
        return (logits, {}) if return_parts else logits


class AttentionHead(nn.Module):
    def __init__(self, in_dim, num_classes, adapter_dim=256, dropout=0.1):
        super().__init__()
        self.adapter = EmbeddingAdapter(in_dim, adapter_dim=adapter_dim, dropout=dropout)
        self.attn = nn.Sequential(
            nn.Linear(adapter_dim, max(adapter_dim // 2, 1)),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(max(adapter_dim // 2, 1), 1),
        )
        self.classifier = nn.Linear(adapter_dim, num_classes)

    def forward(self, x, return_parts=False):
        z = self.adapter(x)
        scores = self.attn(z)
        weights = torch.softmax(scores, dim=1)
        pooled = (weights * z).sum(dim=1)
        logits = self.classifier(pooled)
        return (logits, {"attn_weights": weights}) if return_parts else logits


class NoiseGraphConv(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.msg_mlp = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim, dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, z, knn_idx):
        b, s, d = z.shape
        k = knn_idx.size(-1)
        expanded = z.unsqueeze(1).expand(b, s, s, d)
        gather_idx = knn_idx.unsqueeze(-1).expand(b, s, k, d)
        neighbors = torch.gather(expanded, dim=2, index=gather_idx)
        delta = (neighbors - z.unsqueeze(2)).mean(dim=2)
        return self.norm(self.msg_mlp(torch.cat([z, delta], dim=-1)))


class SignalNoiseHead(nn.Module):
    """Signal-noise decoupled ETA head operating directly on [B, S, D] embeddings."""

    def __init__(
        self,
        in_dim,
        num_classes,
        sig_dim=32,
        noise_dim=32,
        graph_k=2,
        edge_mode="similarity",
        sim_threshold=0.8,
        use_temperature=False,
        signal_top_k=0,
        topk_warmup_epochs=0,
        adapter_dim=256,
        dropout=0.1,
    ):
        super().__init__()
        self.adapter = EmbeddingAdapter(in_dim, adapter_dim=adapter_dim, dropout=dropout)
        self.signal_proj = nn.Sequential(
            nn.Linear(adapter_dim, sig_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(sig_dim, sig_dim),
            nn.LayerNorm(sig_dim),
        )
        self.noise_proj = nn.Sequential(
            nn.Linear(adapter_dim, noise_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(noise_dim, noise_dim),
            nn.LayerNorm(noise_dim),
        )
        self.graph_k = int(graph_k)
        self.edge_mode = str(edge_mode)
        self.sim_threshold = float(sim_threshold)
        self.use_temperature = bool(use_temperature)
        self.signal_top_k = int(signal_top_k)
        self.topk_warmup_epochs = int(topk_warmup_epochs)
        self.current_epoch = 0
        self.noise_graph = NoiseGraphConv(noise_dim, dropout=dropout)
        self.graph_res_scale = nn.Parameter(torch.tensor(0.1))
        self.noise_attn = nn.Sequential(
            nn.Linear(noise_dim, max(noise_dim // 2, 1)),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(max(noise_dim // 2, 1), 1),
        )
        self.noise_to_suppression = nn.Sequential(nn.LayerNorm(noise_dim), nn.Linear(noise_dim, sig_dim), nn.Sigmoid())
        self.signal_attn = nn.Sequential(
            nn.Linear(sig_dim, max(sig_dim // 2, 1)),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(max(sig_dim // 2, 1), 1),
        )
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(sig_dim, num_classes))
        if use_temperature:
            self.temperature = nn.Parameter(torch.tensor(1.0))
            self.noise_temperature = nn.Parameter(torch.tensor(1.0))
        self.last_attn_entropy = torch.tensor(0.0)
        self.last_graph_delta_norm = torch.tensor(0.0)
        self.last_avg_graph_degree = torch.tensor(0.0)
        self.last_signal_topk_count = torch.tensor(0.0)

    def _build_graph(self, z):
        b, s, _ = z.shape
        if s <= 1:
            self.last_avg_graph_degree = z.new_tensor(0.0).detach()
            return torch.zeros((b, s, 1), dtype=torch.long, device=z.device)

        adj = torch.zeros((b, s, s), dtype=torch.bool, device=z.device)
        if self.edge_mode in ("temporal", "temporal_similarity", "threshold_similarity"):
            for i in range(s):
                if i > 0:
                    adj[:, i, i - 1] = True
                if i + 1 < s:
                    adj[:, i, i + 1] = True

        if self.edge_mode == "threshold_similarity":
            normed = F.normalize(z, p=2, dim=-1)
            sim = torch.bmm(normed, normed.transpose(1, 2))
            eye = torch.eye(s, dtype=torch.bool, device=z.device).unsqueeze(0)
            sim = sim.masked_fill(eye, -float("inf"))
            adj = adj | (sim > self.sim_threshold)
        elif self.edge_mode in ("similarity", "temporal_similarity"):
            k = min(self.graph_k, s - 1)
            normed = F.normalize(z, p=2, dim=-1)
            sim = torch.bmm(normed, normed.transpose(1, 2))
            eye = torch.eye(s, dtype=torch.bool, device=z.device).unsqueeze(0)
            sim = sim.masked_fill(eye, -float("inf"))
            sim_idx = sim.topk(k=k, dim=-1).indices
            adj.scatter_(2, sim_idx, True)

        self.last_avg_graph_degree = adj.sum(dim=-1).float().mean().detach()
        max_degree = max(int(adj.sum(dim=-1).max().item()), 1)
        out = torch.zeros((b, s, max_degree), dtype=torch.long, device=z.device)
        for bi in range(b):
            for i in range(s):
                idx = torch.nonzero(adj[bi, i], as_tuple=False).flatten()
                if idx.numel() == 0:
                    idx = torch.tensor([i], dtype=torch.long, device=z.device)
                if idx.numel() < max_degree:
                    idx = torch.cat([idx, idx[:1].expand(max_degree - idx.numel())], dim=0)
                out[bi, i] = idx[:max_degree]
        return out

    def _noise_softmax(self, scores):
        if hasattr(self, "noise_temperature"):
            clamped_temp = torch.clamp(self.noise_temperature, min=0.1)
            return torch.softmax(scores / clamped_temp, dim=1)
        return torch.softmax(scores, dim=1)

    def topk_masked_softmax(self, signal_scores):
        s = signal_scores.size(1)
        apply_topk = False
        if self.signal_top_k > 0 and s > 1:
            apply_topk = True
            if self.training and getattr(self, "current_epoch", 1) <= self.topk_warmup_epochs:
                apply_topk = False

        if apply_topk:
            k = min(self.signal_top_k, s)
            scores_2d = signal_scores.squeeze(-1)
            topk_idx = scores_2d.topk(k=k, dim=1).indices
            mask_2d = torch.zeros_like(scores_2d, dtype=torch.bool)
            mask_2d.scatter_(1, topk_idx, True)
            min_val = torch.finfo(signal_scores.dtype).min
            masked_scores = signal_scores.masked_fill(~mask_2d.unsqueeze(-1), min_val)
            signal_topk_mask = mask_2d
        else:
            masked_scores = signal_scores
            signal_topk_mask = torch.ones_like(signal_scores.squeeze(-1), dtype=torch.bool)

        if hasattr(self, "temperature"):
            clamped_temp = torch.clamp(self.temperature, min=0.1)
            signal_weights = torch.softmax(masked_scores / clamped_temp, dim=1)
        else:
            signal_weights = torch.softmax(masked_scores, dim=1)
        return signal_weights, signal_topk_mask

    def forward(self, x, return_parts=False):
        z = self.adapter(x)
        z_sig = self.signal_proj(z)
        z_noise = self.noise_proj(z)
        if z_noise.size(1) > 1:
            update = self.noise_graph(z_noise, self._build_graph(z_noise))
            z_noise_smooth = z_noise + self.graph_res_scale * update
        else:
            z_noise_smooth = z_noise
            self.last_avg_graph_degree = z_noise.new_tensor(0.0).detach()
        self.last_graph_delta_norm = (z_noise_smooth - z_noise).norm(dim=-1).mean().detach()

        noise_scores = self.noise_attn(z_noise_smooth)
        noise_weights = self._noise_softmax(noise_scores)
        global_noise = (noise_weights * z_noise_smooth).sum(dim=1)
        suppression = self.noise_to_suppression(global_noise).unsqueeze(1)
        z_sig_filtered = z_sig * (1.0 - suppression)

        signal_scores = self.signal_attn(z_sig_filtered)
        signal_weights, signal_topk_mask = self.topk_masked_softmax(signal_scores)
        entropy = -(signal_weights * (signal_weights + 1e-8).log()).sum(dim=1).mean()
        self.last_attn_entropy = entropy.detach()
        self.last_signal_topk_count = signal_topk_mask.float().sum(dim=1).mean().detach()

        pooled = (signal_weights * z_sig_filtered).sum(dim=1)
        logits = self.classifier(pooled)
        parts = {
            "z_sig": z_sig,
            "z_noise": z_noise,
            "z_noise_smooth": z_noise_smooth,
            "global_noise": global_noise,
            "suppression": suppression.squeeze(1),
            "signal_weights": signal_weights,
            "signal_topk_mask": signal_topk_mask,
            "noise_weights": noise_weights,
        }
        return (logits, parts) if return_parts else logits


class SignalNoiseExpDWarmup5Head(nn.Module):
    """ExpD Warmup5 SN head over cached embeddings, without the robustness adapter."""

    def __init__(self, in_dim, num_classes, sig_dim=32, noise_dim=32, graph_k=2, dropout=0.1):
        super().__init__()
        self.input_norm = nn.LayerNorm(in_dim)
        self.signal_proj = nn.Sequential(
            nn.Linear(in_dim, sig_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(sig_dim, sig_dim),
            nn.LayerNorm(sig_dim),
        )
        self.noise_proj = nn.Sequential(
            nn.Linear(in_dim, noise_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(noise_dim, noise_dim),
            nn.LayerNorm(noise_dim),
        )
        self.graph_k = int(graph_k)
        self.edge_mode = EXP_D_EDGE_MODE
        self.sim_threshold = float(EXP_D_SIM_THRESHOLD)
        self.signal_top_k = int(EXP_D_SIGNAL_TOP_K)
        self.topk_warmup_epochs = int(EXP_D_TOPK_WARMUP_EPOCHS)
        self.current_epoch = 0
        self.noise_graph = NoiseGraphConv(noise_dim, dropout=dropout)
        self.graph_res_scale = nn.Parameter(torch.tensor(0.1))
        self.noise_attn = nn.Sequential(
            nn.Linear(noise_dim, max(noise_dim // 2, 1)),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(max(noise_dim // 2, 1), 1),
        )
        self.noise_to_suppression = nn.Sequential(nn.LayerNorm(noise_dim), nn.Linear(noise_dim, sig_dim), nn.Sigmoid())
        self.signal_attn = nn.Sequential(
            nn.Linear(sig_dim, max(sig_dim // 2, 1)),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(max(sig_dim // 2, 1), 1),
        )
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(sig_dim, num_classes))
        self.last_attn_entropy = torch.tensor(0.0)
        self.last_graph_delta_norm = torch.tensor(0.0)
        self.last_avg_graph_degree = torch.tensor(0.0)
        self.last_signal_topk_count = torch.tensor(0.0)

    def _build_graph(self, z):
        b, s, _ = z.shape
        if s <= 1:
            self.last_avg_graph_degree = z.new_tensor(0.0).detach()
            return torch.zeros((b, s, 1), dtype=torch.long, device=z.device)

        adj = torch.zeros((b, s, s), dtype=torch.bool, device=z.device)
        for i in range(s):
            if i > 0:
                adj[:, i, i - 1] = True
            if i + 1 < s:
                adj[:, i, i + 1] = True

        normed = F.normalize(z, p=2, dim=-1)
        sim = torch.bmm(normed, normed.transpose(1, 2))
        eye = torch.eye(s, dtype=torch.bool, device=z.device).unsqueeze(0)
        sim = sim.masked_fill(eye, -float("inf"))
        adj = adj | (sim > self.sim_threshold)

        self.last_avg_graph_degree = adj.sum(dim=-1).float().mean().detach()
        max_degree = max(int(adj.sum(dim=-1).max().item()), 1)
        out = torch.zeros((b, s, max_degree), dtype=torch.long, device=z.device)
        for bi in range(b):
            for i in range(s):
                idx = torch.nonzero(adj[bi, i], as_tuple=False).flatten()
                if idx.numel() == 0:
                    idx = torch.tensor([i], dtype=torch.long, device=z.device)
                if idx.numel() < max_degree:
                    idx = torch.cat([idx, idx[:1].expand(max_degree - idx.numel())], dim=0)
                out[bi, i] = idx[:max_degree]
        return out

    def topk_masked_softmax(self, signal_scores):
        s = signal_scores.size(1)
        apply_topk = self.signal_top_k > 0 and s > 1
        if apply_topk and self.training and getattr(self, "current_epoch", 1) <= self.topk_warmup_epochs:
            apply_topk = False

        if apply_topk:
            k = min(self.signal_top_k, s)
            scores_2d = signal_scores.squeeze(-1)
            topk_idx = scores_2d.topk(k=k, dim=1).indices
            mask_2d = torch.zeros_like(scores_2d, dtype=torch.bool)
            mask_2d.scatter_(1, topk_idx, True)
            min_val = torch.finfo(signal_scores.dtype).min
            masked_scores = signal_scores.masked_fill(~mask_2d.unsqueeze(-1), min_val)
            signal_topk_mask = mask_2d
        else:
            masked_scores = signal_scores
            signal_topk_mask = torch.ones_like(signal_scores.squeeze(-1), dtype=torch.bool)

        return torch.softmax(masked_scores, dim=1), signal_topk_mask

    def forward(self, x, return_parts=False):
        z = self.input_norm(x)
        z_sig = self.signal_proj(z)
        z_noise = self.noise_proj(z)
        if z_noise.size(1) > 1:
            update = self.noise_graph(z_noise, self._build_graph(z_noise))
            z_noise_smooth = z_noise + self.graph_res_scale * update
        else:
            z_noise_smooth = z_noise
            self.last_avg_graph_degree = z_noise.new_tensor(0.0).detach()
        self.last_graph_delta_norm = (z_noise_smooth - z_noise).norm(dim=-1).mean().detach()

        noise_scores = self.noise_attn(z_noise_smooth)
        noise_weights = torch.softmax(noise_scores, dim=1)
        global_noise = (noise_weights * z_noise_smooth).sum(dim=1)
        suppression = self.noise_to_suppression(global_noise).unsqueeze(1)
        z_sig_filtered = z_sig * (1.0 - suppression)

        signal_scores = self.signal_attn(z_sig_filtered)
        signal_weights, signal_topk_mask = self.topk_masked_softmax(signal_scores)
        entropy = -(signal_weights * (signal_weights + 1e-8).log()).sum(dim=1).mean()
        self.last_attn_entropy = entropy.detach()
        self.last_signal_topk_count = signal_topk_mask.float().sum(dim=1).mean().detach()

        pooled = (signal_weights * z_sig_filtered).sum(dim=1)
        logits = self.classifier(pooled)
        parts = {
            "z_sig": z_sig,
            "z_noise": z_noise,
            "z_noise_smooth": z_noise_smooth,
            "global_noise": global_noise,
            "suppression": suppression.squeeze(1),
            "signal_weights": signal_weights,
            "signal_topk_mask": signal_topk_mask,
            "noise_weights": noise_weights,
        }
        return (logits, parts) if return_parts else logits


class BiGRUHead(nn.Module):
    def __init__(self, in_dim, num_classes, hidden_dim=64, adapter_dim=256, dropout=0.1):
        super().__init__()
        self.adapter = EmbeddingAdapter(in_dim, adapter_dim=adapter_dim, dropout=dropout)
        self.gru = nn.GRU(adapter_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim * 2, num_classes))

    def forward(self, x, return_parts=False):
        out, _ = self.gru(self.adapter(x))
        pooled = out.mean(dim=1)
        logits = self.classifier(pooled)
        return (logits, {}) if return_parts else logits


class MILLinearSoftmaxHead(nn.Module):
    """Class-wise softmax MIL pooling over per-clip linear logits."""

    def __init__(self, in_dim, num_classes, adapter_dim=256, dropout=0.1, eps=1e-8):
        super().__init__()
        self.adapter = EmbeddingAdapter(in_dim, adapter_dim=adapter_dim, dropout=dropout)
        self.clip_classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(adapter_dim, num_classes))
        self.eps = float(eps)

    def forward(self, x, return_parts=False):
        z = self.adapter(x)
        clip_logits = self.clip_classifier(z)
        clip_probs = torch.softmax(clip_logits, dim=-1)
        bag_probs = (clip_probs.pow(2)).sum(dim=1) / (clip_probs.sum(dim=1) + self.eps)
        bag_probs = bag_probs / (bag_probs.sum(dim=-1, keepdim=True) + self.eps)
        log_probs = torch.log(bag_probs + self.eps)
        parts = {"clip_logits": clip_logits, "clip_probs": clip_probs, "bag_probs": bag_probs}
        return (log_probs, parts) if return_parts else log_probs


def build_head(name: str, in_dim: int, num_classes: int, args):
    if name == "mean":
        return MeanHead(in_dim, num_classes, adapter_dim=args.adapter_dim, dropout=args.dropout)
    if name == "attention":
        return AttentionHead(in_dim, num_classes, adapter_dim=args.adapter_dim, dropout=args.dropout)
    if name == "sn_decoupled":
        return SignalNoiseHead(
            in_dim,
            num_classes,
            sig_dim=args.sig_dim,
            noise_dim=args.noise_dim,
            graph_k=args.graph_k,
            edge_mode=args.edge_mode,
            sim_threshold=args.sim_threshold,
            use_temperature=args.use_temperature,
            signal_top_k=args.signal_top_k,
            topk_warmup_epochs=args.topk_warmup_epochs,
            adapter_dim=args.adapter_dim,
            dropout=args.dropout,
        )
    if name == "sn_expd_warmup5":
        return SignalNoiseExpDWarmup5Head(
            in_dim,
            num_classes,
            sig_dim=args.sig_dim,
            noise_dim=args.noise_dim,
            graph_k=args.graph_k,
            dropout=args.dropout,
        )
    if name == "bigru":
        return BiGRUHead(in_dim, num_classes, hidden_dim=args.gru_hidden_dim, adapter_dim=args.adapter_dim, dropout=args.dropout)
    if name == "mil_linear_softmax":
        return MILLinearSoftmaxHead(in_dim, num_classes, adapter_dim=args.adapter_dim, dropout=args.dropout)
    raise ValueError(f"Unsupported head: {name}")


def orthogonal_loss(z_sig, z_noise):
    dim = min(z_sig.size(-1), z_noise.size(-1))
    if dim <= 0:
        return z_sig.new_tensor(0.0)
    sig = F.normalize(z_sig[..., :dim], p=2, dim=-1)
    noise = F.normalize(z_noise[..., :dim], p=2, dim=-1)
    return ((sig * noise).sum(dim=-1).pow(2)).mean()


def noise_consistency_loss(z_noise):
    mean_noise = z_noise.mean(dim=1, keepdim=True)
    return F.mse_loss(z_noise, mean_noise.expand_as(z_noise))


def metrics_from_arrays(y_true, y_pred):
    return {
        "ACC": float(accuracy_score(y_true, y_pred)),
        "Macro-F1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "Macro-Precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "Macro-Recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "Weighted-F1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def compute_loss(model, x, labels, criterion, args):
    logits, parts = model(x, return_parts=True)
    task = criterion(logits, labels)
    orth = x.new_tensor(0.0)
    noise = x.new_tensor(0.0)
    if "z_sig" in parts and "z_noise" in parts:
        orth = orthogonal_loss(parts["z_sig"], parts["z_noise"])
        noise = noise_consistency_loss(parts["z_noise"])
    total = task + float(args.lambda_orth) * orth + float(args.lambda_noise_consistency) * noise
    return logits, {"total": total, "task": task, "orth": orth, "noise_consistency": noise}


def run_epoch(model, loader, criterion, device, args, optimizer=None, epoch=None):
    train = optimizer is not None
    model.train(train)
    if train and epoch is not None and hasattr(model, "current_epoch"):
        model.current_epoch = int(epoch)
    elif not train and hasattr(model, "current_epoch"):
        # Evaluation mirrors train_eval_signal_noise_decoupled.py: always past warmup.
        model.current_epoch = 99999
    totals = {"total": 0.0, "task": 0.0, "orth": 0.0, "noise_consistency": 0.0}
    y_true = []
    logits_all = []
    n = 0
    entropy_vals = []
    delta_vals = []
    degree_vals = []
    topk_count_vals = []
    for x, labels, _rids in loader:
        x = x.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        logits, losses = compute_loss(model, x, labels, criterion, args)
        if train:
            losses["total"].backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))
            optimizer.step()
        batch = int(labels.size(0))
        n += batch
        for key, value in losses.items():
            totals[key] += float(value.detach().cpu()) * batch
        y_true.extend(labels.detach().cpu().numpy().tolist())
        logits_all.append(logits.detach().cpu())
        if hasattr(model, "last_attn_entropy"):
            entropy_vals.append(float(model.last_attn_entropy.detach().cpu()))
        if hasattr(model, "last_graph_delta_norm"):
            delta_vals.append(float(model.last_graph_delta_norm.detach().cpu()))
        if hasattr(model, "last_avg_graph_degree"):
            degree_vals.append(float(model.last_avg_graph_degree.detach().cpu()))
        if hasattr(model, "last_signal_topk_count"):
            topk_count_vals.append(float(model.last_signal_topk_count.detach().cpu()))
    logits_np = torch.cat(logits_all, dim=0).numpy() if logits_all else np.zeros((0, 1), dtype=np.float32)
    pred_np = logits_np.argmax(axis=1) if logits_np.size else np.asarray([], dtype=np.int64)
    metrics = metrics_from_arrays(np.asarray(y_true), pred_np) if y_true else {"ACC": math.nan, "Macro-F1": math.nan, "Macro-Precision": math.nan, "Macro-Recall": math.nan, "Weighted-F1": math.nan}
    for key, value in totals.items():
        metrics[f"{key}_loss"] = value / max(n, 1)
    if entropy_vals:
        metrics["attn_entropy"] = float(np.mean(entropy_vals))
    if delta_vals:
        metrics["graph_delta_norm"] = float(np.mean(delta_vals))
    if degree_vals:
        metrics["avg_graph_degree"] = float(np.mean(degree_vals))
    if topk_count_vals:
        metrics["avg_signal_topk_count"] = float(np.mean(topk_count_vals))
    if hasattr(model, "graph_res_scale"):
        metrics["graph_res_scale"] = float(model.graph_res_scale.detach().cpu())
    if hasattr(model, "temperature"):
        metrics["learned_temperature"] = float(model.temperature.detach().cpu())
    if hasattr(model, "noise_temperature"):
        metrics["learned_noise_temperature"] = float(model.noise_temperature.detach().cpu())
    return metrics


@torch.no_grad()
def collect_multisample_embedding_predictions(model, dataset, criterion, device, args):
    model.eval()
    if hasattr(model, "current_epoch"):
        model.current_epoch = 99999
    eval_samples = max(int(args.eval_samples), 1)
    batch_size = max(int(args.batch_size), 1)
    y_true = []
    recording_ids = []
    logits_all = []
    totals = {"total": 0.0, "task": 0.0, "orth": 0.0, "noise_consistency": 0.0}
    entropy_vals = []
    delta_vals = []
    degree_vals = []
    topk_count_vals = []

    for start in range(0, len(dataset), batch_size):
        batch_indices = list(range(start, min(start + batch_size, len(dataset))))
        labels = torch.stack([dataset.labels[i] for i in batch_indices]).to(device, non_blocking=True)
        sample_logits = []
        sample_losses = []
        for sample_id in range(eval_samples):
            xs = []
            for i in batch_indices:
                x, _label, _rid = dataset.get_eval_item(i, sample_id, eval_samples)
                xs.append(x)
            x_batch = torch.stack(xs, dim=0).to(device, non_blocking=True)
            logits, losses = compute_loss(model, x_batch, labels, criterion, args)
            sample_logits.append(logits.detach())
            sample_losses.append({key: value.detach() for key, value in losses.items()})
            if hasattr(model, "last_attn_entropy"):
                entropy_vals.append(float(model.last_attn_entropy.detach().cpu()))
            if hasattr(model, "last_graph_delta_norm"):
                delta_vals.append(float(model.last_graph_delta_norm.detach().cpu()))
            if hasattr(model, "last_avg_graph_degree"):
                degree_vals.append(float(model.last_avg_graph_degree.detach().cpu()))
            if hasattr(model, "last_signal_topk_count"):
                topk_count_vals.append(float(model.last_signal_topk_count.detach().cpu()))

        mean_logits = torch.stack(sample_logits, dim=0).mean(dim=0)
        task_on_mean = criterion(mean_logits, labels).detach()
        batch = int(labels.size(0))
        mean_losses = {key: torch.stack([losses[key] for losses in sample_losses]).mean() for key in totals}
        mean_losses["task"] = task_on_mean
        mean_losses["total"] = (
            task_on_mean
            + float(args.lambda_orth) * mean_losses["orth"]
            + float(args.lambda_noise_consistency) * mean_losses["noise_consistency"]
        )
        for key, value in mean_losses.items():
            totals[key] += float(value.detach().cpu()) * batch
        y_true.extend(labels.detach().cpu().numpy().tolist())
        recording_ids.extend([dataset.recording_ids[i] if i < len(dataset.recording_ids) else str(i) for i in batch_indices])
        logits_all.append(mean_logits.detach().cpu())

    logits_np = torch.cat(logits_all, dim=0).numpy() if logits_all else np.zeros((0, 1), dtype=np.float32)
    pred_np = logits_np.argmax(axis=1) if logits_np.size else np.asarray([], dtype=np.int64)
    metrics = metrics_from_arrays(np.asarray(y_true), pred_np) if y_true else {
        "ACC": math.nan,
        "Macro-F1": math.nan,
        "Macro-Precision": math.nan,
        "Macro-Recall": math.nan,
        "Weighted-F1": math.nan,
    }
    n = len(y_true)
    for key, value in totals.items():
        metrics[f"{key}_loss"] = value / max(n, 1)
    if entropy_vals:
        metrics["attn_entropy"] = float(np.mean(entropy_vals))
    if delta_vals:
        metrics["graph_delta_norm"] = float(np.mean(delta_vals))
    if degree_vals:
        metrics["avg_graph_degree"] = float(np.mean(degree_vals))
    if topk_count_vals:
        metrics["avg_signal_topk_count"] = float(np.mean(topk_count_vals))
    if hasattr(model, "graph_res_scale"):
        metrics["graph_res_scale"] = float(model.graph_res_scale.detach().cpu())
    metrics["recording_ids"] = recording_ids
    return metrics


def save_epoch_csv(path: Path, rows: list):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def train_sn_expd_warmup5_head(head_name: str, frontend_name: str, cache_payloads: dict, cache_paths: dict, args, output_dir: Path, device: torch.device):
    train_set = DynamicEmbeddingBagDataset(cache_payloads["train"], args.clips_per_recording, train=True, seed=args.seed)
    val_set = DynamicEmbeddingBagDataset(cache_payloads["val"], args.clips_per_recording, train=False, seed=args.seed)
    test_set = DynamicEmbeddingBagDataset(cache_payloads["test"], args.clips_per_recording, train=False, seed=args.seed)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory)
    in_dim = int(train_set.features.size(-1))
    num_classes = int(torch.cat([train_set.labels, val_set.labels, test_set.labels]).max().item()) + 1
    model = build_head(head_name, in_dim, num_classes, args).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)

    head_dir = output_dir / head_name
    head_dir.mkdir(parents=True, exist_ok=True)
    best_path = head_dir / "best_head.pt"
    best_val = -1.0
    best_epoch = -1
    stale = 0
    rows = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, args, optimizer=optimizer, epoch=epoch)
        val_metrics = collect_multisample_embedding_predictions(model, val_set, criterion, device, args)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["total_loss"],
            "train_task_loss": train_metrics["task_loss"],
            "train_orth_loss": train_metrics["orth_loss"],
            "train_noise_consistency_loss": train_metrics["noise_consistency_loss"],
            "train_acc": train_metrics["ACC"],
            "train_macro_f1": train_metrics["Macro-F1"],
            "val_loss": val_metrics["total_loss"],
            "val_task_loss": val_metrics["task_loss"],
            "val_orth_loss": val_metrics["orth_loss"],
            "val_noise_consistency_loss": val_metrics["noise_consistency_loss"],
            "val_acc": val_metrics["ACC"],
            "val_macro_f1": val_metrics["Macro-F1"],
            "val_macro_precision": val_metrics["Macro-Precision"],
            "val_macro_recall": val_metrics["Macro-Recall"],
        }
        for metric_key in ["attn_entropy", "graph_delta_norm", "avg_graph_degree", "avg_signal_topk_count", "graph_res_scale"]:
            if metric_key in train_metrics:
                row[f"train_{metric_key}"] = train_metrics[metric_key]
            if metric_key in val_metrics:
                row[f"val_{metric_key}"] = val_metrics[metric_key]
        rows.append(row)
        print(
            f"{head_name} epoch={epoch} train_loss={row['train_loss']:.6f} "
            f"val_acc={row['val_acc']:.6f} val_macro_f1={row['val_macro_f1']:.6f} "
            f"val_avg_degree={row.get('val_avg_graph_degree', float('nan')):.3f} "
            f"val_topk={row.get('val_avg_signal_topk_count', float('nan')):.3f}",
            flush=True,
        )
        if math.isfinite(val_metrics["Macro-F1"]) and val_metrics["Macro-F1"] > best_val:
            best_val = val_metrics["Macro-F1"]
            best_epoch = epoch
            stale = 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "best_val_macro_f1": best_val, "args": vars(args)}, best_path)
        else:
            stale += 1
            if args.patience > 0 and stale >= args.patience:
                print(f"{head_name} early stopping at epoch {epoch}", flush=True)
                break

    save_epoch_csv(head_dir / "epoch_metrics.csv", rows)
    if best_path.exists():
        state = torch_load(best_path, map_location=device)
        model.load_state_dict(state["model_state"])
    test_metrics = collect_multisample_embedding_predictions(model, test_set, criterion, device, args)
    param_summary = {
        "total_params": int(sum(p.numel() for p in model.parameters())),
        "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    }
    payload = {
        "frontend": frontend_name,
        "head": head_name,
        "head_type": "signal_noise_expd_warmup5",
        "dataset": args.dataset,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val,
        "best_checkpoint_selection": "validation_recording_macro_f1",
        "clips_per_recording": int(args.clips_per_recording),
        "eval_samples": int(args.eval_samples),
        "embed_dim": in_dim,
        "embed_dim_raw": in_dim,
        "adapter_dim": None,
        "edge_mode": EXP_D_EDGE_MODE,
        "graph_k": int(args.graph_k),
        "sim_threshold": float(EXP_D_SIM_THRESHOLD),
        "signal_top_k": int(EXP_D_SIGNAL_TOP_K),
        "topk_warmup_epochs": int(EXP_D_TOPK_WARMUP_EPOCHS),
        "use_temperature": False,
        "lambda_orth": float(args.lambda_orth),
        "lambda_noise_consistency": float(args.lambda_noise_consistency),
        "optimizer": "Adam",
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "grad_clip": float(args.grad_clip),
        "sampling_protocol": "train_random_eval_deterministic_multisample_mean_logits",
        "num_classes": num_classes,
        "train_recordings": len(train_set),
        "val_recordings": len(val_set),
        "test_recordings": len(test_set),
        "cache_paths": {k: str(v) for k, v in cache_paths.items()},
        "test_acc": test_metrics["ACC"],
        "test_macro_f1": test_metrics["Macro-F1"],
        "test_macro_precision": test_metrics["Macro-Precision"],
        "test_macro_recall": test_metrics["Macro-Recall"],
        "test_weighted_f1": test_metrics["Weighted-F1"],
        "test_loss": test_metrics["total_loss"],
        "test_task_loss": test_metrics["task_loss"],
        "test_orth_loss": test_metrics["orth_loss"],
        "test_noise_consistency_loss": test_metrics["noise_consistency_loss"],
        **param_summary,
    }
    for metric_key in ["attn_entropy", "graph_delta_norm", "avg_graph_degree", "avg_signal_topk_count", "graph_res_scale"]:
        if metric_key in test_metrics:
            payload[metric_key] = test_metrics[metric_key]
    write_json(head_dir / "metrics.json", payload)
    with (head_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(payload.keys()))
        writer.writeheader()
        writer.writerow(payload)
    return payload


def train_one_head(head_name: str, frontend_name: str, cache_payloads: dict, cache_paths: dict, args, output_dir: Path, device: torch.device):
    if head_name == "sn_expd_warmup5":
        return train_sn_expd_warmup5_head(head_name, frontend_name, cache_payloads, cache_paths, args, output_dir, device)

    train_set = EmbeddingBagDataset(cache_payloads["train"])
    val_set = EmbeddingBagDataset(cache_payloads["val"])
    test_set = EmbeddingBagDataset(cache_payloads["test"])
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)
    in_dim = int(train_set.features.size(-1))
    num_classes = int(torch.cat([train_set.labels, val_set.labels, test_set.labels]).max().item()) + 1
    model = build_head(head_name, in_dim, num_classes, args).to(device)
    criterion = nn.NLLLoss() if head_name == "mil_linear_softmax" else nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)

    head_dir = output_dir / head_name
    head_dir.mkdir(parents=True, exist_ok=True)
    best_path = head_dir / "best_head.pt"
    best_val = -1.0
    best_epoch = -1
    stale = 0
    rows = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, args, optimizer=optimizer, epoch=epoch)
        val_metrics = run_epoch(model, val_loader, criterion, device, args, optimizer=None)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["total_loss"],
            "train_acc": train_metrics["ACC"],
            "train_macro_f1": train_metrics["Macro-F1"],
            "val_loss": val_metrics["total_loss"],
            "val_acc": val_metrics["ACC"],
            "val_macro_f1": val_metrics["Macro-F1"],
            "val_macro_precision": val_metrics["Macro-Precision"],
            "val_macro_recall": val_metrics["Macro-Recall"],
        }
        for metric_key in [
            "attn_entropy",
            "graph_delta_norm",
            "avg_graph_degree",
            "avg_signal_topk_count",
            "graph_res_scale",
            "learned_temperature",
            "learned_noise_temperature",
        ]:
            if metric_key in train_metrics:
                row[f"train_{metric_key}"] = train_metrics[metric_key]
            if metric_key in val_metrics:
                row[f"val_{metric_key}"] = val_metrics[metric_key]
        rows.append(row)
        print(f"{head_name} epoch={epoch} train_loss={row['train_loss']:.6f} val_acc={row['val_acc']:.6f} val_macro_f1={row['val_macro_f1']:.6f}", flush=True)
        if math.isfinite(val_metrics["Macro-F1"]) and val_metrics["Macro-F1"] > best_val:
            best_val = val_metrics["Macro-F1"]
            best_epoch = epoch
            stale = 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "best_val_macro_f1": best_val, "args": vars(args)}, best_path)
        else:
            stale += 1
            if args.patience > 0 and stale >= args.patience:
                print(f"{head_name} early stopping at epoch {epoch}", flush=True)
                break
    save_epoch_csv(head_dir / "epoch_metrics.csv", rows)
    if best_path.exists():
        state = torch_load(best_path, map_location=device)
        model.load_state_dict(state["model_state"])
    test_metrics = run_epoch(model, test_loader, criterion, device, args, optimizer=None)
    param_summary = {
        "total_params": int(sum(p.numel() for p in model.parameters())),
        "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    }
    payload = {
        "frontend": frontend_name,
        "head": head_name,
        "dataset": args.dataset,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val,
        "clips_per_recording": int(args.clips_per_recording),
        "embed_dim": in_dim,
        "embed_dim_raw": in_dim,
        "adapter_dim": int(args.adapter_dim),
        "edge_mode": args.edge_mode if head_name == "sn_decoupled" else None,
        "graph_k": int(args.graph_k) if head_name == "sn_decoupled" else None,
        "sim_threshold": float(args.sim_threshold) if head_name == "sn_decoupled" else None,
        "signal_top_k": int(args.signal_top_k) if head_name == "sn_decoupled" else None,
        "topk_warmup_epochs": int(args.topk_warmup_epochs) if head_name == "sn_decoupled" else None,
        "use_temperature": bool(args.use_temperature) if head_name == "sn_decoupled" else None,
        "num_classes": num_classes,
        "train_recordings": len(train_set),
        "val_recordings": len(val_set),
        "test_recordings": len(test_set),
        "cache_paths": {k: str(v) for k, v in cache_paths.items()},
        "test_acc": test_metrics["ACC"],
        "test_macro_f1": test_metrics["Macro-F1"],
        "test_macro_precision": test_metrics["Macro-Precision"],
        "test_macro_recall": test_metrics["Macro-Recall"],
        "test_weighted_f1": test_metrics["Weighted-F1"],
        "test_loss": test_metrics["total_loss"],
        **param_summary,
    }
    for metric_key in [
        "attn_entropy",
        "graph_delta_norm",
        "avg_graph_degree",
        "avg_signal_topk_count",
        "graph_res_scale",
        "learned_temperature",
        "learned_noise_temperature",
    ]:
        if metric_key in test_metrics:
            payload[metric_key] = test_metrics[metric_key]
    write_json(head_dir / "metrics.json", payload)
    with (head_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(payload.keys()))
        writer.writeheader()
        writer.writerow(payload)
    return payload


def parse_list_arg(value: str, choices: list):
    if value == "all":
        return list(choices)
    out = [item.strip() for item in str(value).split(",") if item.strip()]
    unknown = [item for item in out if item not in choices]
    if unknown:
        raise ValueError(f"Unsupported values {unknown}; choices={choices} or all")
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-front-end recording-level robustness experiments.")
    parser.add_argument("--model_config", required=True, help="ShuffleFAC model_config.json containing strict recording-level cache_paths.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset", default="auto")
    parser.add_argument("--frontend", default="shufflefac", help="One of shufflefac/resnet18/mobilenet_v2/panns_cnn14, all, or a comma list.")
    parser.add_argument("--head", default="attention", help="One of mean/attention/sn_decoupled/sn_expd_warmup5/bigru/mil_linear_softmax, all, or a comma list.")
    parser.add_argument("--frontend_ckpt", default=None, help="Optional front-end checkpoint. Required for resnet18/mobilenet_v2 paper experiments.")
    parser.add_argument("--allow_random_frontend_for_smoke_test", action="store_true", help="Allow random frozen resnet18/mobilenet_v2 only for explicit smoke tests.")
    parser.add_argument("--debug_waveform_entry", action="store_true")
    parser.add_argument("--debug_split", default="train")
    parser.add_argument("--panns_sample_rate", type=int, default=16000)
    parser.add_argument("--segment_length", type=float, default=3.0)
    parser.add_argument("--panns_repo_dir", default=None)
    parser.add_argument("--panns_ckpt", default=None)
    parser.add_argument("--debug_panns_cache", action="store_true")
    parser.add_argument("--smoke_max_recordings_per_split", type=int, default=0)
    parser.add_argument("--embedding_cache_dir", default=None)
    parser.add_argument("--rebuild_embedding_cache", action="store_true")
    parser.add_argument("--clips_per_recording", type=int, default=8)
    parser.add_argument("--frontend_batch_size", type=int, default=128)
    parser.add_argument("--frontend_embed_dim", type=int, default=None, help="Projection size for mobilenet_v2/panns_cnn14; ignored by shufflefac/resnet18.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--eval_samples", type=int, default=5, help="Deterministic multi-sample passes for sn_expd_warmup5 validation/test.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--sig_dim", type=int, default=32)
    parser.add_argument("--noise_dim", type=int, default=32)
    parser.add_argument("--graph_k", type=int, default=2)
    parser.add_argument(
        "--edge_mode",
        choices=["temporal", "similarity", "temporal_similarity", "threshold_similarity"],
        default="similarity",
        help="SN-Decoupled noise graph mode. Default preserves the old robustness-script behavior.",
    )
    parser.add_argument("--sim_threshold", type=float, default=0.8, help="Cosine threshold for SN threshold_similarity graph mode.")
    parser.add_argument("--signal_top_k", type=int, default=0, help="SN signal Top-K sparse pooling; 0 keeps full soft attention.")
    parser.add_argument("--topk_warmup_epochs", type=int, default=0, help="SN epochs using full soft attention before Top-K masking.")
    parser.add_argument("--use_temperature", action="store_true", help="Enable learnable temperature for SN noise and signal attention.")
    parser.add_argument("--lambda_orth", type=float, default=0.1)
    parser.add_argument("--lambda_noise_consistency", type=float, default=0.1)
    parser.add_argument("--gru_hidden_dim", type=int, default=64)
    parser.add_argument("--adapter_dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def run_frontend(frontend_name: str, head_names: list, args, root: Path, model_config_path: Path, config: dict, output_root: Path, device: torch.device):
    frontend_dir = output_root / frontend_name
    frontend_dir.mkdir(parents=True, exist_ok=True)
    frontend, embed_dim, frontend_ckpt = build_frontend(frontend_name, args, config, model_config_path, root, device)
    source_cache_paths = resolve_cache_paths(config, root)
    cache_payloads = {}
    embedding_cache_paths = {}
    for split in SPLITS:
        if frontend_name == "panns_cnn14":
            payload, path, reused = encode_recording_cache_panns(
                split,
                source_cache_paths[split],
                frontend,
                frontend_name,
                frontend_ckpt,
                embed_dim,
                args,
                device,
                root,
            )
        else:
            payload, path, reused = encode_recording_cache(
                split,
                source_cache_paths[split],
                frontend,
                frontend_name,
                frontend_ckpt,
                embed_dim,
                args,
                device,
            )
        cache_payloads[split] = payload
        embedding_cache_paths[split] = path
        print(f"{frontend_name} {split} embedding cache: {path} ({'reused' if reused else 'created'})", flush=True)
    overlaps = validate_recording_overlap(cache_payloads)
    frontend_checkpoint = getattr(frontend, "checkpoint", None)
    frontend_summary = {
        "frontend": frontend_name,
        "frontend_source": getattr(frontend, "frontend_source", "shufflefac" if frontend_name == "shufflefac" else "script_local"),
        "frontend_ckpt": frontend_ckpt,
        "frontend_ckpt_metadata": checkpoint_cache_metadata(frontend_ckpt, frontend_checkpoint),
        "frontend_best_epoch": frontend_checkpoint.get("best_epoch") if isinstance(frontend_checkpoint, dict) else None,
        "frontend_best_val_recording_macro_f1": frontend_checkpoint.get("best_val_recording_macro_f1") if isinstance(frontend_checkpoint, dict) else None,
        "frontend_best_val_clip_macro_f1": frontend_checkpoint.get("best_val_clip_macro_f1") if isinstance(frontend_checkpoint, dict) else None,
        "embed_dim": int(cache_payloads["train"]["features"].size(-1)),
        "embed_dim_raw": int(cache_payloads["train"]["features"].size(-1)),
        "adapter_dim": int(args.adapter_dim),
        "input_shape": getattr(frontend, "input_shape", None),
        "panns_repo_dir": getattr(frontend, "panns_repo_dir", None),
        "panns_ckpt": getattr(frontend, "panns_ckpt", None),
        "input_type": getattr(frontend, "input_type", None),
        "sample_rate": getattr(frontend, "sample_rate", None),
        "segment_length": float(args.segment_length) if getattr(frontend, "input_type", None) == "waveform" else None,
        "pretraining": getattr(frontend, "pretraining", "none"),
        "weights": getattr(frontend, "weights", "trained_checkpoint" if frontend_ckpt else "random_smoke_test"),
        "clips_per_recording": int(args.clips_per_recording),
        "source_cache_paths": {k: str(v) for k, v in source_cache_paths.items()},
        "embedding_cache_paths": {k: str(v) for k, v in embedding_cache_paths.items()},
        "recordings": {k: int(cache_payloads[k]["labels"].numel()) for k in SPLITS},
        "recording_overlap": overlaps,
        "cache_schema": EMBEDDING_CACHE_SCHEMA,
        "smoke_max_recordings_per_split": int(args.smoke_max_recordings_per_split),
    }
    write_json(frontend_dir / "frontend_cache_summary.json", frontend_summary)
    rows = []
    for head_name in head_names:
        print(f"Running frontend={frontend_name} head={head_name}", flush=True)
        rows.append(train_one_head(head_name, frontend_name, cache_payloads, embedding_cache_paths, args, frontend_dir, device))
    return rows


def write_summary(output_dir: Path, rows: list):
    write_json(output_dir / "summary.json", {"runs": rows})
    if rows:
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


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
    if args.debug_waveform_entry:
        debug_first_waveform_segment(
            model_config_path,
            args.debug_split,
            root,
            target_sample_rate=args.panns_sample_rate,
            segment_length=args.segment_length,
        )
        return
    if args.debug_panns_cache:
        source_cache_paths = resolve_cache_paths(config, root)
        if args.debug_split not in source_cache_paths:
            raise ValueError(f"Unknown split {args.debug_split!r}; available={sorted(source_cache_paths)}")
        repo_dir, ckpt_path = resolve_panns_repo_and_ckpt(args, root)
        frontend = OfficialPannsCnn14_16kFrontend(
            repo_dir=repo_dir,
            ckpt_path=ckpt_path,
            device=device,
            sample_rate=args.panns_sample_rate,
        )
        payload, cache_path, reused = encode_recording_cache_panns(
            args.debug_split,
            source_cache_paths[args.debug_split],
            frontend,
            "panns_cnn14",
            str(ckpt_path),
            int(frontend.embed_dim),
            args,
            device,
            root,
        )
        summary = {
            "split": args.debug_split,
            "cache_path": str(cache_path),
            "reused": bool(reused),
            "features_shape": list(payload["features"].shape),
            "labels_shape": list(payload["labels"].shape),
            "embed_dim": int(payload["embed_dim"]),
            "input_type": payload["input_type"],
            "sample_rate": int(payload["sample_rate"]),
            "segment_length": float(payload["segment_length"]),
            "panns_ckpt": str(ckpt_path),
            "panns_repo_dir": str(repo_dir),
            "smoke_max_recordings_per_split": int(args.smoke_max_recordings_per_split),
        }
        print(f"PANNs embedding cache: {cache_path} ({'reused' if reused else 'created'})", flush=True)
        print(f"features shape: {tuple(payload['features'].shape)}", flush=True)
        print(f"labels shape: {tuple(payload['labels'].shape)}", flush=True)
        print(f"recordings: {int(payload['labels'].numel())}", flush=True)
        print(f"first recording_ids: {payload.get('recording_ids', [])[:5]}", flush=True)
        write_json(output_dir / "debug_panns_cache_summary.json", summary)
        return
    frontend_names = parse_list_arg(args.frontend, FRONTENDS)
    head_names = parse_list_arg(args.head, HEADS)

    all_rows = []
    for frontend_name in frontend_names:
        all_rows.extend(run_frontend(frontend_name, head_names, args, root, model_config_path, config, output_dir, device))
    write_summary(output_dir, all_rows)
    print(json.dumps({"runs": all_rows}, indent=2), flush=True)


if __name__ == "__main__":
    main()
