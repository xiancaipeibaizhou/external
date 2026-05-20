"""TorchVision clip-level mel-spectrogram front-ends.

These models are trained as clip-level classifiers first. Their
``forward_features`` methods are later used to build recording-level embedding
caches for cross-front-end robustness experiments.
"""

import torch.nn as nn
from torchvision import models


class ResNet18MelClassifier(nn.Module):
    """Standard TorchVision ResNet18 adapted to single-channel log-mel input."""

    def __init__(self, num_classes: int, pretrained: str = "none"):
        super().__init__()
        if pretrained != "none":
            raise ValueError("Only pretrained='none' is supported in this task.")
        self.backbone = models.resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(
            1,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.embed_dim = int(self.backbone.fc.in_features)
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(self.embed_dim, int(num_classes))

    def forward_features(self, x):
        """Return clip embeddings with shape [B, 512]."""

        return self.backbone(x)

    def forward(self, x):
        """Return clip-level logits with shape [B, num_classes]."""

        return self.classifier(self.forward_features(x))


class MobileNetV2MelClassifier(nn.Module):
    """Standard TorchVision MobileNetV2 adapted to single-channel log-mel input."""

    def __init__(self, num_classes: int, pretrained: str = "none"):
        super().__init__()
        if pretrained != "none":
            raise ValueError("Only pretrained='none' is supported in this task.")
        self.backbone = models.mobilenet_v2(weights=None)
        old_conv = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            1,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            dilation=old_conv.dilation,
            groups=old_conv.groups,
            bias=False,
        )
        self.embed_dim = int(self.backbone.last_channel)
        self.backbone.classifier = nn.Identity()
        self.classifier = nn.Linear(self.embed_dim, int(num_classes))

    def forward_features(self, x):
        """Return clip embeddings with shape [B, 1280]."""

        return self.backbone(x)

    def forward(self, x):
        """Return clip-level logits with shape [B, num_classes]."""

        return self.classifier(self.forward_features(x))


def build_clip_frontend(frontend: str, num_classes: int, pretrained: str = "none"):
    """Build a TorchVision clip-level classifier and return ``(model, embed_dim)``."""

    if frontend == "resnet18":
        model = ResNet18MelClassifier(num_classes=num_classes, pretrained=pretrained)
    elif frontend == "mobilenet_v2":
        model = MobileNetV2MelClassifier(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported clip front-end: {frontend}")
    return model, int(model.embed_dim)
