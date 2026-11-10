"""
Model module — EfficientNet-B0 backbone with dual classification heads
for multi-task breadfruit maturity and quality prediction.
"""

from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models

from config import Config


class MultiTaskHead(nn.Module):
    """A single classification head with dropout for multi-task learning."""

    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class BreadfruitClassifier(nn.Module):
    """EfficientNet-B0 with separate heads for maturity and quality.

    Architecture:
        - Shared EfficientNet-B0 backbone (pretrained on ImageNet)
        - Maturity classification head
        - Quality classification head
        - Optional backbone freezing for initial transfer learning
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.num_maturity = len(config.model.maturity_classes)
        self.num_quality = len(config.model.quality_classes)
        self.has_multi_task = self.num_maturity > 0 and self.num_quality > 0

        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if config.model.pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)

        in_features = self.backbone.classifier[1].in_features

        self.backbone.classifier = nn.Identity()

        self.maturity_head = MultiTaskHead(
            in_features=in_features,
            num_classes=max(self.num_maturity, 1),
            dropout=config.model.dropout,
        )

        if self.has_multi_task:
            self.quality_head = MultiTaskHead(
                in_features=in_features,
                num_classes=self.num_quality,
                dropout=config.model.dropout,
            )

        self._init_heads()

    def _init_heads(self):
        """Initialize classification heads with small weights for stable training."""
        for head_name in ["maturity_head", "quality_head"]:
            head = getattr(self, head_name, None)
            if head is None:
                continue
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.1)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)

    def freeze_backbone(self, freeze: bool = True):
        """Freeze or unfreeze the backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = not freeze

    def forward(self, x: torch.Tensor) -> dict:
        features = self.backbone(x)

        out = {"maturity_logits": self.maturity_head(features)}

        if self.has_multi_task:
            out["quality_logits"] = self.quality_head(features)
        else:
            out["quality_logits"] = out["maturity_logits"]

        return out


class MultiTaskLoss(nn.Module):
    """Weighted sum of cross-entropy losses for maturity and quality tasks."""

    def __init__(self, maturity_weight: float = 0.5, quality_weight: float = 0.5):
        super().__init__()
        self.maturity_weight = maturity_weight
        self.quality_weight = quality_weight
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        predictions: dict,
        targets: dict,
    ) -> Tuple[torch.Tensor, dict]:
        maturity_loss = self.ce_loss(
            predictions["maturity_logits"], targets["maturity"]
        )
        quality_loss = self.ce_loss(
            predictions["quality_logits"], targets["quality"]
        )

        total_loss = (
            self.maturity_weight * maturity_loss
            + self.quality_weight * quality_loss
        )

        loss_dict = {
            "total_loss": total_loss,
            "maturity_loss": maturity_loss,
            "quality_loss": quality_loss,
        }

        return total_loss, loss_dict


def create_model(config: Config) -> BreadfruitClassifier:
    """Factory function for model creation with proper device placement."""
    model = BreadfruitClassifier(config)
    model = model.to(config.device)
    return model


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_feature_extractor(model: BreadfruitClassifier) -> nn.Module:
    """Return the feature extraction submodule (backbone without heads).

    Useful for Grad-CAM and feature visualization.
    """
    return model.backbone
