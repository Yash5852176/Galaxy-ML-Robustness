"""Robust convolutional neural network for galaxy morphology classification.

This module defines :class:`RobustGalaxyCNN`, a PyTorch model skeleton designed
for classifying galaxy images under noisy and blurred observational conditions.
The architecture uses residual blocks, batch normalisation, and dropout to
improve generalisation to degraded inputs.

Typical usage::

    import torch
    from models.robust_cnn import RobustGalaxyCNN

    model = RobustGalaxyCNN(in_channels=1, num_classes=3)
    x = torch.randn(8, 1, 128, 128)
    logits = model(x)           # shape: (8, 3)
    probs  = model.predict(x)   # softmax probabilities
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class ConvBlock(nn.Module):
    """Conv → BatchNorm → ReLU with optional max-pooling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pool: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(F.relu(self.bn(self.conv(x))))


class ResidualBlock(nn.Module):
    """Two-layer residual block with skip connection."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------


class RobustGalaxyCNN(nn.Module):
    """CNN for galaxy morphology classification with robustness features.

    The network comprises four down-sampling convolutional stages, each
    followed by a residual block, and a fully-connected classification
    head with dropout regularisation.

    Parameters
    ----------
    in_channels : int
        Number of input channels (e.g. 1 for mono, 3 for RGB).  Default 1.
    num_classes : int
        Number of morphology classes.  Default 3.
    base_filters : int
        Number of filters in the first convolutional layer.  Subsequent
        layers double the filter count.  Default 32.
    dropout_rate : float
        Dropout probability applied before the final linear layer.
        Default 0.5.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 3,
        base_filters: int = 32,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()

        f = base_filters  # shorthand

        # --- Encoder stages ---------------------------------------------------
        self.stage1 = nn.Sequential(ConvBlock(in_channels, f), ResidualBlock(f))
        self.stage2 = nn.Sequential(ConvBlock(f, f * 2), ResidualBlock(f * 2))
        self.stage3 = nn.Sequential(ConvBlock(f * 2, f * 4), ResidualBlock(f * 4))
        self.stage4 = nn.Sequential(ConvBlock(f * 4, f * 8), ResidualBlock(f * 8))

        # --- Classification head ---------------------------------------------
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(f * 8, num_classes)

        self._init_weights()

        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "RobustGalaxyCNN initialised – in_ch=%d, classes=%d, params=%s",
            in_channels,
            num_classes,
            f"{total_params:,}",
        )

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Apply Kaiming initialisation to conv layers and Xavier to fc."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class logits for a batch of galaxy images.

        Parameters
        ----------
        x : Tensor, shape (B, C, H, W)
            Batch of input images.

        Returns
        -------
        Tensor, shape (B, num_classes)
            Raw (unnormalised) class scores.
        """
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.global_pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities (inference mode)."""
        self.eval()
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

    def feature_maps(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract intermediate feature maps for visualisation / analysis."""
        features: list[torch.Tensor] = []
        for stage in (self.stage1, self.stage2, self.stage3, self.stage4):
            x = stage(x)
            features.append(x)
        return features
