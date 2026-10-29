"""Shared fixtures: a tiny stand-in classifier so tests run fast and offline."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from adai_sticker.labels import imagenet_classes
from adai_sticker.models import VictimModel


class TinyClassifier(VictimModel):
    """A random 2-layer CNN with 1,000 outputs that mimics the VictimModel interface."""

    def __init__(self, key: str = "resnet50", seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.key = key
        self.features = nn.Sequential(nn.Conv2d(3, 8, 3, stride=4, padding=1), nn.ReLU())
        self.head = nn.Linear(8, 1000)
        for p in self.parameters():
            p.requires_grad_(False)

    @property
    def class_names(self) -> list[str]:
        return list(imagenet_classes())

    @property
    def gradcam_layer(self) -> nn.Module:
        return self.features

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(self._resize(images)).mean((2, 3)))


@pytest.fixture
def tiny_model() -> TinyClassifier:
    return TinyClassifier().eval()


@pytest.fixture
def cover_path(tmp_path) -> str:
    """A smooth synthetic 'photo' to use as a cover image."""
    y, x = np.mgrid[0:96, 0:128]
    img = np.stack([x / 128, y / 96, 0.5 + 0.5 * np.sin(x / 9.0)], -1)
    path = tmp_path / "cover.png"
    Image.fromarray((img * 255).astype(np.uint8)).save(path)
    return str(path)
