"""Victim models behind one interface: images in [0, 1] -> logits over a label set.

Two families are supported:

* ImageNet classifiers from torchvision (CNNs and a vision transformer).
* CLIP zero-shot classifiers from open_clip, scored against the ImageNet label
  set plus any free-text targets the user adds.

Every wrapper normalizes its own inputs, so the optimizer and evaluator only ever
deal with plain RGB tensors in [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .labels import Target, imagenet_classes

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ModelInfo:
    """Plain-language model facts for reports and user interfaces."""

    key: str
    display_name: str
    family: str  # "imagenet" or "clip"
    year: int
    params_millions: float
    description: str


MODEL_INFO: dict[str, ModelInfo] = {
    "resnet50": ModelInfo(
        "resnet50", "ResNet-50", "imagenet", 2015, 25.6,
        "A classic convolutional network and one of the most widely deployed image "
        "recognizers; the standard model for adversarial-patch research.",
    ),
    "mobilenet_v3_large": ModelInfo(
        "mobilenet_v3_large", "MobileNet V3", "imagenet", 2019, 5.5,
        "A small, fast network designed to run on phones and embedded cameras.",
    ),
    "efficientnet_b0": ModelInfo(
        "efficientnet_b0", "EfficientNet-B0", "imagenet", 2019, 5.3,
        "A compact network found by automated architecture search, common in mobile apps.",
    ),
    "convnext_tiny": ModelInfo(
        "convnext_tiny", "ConvNeXt-Tiny", "imagenet", 2022, 28.6,
        "A modernized convolutional network that borrows design ideas from transformers.",
    ),
    "vit_b_16": ModelInfo(
        "vit_b_16", "Vision Transformer (ViT-B/16)", "imagenet", 2020, 86.6,
        "A transformer that reads the image as a sequence of 16x16 pixel tiles.",
    ),
    "clip-vit-b-32": ModelInfo(
        "clip-vit-b-32", "CLIP (ViT-B/32)", "clip", 2021, 151.3,
        "OpenAI's image-text model. It labels images by comparing them with text "
        "descriptions, so it can recognize labels it was never explicitly trained on.",
    ),
}

# (torchvision constructor, weights enum name, Grad-CAM layer attribute path)
_TORCHVISION_SPECS: dict[str, tuple[str, str, str | None]] = {
    "resnet50": ("resnet50", "IMAGENET1K_V2", "layer4"),
    "mobilenet_v3_large": ("mobilenet_v3_large", "IMAGENET1K_V2", "features"),
    "efficientnet_b0": ("efficientnet_b0", "IMAGENET1K_V1", "features"),
    "convnext_tiny": ("convnext_tiny", "IMAGENET1K_V1", "features"),
    "vit_b_16": ("vit_b_16", "IMAGENET1K_V1", None),
}

_CLIP_SPECS: dict[str, tuple[str, str]] = {
    "clip-vit-b-32": ("ViT-B-32", "openai"),
}

# Prompt templates averaged for CLIP zero-shot classification (subset of the
# templates used by Radford et al., 2021).
CLIP_TEMPLATES: tuple[str, ...] = (
    "a photo of a {}.",
    "a blurry photo of a {}.",
    "a close-up photo of a {}.",
    "a bright photo of a {}.",
    "a photo of the small {}.",
    "a photo of the large {}.",
    "a cropped photo of a {}.",
)


def available_models() -> list[str]:
    return list(MODEL_INFO)


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class VictimModel(nn.Module):
    """Common interface for all attacked models."""

    key: str
    input_size: int = 224

    @property
    def info(self) -> ModelInfo:
        return MODEL_INFO[self.key]

    @property
    def class_names(self) -> list[str]:
        raise NotImplementedError

    @property
    def gradcam_layer(self) -> nn.Module | None:
        """Last convolutional block for Grad-CAM, or None for transformers."""
        return None

    def class_index(self, target: Target) -> int:
        """Index of ``target`` in this model's output logits."""
        if target.imagenet_index is not None:
            return target.imagenet_index
        raise ValueError(f"{self.key} only supports ImageNet targets, not free text.")

    def forward(self, images: torch.Tensor) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def _resize(self, images: torch.Tensor) -> torch.Tensor:
        if images.shape[-1] != self.input_size or images.shape[-2] != self.input_size:
            images = F.interpolate(
                images, size=(self.input_size, self.input_size), mode="bilinear",
                align_corners=False, antialias=True,
            )
        return images


class PatchEmbedding(nn.Module):
    """A vision transformer's patch-embedding convolution, computed as reshape + matmul.

    The result is the same as the convolution it replaces (kernel = stride = patch size).
    The point is speed: on Apple's MPS backend that convolution's input gradient is
    pathologically slow. One CLIP ViT-B/32 training step took 23 s, versus 0.006 s for
    this version of the layer.
    """

    def __init__(self, conv: nn.Conv2d):
        super().__init__()
        self.patch = conv.kernel_size[0]
        self.weight = conv.weight  # (dim, channels, patch, patch); same state-dict keys
        self.bias = conv.bias

    @staticmethod
    def replaces(conv: nn.Module) -> bool:
        return (isinstance(conv, nn.Conv2d) and conv.kernel_size == conv.stride
                and conv.kernel_size[0] == conv.kernel_size[1] and conv.padding == (0, 0)
                and conv.dilation == (1, 1) and conv.groups == 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        p = self.patch
        gh, gw = h // p, w // p
        x = x[:, :, :gh * p, :gw * p]  # the strided convolution ignores any remainder
        patches = x.reshape(b, c, gh, p, gw, p).permute(0, 2, 4, 1, 3, 5).reshape(b, gh * gw, -1)
        out = patches @ self.weight.reshape(self.weight.shape[0], -1).t()
        if self.bias is not None:
            out = out + self.bias
        return out.transpose(1, 2).reshape(b, -1, gh, gw)


def _use_fast_patch_embedding(module: nn.Module, attr: str) -> None:
    conv = getattr(module, attr, None)
    if PatchEmbedding.replaces(conv):
        setattr(module, attr, PatchEmbedding(conv))


class TorchvisionClassifier(VictimModel):
    def __init__(self, key: str):
        super().__init__()
        import torchvision.models as tvm

        ctor_name, weights_name, cam_attr = _TORCHVISION_SPECS[key]
        weights = tvm.get_model_weights(ctor_name)[weights_name]
        self.key = key
        self.net = getattr(tvm, ctor_name)(weights=weights).eval()
        _use_fast_patch_embedding(self.net, "conv_proj")  # vision transformers only
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self._cam_attr = cam_attr
        for p in self.net.parameters():
            p.requires_grad_(False)

    @property
    def class_names(self) -> list[str]:
        return list(imagenet_classes())

    @property
    def gradcam_layer(self) -> nn.Module | None:
        return getattr(self.net, self._cam_attr) if self._cam_attr else None

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = (self._resize(images) - self.mean) / self.std
        return self.net(x)


class CLIPClassifier(VictimModel):
    """Zero-shot CLIP over ImageNet labels plus optional free-text labels."""

    def __init__(self, key: str):
        super().__init__()
        import open_clip

        arch, pretrained = _CLIP_SPECS[key]
        self.key = key
        model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
        self.clip = model.eval()
        _use_fast_patch_embedding(self.clip.visual, "conv1")
        self.tokenizer = open_clip.get_tokenizer(arch)
        for p in self.clip.parameters():
            p.requires_grad_(False)
        normalize = next(t for t in preprocess.transforms if type(t).__name__ == "Normalize")
        self.register_buffer("mean", torch.tensor(normalize.mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(normalize.std).view(1, 3, 1, 1))
        self.input_size = self.clip.visual.image_size[0] if isinstance(
            self.clip.visual.image_size, (tuple, list)) else self.clip.visual.image_size
        self._names: list[str] = list(imagenet_classes())
        self.register_buffer("text_features", torch.empty(0), persistent=False)
        self._text_ready = False

    @property
    def class_names(self) -> list[str]:
        return list(self._names)

    @torch.no_grad()
    def _encode_labels(self, labels: list[str]) -> torch.Tensor:
        device = self.mean.device
        feats = []
        for chunk_start in range(0, len(labels), 100):
            chunk = labels[chunk_start:chunk_start + 100]
            prompts = [t.format(name) for name in chunk for t in CLIP_TEMPLATES]
            tokens = self.tokenizer(prompts).to(device)
            emb = self.clip.encode_text(tokens).float()
            emb = F.normalize(emb, dim=-1).view(len(chunk), len(CLIP_TEMPLATES), -1).mean(1)
            feats.append(F.normalize(emb, dim=-1))
        return torch.cat(feats)

    def _ensure_text(self) -> None:
        if not self._text_ready or self.text_features.device != self.mean.device:
            self.text_features = self._encode_labels(self._names)
            self._text_ready = True

    def add_label(self, text: str) -> int:
        """Add a free-text label to the candidate set (returns its index)."""
        lowered = [n.lower() for n in self._names]
        if text.lower() in lowered:
            return lowered.index(text.lower())
        self._ensure_text()
        self._names.append(text)
        self.text_features = torch.cat([self.text_features, self._encode_labels([text])])
        return len(self._names) - 1

    def class_index(self, target: Target) -> int:
        if target.imagenet_index is not None:
            return target.imagenet_index
        return self.add_label(target.label)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self._ensure_text()
        x = (self._resize(images) - self.mean) / self.std
        img = F.normalize(self.clip.encode_image(x).float(), dim=-1)
        return self.clip.logit_scale.exp() * img @ self.text_features.T


def load_model(key: str, device: torch.device | str | None = None) -> VictimModel:
    """Load a victim model by registry key and move it to ``device``."""
    device = torch.device(device) if device is not None else default_device()
    if key in _TORCHVISION_SPECS:
        model: VictimModel = TorchvisionClassifier(key)
    elif key in _CLIP_SPECS:
        model = CLIPClassifier(key)
    else:
        raise KeyError(f"Unknown model '{key}'. Available: {', '.join(available_models())}")
    return model.to(device).eval()


def supports_free_text(key: str) -> bool:
    return MODEL_INFO[key].family == "clip"
