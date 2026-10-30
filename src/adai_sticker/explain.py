"""Explanations: where is the model looking when the sticker fools it?

Grad-CAM (Selvaraju et al., 2017) highlights the image regions that most
increased the score of a class. For convolutional models we use the last
convolutional block. Transformers (ViT, CLIP) have no such feature map, so we
fall back to SmoothGrad-style input-gradient saliency.

The *attention share* turns the picture into a number: the fraction of the
explanation's mass that falls on the sticker, compared with the fraction of the
image the sticker covers. A share far above the area fraction means the sticker
is hijacking the model's attention.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from .models import VictimModel
from .transforms import gaussian_blur


def _class_scores(logits: torch.Tensor, class_idx: int | torch.Tensor) -> torch.Tensor:
    """Score of one class for every image, or of a per-image class (tensor of shape (B,))."""
    if isinstance(class_idx, torch.Tensor):
        return logits.gather(1, class_idx.to(logits.device).view(-1, 1)).squeeze(1)
    return logits[:, class_idx]


def gradcam(model: VictimModel, images: torch.Tensor,
            class_idx: int | torch.Tensor) -> tuple[torch.Tensor, str]:
    """Return (heatmaps (B, H, W) in [0, 1], method name).

    ``class_idx`` is one class for the whole batch or a (B,) tensor of per-image classes.
    """
    if model.gradcam_layer is None:
        return input_saliency(model, images, class_idx), "gradient saliency"

    store: dict[str, torch.Tensor] = {}

    def hook(_module, _inp, out):
        out.retain_grad()
        store["act"] = out

    handle = model.gradcam_layer.register_forward_hook(hook)
    try:
        with torch.enable_grad():
            x = images.detach().clone().requires_grad_(True)
            logits = model(x)
            _class_scores(logits, class_idx).sum().backward()
    finally:
        handle.remove()
    act = store["act"]
    grad = act.grad
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * act).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=images.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    return _normalize(cam.detach()), "Grad-CAM"


def differentiable_gradcam(model: VictimModel, images: torch.Tensor,
                           class_idx: int | torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Grad-CAM that keeps its graph, so an attacker can optimize against the explanation.

    Returns (heatmaps (B, H, W) in [0, 1], logits). ``class_idx=None`` explains each
    image's top-1 class, as the saliency defense does. The channel weights are held
    constant and the gradient flows through the activations. For ResNets, whose last
    block feeds global pooling and one linear layer, the weights don't depend on the
    input anyway, so the heatmap's gradient is exact.
    """
    if model.gradcam_layer is None:
        raise ValueError(f"{model.key} has no convolutional layer for Grad-CAM; attacking "
                         "the explanation is only supported for CNNs.")
    store: dict[str, torch.Tensor] = {}

    def hook(_module, _inp, out):
        store["act"] = out

    if not images.requires_grad:
        images = images.detach().requires_grad_(True)
    handle = model.gradcam_layer.register_forward_hook(hook)
    try:
        with torch.enable_grad():
            logits = model(images)
            if class_idx is None:
                class_idx = logits.detach().argmax(-1)
            (grad,) = torch.autograd.grad(_class_scores(logits, class_idx).sum(), store["act"],
                                          retain_graph=True)
            weights = grad.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * store["act"]).sum(1, keepdim=True))
            cam = F.interpolate(cam, size=images.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
            return _normalize(cam), logits
    finally:
        handle.remove()


def input_saliency(model: VictimModel, images: torch.Tensor, class_idx: int | torch.Tensor,
                   samples: int = 8, noise: float = 0.05) -> torch.Tensor:
    """SmoothGrad: average |d score / d pixel| over noisy copies, then blur."""
    total = torch.zeros(images.shape[0], images.shape[-2], images.shape[-1], device=images.device)
    with torch.enable_grad():
        for _ in range(samples):
            x = (images.detach() + noise * torch.randn_like(images)).clamp(0, 1).requires_grad_(True)
            _class_scores(model(x), class_idx).sum().backward()
            total += x.grad.abs().amax(1)
    sal = gaussian_blur(total[:, None], 4.0)[:, 0]
    return _normalize(sal)


def _normalize(maps: torch.Tensor) -> torch.Tensor:
    flat = maps.flatten(1)
    lo = flat.min(1).values[:, None, None]
    hi = flat.max(1).values[:, None, None]
    return (maps - lo) / (hi - lo).clamp_min(1e-8)


@dataclass
class AttentionShare:
    share_on_sticker: float  # fraction of explanation mass on the sticker
    sticker_area: float  # fraction of the image the sticker covers
    method: str

    @property
    def ratio(self) -> float:
        return self.share_on_sticker / max(self.sticker_area, 1e-8)

    def to_dict(self) -> dict:
        return {**asdict(self), "ratio": self.ratio}


def share_on_sticker(cams: torch.Tensor, scene_alpha: torch.Tensor) -> torch.Tensor:
    """Per-image fraction of explanation mass on the sticker, shape (B,). Differentiable."""
    a = scene_alpha[:, 0] if scene_alpha.dim() == 4 else scene_alpha
    mass = cams.flatten(1).sum(1).clamp_min(1e-8)
    return (cams * a).flatten(1).sum(1) / mass


def attention_share(cams: torch.Tensor, scene_alpha: torch.Tensor, method: str) -> AttentionShare:
    a = scene_alpha[:, 0] if scene_alpha.dim() == 4 else scene_alpha
    on = share_on_sticker(cams, a)
    area = a.flatten(1).mean(1)
    return AttentionShare(float(on.mean()), float(area.mean()), method)
