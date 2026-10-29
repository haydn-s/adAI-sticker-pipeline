"""Loss terms: the attack objective plus smoothness and printability penalties."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def target_loss(logits: torch.Tensor, target: int, kind: str = "ce", margin: float = 5.0) -> torch.Tensor:
    """Push every simulated photo toward the target class.

    ``ce``     — cross-entropy toward the target (Brown et al., 2017).
    ``margin`` — Carlini-Wagner style logit margin, capped at ``margin`` so easy
                 photos stop contributing and the optimizer focuses on hard ones.
    """
    labels = torch.full((logits.shape[0],), target, device=logits.device, dtype=torch.long)
    if kind == "ce":
        return F.cross_entropy(logits, labels)
    if kind == "margin":
        target_logit = logits[:, target]
        others = logits.clone()
        others[:, target] = float("-inf")
        return (others.max(1).values - target_logit).clamp_min(-margin).mean()
    raise ValueError(f"Unknown loss '{kind}'.")


def total_variation(img: torch.Tensor, weight_map: torch.Tensor | None = None) -> torch.Tensor:
    """Mean absolute difference between neighboring pixels (lower = smoother, more printable)."""
    dx = (img[..., :, 1:] - img[..., :, :-1]).abs()
    dy = (img[..., 1:, :] - img[..., :-1, :]).abs()
    if weight_map is not None:
        dx = dx * weight_map[..., :, 1:]
        dy = dy * weight_map[..., 1:, :]
        denom = weight_map.sum().clamp_min(1.0) * img.shape[1] * 2
        return (dx.sum() + dy.sum()) / denom
    return dx.mean() + dy.mean()


def non_printability(img: torch.Tensor, printable: torch.Tensor, weight_map: torch.Tensor | None = None) -> torch.Tensor:
    """Distance from each pixel to the nearest printable color (Sharif et al., 2016)."""
    pixels = img.permute(0, 2, 3, 1).reshape(-1, 3)
    dist = torch.cdist(pixels, printable.to(pixels)).min(1).values
    if weight_map is not None:
        w = weight_map.expand(img.shape[0], 1, -1, -1).reshape(-1)
        return (dist * w).sum() / w.sum().clamp_min(1.0)
    return dist.mean()
