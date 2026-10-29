"""Background photos the sticker is pasted onto during optimization and evaluation.

The default source is Imagenette (Howard, fast.ai), a freely downloadable
10-class subset of ImageNet. Its *train* split is used for optimization and its
*val* split for evaluation, so reported success rates are measured on photos
the sticker never saw. Any folder of photos can be used instead.
"""

from __future__ import annotations

import hashlib
import os
import random
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

IMAGENETTE_URLS = {
    160: "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz",
    320: "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz",
}

# Imagenette WordNet ids -> ImageNet-1k class index.
IMAGENETTE_WNIDS = {
    "n01440764": 0,  # tench
    "n02102040": 217,  # English springer
    "n02979186": 482,  # cassette player
    "n03000684": 491,  # chain saw
    "n03028079": 497,  # church
    "n03394916": 566,  # French horn
    "n03417042": 569,  # garbage truck
    "n03425413": 571,  # gas pump
    "n03445777": 574,  # golf ball
    "n03888605": 701,  # parachute
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def default_data_dir() -> Path:
    return Path(os.environ.get("ADAI_DATA_DIR", Path.cwd() / "data"))


def fetch_imagenette(data_dir: Path | None = None, size: int = 320) -> Path:
    """Download and extract Imagenette (idempotent). Returns the dataset root."""
    data_dir = data_dir or default_data_dir()
    root = data_dir / f"imagenette2-{size}"
    if (root / "train").exists() and (root / "val").exists():
        return root
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / f"imagenette2-{size}.tgz"
    if not archive.exists():
        print(f"Downloading Imagenette ({size}px) to {archive} ...")
        urllib.request.urlretrieve(IMAGENETTE_URLS[size], archive)
    print(f"Extracting {archive.name} ...")
    with tarfile.open(archive) as tar:
        members = [
            m for m in tar.getmembers()
            if m.name.startswith(f"imagenette2-{size}/") and ".." not in Path(m.name).parts
            and (m.isdir() or m.isfile())
        ]
        tar.extractall(data_dir, members=members, filter="data")
    return root


def find_imagenette(data_dir: Path | None = None) -> Path | None:
    data_dir = data_dir or default_data_dir()
    for size in (320, 160):
        root = data_dir / f"imagenette2-{size}"
        if (root / "train").exists():
            return root
    return None


def _list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def _label_for(path: Path) -> int:
    """ImageNet label from an Imagenette path, or -1 when unknown."""
    for part in reversed(path.parts):
        if part in IMAGENETTE_WNIDS:
            return IMAGENETTE_WNIDS[part]
    return -1


def _load_square(path: Path, size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = size / min(w, h)
    img = img.resize((max(size, round(w * scale)), max(size, round(h * scale))), Image.Resampling.BILINEAR)
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    return np.asarray(img.crop((left, top, left + size, top + size)), dtype=np.uint8)


@dataclass
class BackgroundSet:
    """Background photos held in memory as uint8, with ImageNet labels (-1 if unknown)."""

    images: torch.Tensor  # (N, 3, S, S) uint8
    labels: torch.Tensor  # (N,) int64
    source: str

    def __len__(self) -> int:
        return self.images.shape[0]

    @classmethod
    def from_folder(
        cls, folder: str | Path, image_size: int = 256, limit: int | None = None, seed: int = 0
    ) -> BackgroundSet:
        folder = Path(folder)
        paths = _list_images(folder)
        if not paths:
            raise FileNotFoundError(f"No images found under {folder}.")
        if limit is not None and limit < len(paths):
            paths = random.Random(seed).sample(paths, limit)
        arrays = [_load_square(p, image_size) for p in paths]
        images = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).contiguous()
        labels = torch.tensor([_label_for(p) for p in paths], dtype=torch.long)
        return cls(images, labels, str(folder))

    @classmethod
    def synthetic(cls, n: int = 64, image_size: int = 256, seed: int = 0) -> BackgroundSet:
        """Smooth random color fields — for tests and offline smoke runs only."""
        gen = torch.Generator().manual_seed(seed)
        low = torch.rand(n, 3, 8, 8, generator=gen)
        imgs = torch.nn.functional.interpolate(low, size=(image_size, image_size), mode="bicubic")
        imgs = (imgs.clamp(0, 1) * 255).to(torch.uint8)
        return cls(imgs, torch.full((n,), -1, dtype=torch.long), "synthetic")

    def without_label(self, label: int | None) -> BackgroundSet:
        """Drop photos whose true class is the attack target (they'd count as free wins)."""
        if label is None:
            return self
        keep = self.labels != label
        return BackgroundSet(self.images[keep], self.labels[keep], self.source)

    def fingerprint(self) -> str:
        h = hashlib.sha1(self.images[:, :, ::16, ::16].numpy().tobytes()).hexdigest()
        return f"{self.source}:{len(self)}:{h[:10]}"

    def sample(
        self,
        batch: int,
        out_size: int,
        device: torch.device | str,
        gen: torch.Generator | None = None,
        augment: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Random batch with random crop + flip (augment) -> floats in [0, 1]."""
        idx = torch.randint(len(self), (batch,), generator=gen)
        imgs = self.images[idx]
        s = imgs.shape[-1]
        if augment and s > out_size:
            crops = []
            offs = torch.randint(s - out_size + 1, (batch, 2), generator=gen)
            flips = torch.rand(batch, generator=gen) < 0.5
            for img, (oy, ox), flip in zip(imgs, offs.tolist(), flips.tolist(), strict=True):
                crop = img[:, oy:oy + out_size, ox:ox + out_size]
                crops.append(crop.flip(-1) if flip else crop)
            imgs = torch.stack(crops)
        else:
            imgs = self.center(imgs, out_size)
        return imgs.to(device).float() / 255.0, self.labels[idx].to(device)

    @staticmethod
    def center(imgs: torch.Tensor, out_size: int) -> torch.Tensor:
        s = imgs.shape[-1]
        if s == out_size:
            return imgs
        o = (s - out_size) // 2
        return imgs[:, :, o:o + out_size, o:o + out_size]

    def batches(self, batch: int, out_size: int, device: torch.device | str):
        """Deterministic, center-cropped iteration over all photos (for evaluation)."""
        for start in range(0, len(self), batch):
            imgs = self.center(self.images[start:start + batch], out_size)
            yield imgs.to(device).float() / 255.0, self.labels[start:start + batch].to(device)


def load_backgrounds(
    split: str,
    source: str | Path | None = None,
    limit: int | None = None,
    image_size: int = 256,
    seed: int = 0,
) -> BackgroundSet:
    """Load the train or val backgrounds.

    ``source`` may be a folder of photos (used for both splits), the string
    ``"synthetic"``, or ``None`` to use Imagenette from the data directory.
    """
    if source == "synthetic":
        return BackgroundSet.synthetic(limit or 64, image_size, seed=seed + (split == "val"))
    if source is not None:
        folder = Path(source)
        if (folder / split).exists():
            folder = folder / split
        return BackgroundSet.from_folder(folder, image_size, limit, seed)
    root = find_imagenette()
    if root is None:
        raise FileNotFoundError(
            "No background photos found. Run `adai-sticker fetch-data` to download Imagenette, "
            "or pass --backgrounds <folder>."
        )
    return BackgroundSet.from_folder(root / split, image_size, limit, seed)
