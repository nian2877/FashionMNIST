"""Folder-based dataset utilities for the Day 5 transfer-learning task."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


class DatasetStructureError(RuntimeError):
    """Raised when the real-image dataset folder structure is incomplete."""


@dataclass(frozen=True)
class ImageSample:
    """One image file and its integer label."""

    path: Path
    label: int


def _import_torch():
    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for Day 5 transfer learning.") from exc
    return torch, DataLoader, Dataset


def list_class_names(split_dir: Path) -> list[str]:
    """Return sorted class-folder names inside one split directory."""
    if not split_dir.exists():
        raise DatasetStructureError(
            f"Missing split directory: {split_dir}\n"
            "Expected a folder structure like:\n"
            "data/transfer_real/train/class_name/*.jpg\n"
            "data/transfer_real/val/class_name/*.jpg"
        )

    class_names = sorted(
        child.name for child in split_dir.iterdir() if child.is_dir()
    )
    if not class_names:
        raise DatasetStructureError(
            f"No class folders were found under: {split_dir}"
        )
    return class_names


def make_samples(split_dir: Path, class_names: list[str]) -> list[ImageSample]:
    """Build a flat sample list from an ImageFolder-like directory."""
    valid_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    samples: list[ImageSample] = []

    for label, class_name in enumerate(class_names):
        class_dir = split_dir / class_name
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in valid_suffixes:
                samples.append(ImageSample(path=path, label=label))

    if not samples:
        raise DatasetStructureError(
            f"No image files were found under: {split_dir}"
        )
    return samples


class SimpleImageFolderDataset:
    """A tiny reimplementation of ImageFolder using only PIL + PyTorch.

    We avoid torchvision here because the current environment does not provide it.
    """

    def __init__(
        self,
        samples: list[ImageSample],
        image_size: int,
        train_mode: bool,
        random_seed: int,
        normalize_mean: tuple[float, float, float] | None = None,
        normalize_std: tuple[float, float, float] | None = None,
    ) -> None:
        torch, _, _ = _import_torch()
        self.samples = samples
        self.image_size = image_size
        self.train_mode = train_mode
        self.random_seed = random_seed
        self._torch = torch
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]

        with Image.open(sample.path) as image:
            image = image.convert("RGB")
            image = image.resize((self.image_size, self.image_size))

            # A tiny bit of augmentation keeps the Day 5 example realistic.
            if self.train_mode:
                random.seed(self.random_seed + index)
                if random.random() < 0.5:
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)

            image_bytes = image.tobytes()

        tensor = self._torch.frombuffer(
            bytearray(image_bytes),
            dtype=self._torch.uint8,
        ).clone()
        tensor = tensor.view(self.image_size, self.image_size, 3)
        tensor = tensor.permute(2, 0, 1).float() / 255.0

        if self.normalize_mean is not None and self.normalize_std is not None:
            mean_tensor = self._torch.tensor(self.normalize_mean).view(3, 1, 1)
            std_tensor = self._torch.tensor(self.normalize_std).view(3, 1, 1)
            tensor = (tensor - mean_tensor) / std_tensor

        label = self._torch.tensor(sample.label, dtype=self._torch.long)
        return tensor, label


def create_transfer_dataloaders(
    root: str | Path,
    image_size: int,
    batch_size: int,
    random_seed: int,
    normalize_mean: tuple[float, float, float] | None = None,
    normalize_std: tuple[float, float, float] | None = None,
):
    """Create train/val dataloaders for a folder-based real image dataset."""
    _, DataLoader, _ = _import_torch()
    root = Path(root)

    train_dir = root / "train"
    val_dir = root / "val"
    if not val_dir.exists():
        val_dir = root / "test"

    class_names = list_class_names(train_dir)
    val_class_names = list_class_names(val_dir)
    if class_names != val_class_names:
        raise DatasetStructureError(
            "Train and val/test splits must contain the same class folders."
        )

    train_samples = make_samples(train_dir, class_names)
    val_samples = make_samples(val_dir, class_names)

    train_dataset = SimpleImageFolderDataset(
        samples=train_samples,
        image_size=image_size,
        train_mode=True,
        random_seed=random_seed,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
    )
    val_dataset = SimpleImageFolderDataset(
        samples=val_samples,
        image_size=image_size,
        train_mode=False,
        random_seed=random_seed,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, val_loader, class_names
