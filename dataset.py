"""Utilities for downloading and loading the FashionMNIST dataset.

This module intentionally uses only the Python standard library plus PyTorch.
That makes it a good teaching example when `torchvision` is unavailable.
"""

from __future__ import annotations

import gzip
import hashlib
import shutil
import struct
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


class MissingTorchDependencyError(RuntimeError):
    """Raised when torch is unavailable in the current Python environment."""


class DatasetDownloadError(RuntimeError):
    """Raised when dataset files are missing and auto-download did not succeed."""


@dataclass(frozen=True)
class FashionMNISTResource:
    """Description of one dataset file."""

    file_name: str
    url: str
    md5: str


CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


RESOURCES = [
    FashionMNISTResource(
        file_name="train-images-idx3-ubyte.gz",
        url="https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/train-images-idx3-ubyte.gz",
        md5="8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    ),
    FashionMNISTResource(
        file_name="train-labels-idx1-ubyte.gz",
        url="https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/train-labels-idx1-ubyte.gz",
        md5="25c81989df183df01b3e8a0aad5dffbe",
    ),
    FashionMNISTResource(
        file_name="t10k-images-idx3-ubyte.gz",
        url="https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/t10k-images-idx3-ubyte.gz",
        md5="bef4ecab320f06d8554ea6380940ec79",
    ),
    FashionMNISTResource(
        file_name="t10k-labels-idx1-ubyte.gz",
        url="https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/t10k-labels-idx1-ubyte.gz",
        md5="bb300cfdad3c16e7a12a480ee83cd310",
    ),
]


def _import_torch():
    """Import torch lazily so this file can still be opened without torch installed."""
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise MissingTorchDependencyError(
            "PyTorch is not installed in this Python environment."
        ) from exc
    return torch, DataLoader, TensorDataset


def _md5(path: Path) -> str:
    """Compute the MD5 checksum of a file."""
    digest = hashlib.md5()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_file(resource: FashionMNISTResource, destination: Path) -> None:
    """Download one FashionMNIST file to the given path."""
    print(f"Downloading {resource.file_name} ...")
    with urllib.request.urlopen(resource.url) as response, destination.open("wb") as file:
        shutil.copyfileobj(response, file)


def ensure_fashion_mnist_files(root: str | Path = "data") -> Path:
    """Ensure all raw `.gz` files exist locally and pass checksum validation."""
    raw_dir = Path(root) / "FashionMNIST" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    missing_or_invalid = []
    for resource in RESOURCES:
        target = raw_dir / resource.file_name
        if not target.exists():
            missing_or_invalid.append(resource)
            continue
        if _md5(target) != resource.md5:
            print(f"Checksum mismatch detected for {resource.file_name}, re-downloading it.")
            missing_or_invalid.append(resource)

    if not missing_or_invalid:
        return raw_dir

    print("FashionMNIST files are missing locally. Auto-download will start now.")
    for resource in missing_or_invalid:
        target = raw_dir / resource.file_name
        try:
            _download_file(resource, target)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise DatasetDownloadError(
                "Could not download FashionMNIST automatically.\n"
                f"Failed file: {resource.file_name}\n"
                "You can retry later, or manually place the four `.gz` files under:\n"
                f"{raw_dir}\n"
                "Official dataset source: https://github.com/zalandoresearch/fashion-mnist"
            ) from exc

        if _md5(target) != resource.md5:
            raise DatasetDownloadError(
                f"Downloaded file {resource.file_name} but its checksum did not match."
            )

    return raw_dir


def _read_idx_images(path: Path):
    """Read an IDX image file and return a float tensor in shape [N, 1, 28, 28]."""
    torch, _, _ = _import_torch()

    with gzip.open(path, "rb") as file:
        magic, num_images, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError(f"Unexpected image magic number in {path.name}: {magic}")
        data = file.read()

    images = torch.frombuffer(bytearray(data), dtype=torch.uint8).clone()
    images = images.view(num_images, 1, rows, cols).float() / 255.0
    return images


def _read_idx_labels(path: Path):
    """Read an IDX label file and return a long tensor in shape [N]."""
    torch, _, _ = _import_torch()

    with gzip.open(path, "rb") as file:
        magic, num_labels = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError(f"Unexpected label magic number in {path.name}: {magic}")
        data = file.read()

    labels = torch.frombuffer(bytearray(data), dtype=torch.uint8).clone().long()
    if labels.numel() != num_labels:
        raise ValueError(f"Label count mismatch in {path.name}.")
    return labels


def load_fashion_mnist_tensors(root: str | Path = "data"):
    """Load FashionMNIST as tensors."""
    raw_dir = ensure_fashion_mnist_files(root=root)

    x_train = _read_idx_images(raw_dir / "train-images-idx3-ubyte.gz")
    y_train = _read_idx_labels(raw_dir / "train-labels-idx1-ubyte.gz")
    x_test = _read_idx_images(raw_dir / "t10k-images-idx3-ubyte.gz")
    y_test = _read_idx_labels(raw_dir / "t10k-labels-idx1-ubyte.gz")
    return x_train, y_train, x_test, y_test


def make_fashion_mnist_dataloaders(
    batch_size: int,
    root: str | Path = "data",
):
    """Create train/test DataLoaders for FashionMNIST."""
    _, DataLoader, TensorDataset = _import_torch()

    x_train, y_train, x_test, y_test = load_fashion_mnist_tensors(root=root)

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    return train_loader, test_loader
