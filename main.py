"""Project entrypoint for the beginner FashionMNIST project."""

from __future__ import annotations

from dataset import DatasetDownloadError, MissingTorchDependencyError
from train import run_training


def main() -> None:
    try:
        run_training()
    except (MissingTorchDependencyError, DatasetDownloadError) as exc:
        print(exc)
        print()
        print(
            "Tips:\n"
            "1. Make sure PyTorch is installed in this interpreter.\n"
            "2. If FashionMNIST cannot be downloaded automatically, "
            "run again later or place the dataset files under data/FashionMNIST/raw."
        )


if __name__ == "__main__":
    main()
