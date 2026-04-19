"""Project entrypoint for the latest learning stage in the FashionMNIST project."""

from __future__ import annotations

from dataset import DatasetDownloadError, MissingTorchDependencyError
from transfer_dataset import DatasetStructureError
from transfer_learning import (
    MissingPretrainedCheckpointError,
    TransferBackboneError,
    run_transfer_learning,
)


def main() -> None:
    try:
        run_transfer_learning()
    except (
        MissingTorchDependencyError,
        DatasetDownloadError,
        DatasetStructureError,
        MissingPretrainedCheckpointError,
        TransferBackboneError,
    ) as exc:
        print(exc)
        print()
        print(
            "Tips:\n"
            "1. Make sure PyTorch and matplotlib are installed in this interpreter.\n"
            "2. For Day 5, prepare a folder dataset like data/transfer_real/train/class_name/*.jpg\n"
            "3. With torchvision/timm installed, you can now use official pretrained ResNet backbones.\n"
            "4. If pretrained weights cannot download, either connect to the internet or set use_pretrained_weights=False.\n"
            "5. The fallback route custom_cnn still uses checkpoints under results/day3 or results/day4."
        )


if __name__ == "__main__":
    main()
