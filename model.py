"""Model definitions for the beginner-friendly FashionMNIST project."""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - handled by caller at runtime
    raise ImportError(
        "PyTorch is required for model.py. Install torch before running this demo."
    ) from exc


class FashionMNISTMLP(nn.Module):
    """A simple MLP baseline for Day 2.

    We intentionally avoid CNN for now.
    Day 2 is about understanding how real image data moves through:
    DataLoader -> model -> loss -> optimizer
    """

    def __init__(
        self,
        input_dim: int = 28 * 28,
        hidden_dim: int = 256,
        num_classes: int = 10,
    ) -> None:
        super().__init__()

        # Flatten changes [batch, 1, 28, 28] into [batch, 784].
        self.flatten = nn.Flatten()

        # This is the actual classifier.
        # First linear layer extracts a hidden representation.
        # ReLU adds non-linearity.
        # Final linear layer outputs 10 class logits.
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.flatten(images)
        logits = self.classifier(features)
        return logits
