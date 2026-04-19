"""Model definitions for the beginner-friendly FashionMNIST project.

Day 2 used a simple MLP baseline.
Day 3 added a real CNN for FashionMNIST.
Day 5 extends that CNN into a transfer-learning backbone for real image folders.
"""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - handled by caller at runtime
    raise ImportError(
        "PyTorch is required for model.py. Install torch before running this demo."
    ) from exc


class FashionMNISTMLP(nn.Module):
    """A simple MLP baseline.

    This model treats the image as one long vector.
    It is useful as a baseline because it is easy to understand,
    but it does not preserve local spatial structure very well.
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


class FashionMNISTCNN(nn.Module):
    """A small CNN for Day 3.

    This model is intentionally simple and heavily commented.
    It introduces the key building blocks of image deep learning:
    - convolution
    - non-linearity
    - pooling
    - feature flattening
    - fully connected classification
    """

    def __init__(
        self,
        num_classes: int = 10,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # Block 1:
        # Input shape:  [batch, 1, 28, 28]
        # Output shape: [batch, 16, 14, 14]
        # Why 14 x 14 at the end?
        # Because MaxPool2d(kernel_size=2) halves height and width.
        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Block 2:
            # Input shape:  [batch, 16, 14, 14]
            # Output shape: [batch, 32, 7, 7]
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # After the two pooling steps, the spatial size becomes 7 x 7.
        # So the flattened feature size is:
        # 32 channels * 7 * 7 = 1568
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            # Dropout randomly zeroes part of the hidden features during training.
            # It is a classic regularization method used to reduce overfitting.
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Step 1: extract local visual features with convolution.
        features = self.features(images)

        # Step 2: map high-level features to class logits.
        logits = self.classifier(features)
        return logits


class TransferLearningCNN(nn.Module):
    """A transfer-learning model built on top of the Day 3/4 CNN backbone.

    This model accepts RGB images, reuses the same convolutional idea,
    and adds an adaptive pooling step so we are not locked to 28x28 inputs.
    """

    def __init__(
        self,
        num_classes: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # Adaptive pooling lets us handle larger real images while still
        # producing the 7x7 feature grid that our classifier expects.
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        pooled_features = self.adaptive_pool(features)
        logits = self.classifier(pooled_features)
        return logits


def load_fashion_cnn_pretrained_weights(model: TransferLearningCNN, state_dict: dict) -> None:
    """Load transferable weights from a FashionMNISTCNN checkpoint.

    The source CNN was trained on 1-channel FashionMNIST images.
    The Day 5 model expects 3-channel RGB images.
    To bridge that gap, we copy the learned grayscale filters into
    all three input channels and average them.
    """

    source_conv1_weight = state_dict["features.0.weight"]
    source_conv1_bias = state_dict["features.0.bias"]
    source_conv2_weight = state_dict["features.3.weight"]
    source_conv2_bias = state_dict["features.3.bias"]

    with torch.no_grad():
        repeated_weight = source_conv1_weight.repeat(1, 3, 1, 1) / 3.0
        model.features[0].weight.copy_(repeated_weight)
        model.features[0].bias.copy_(source_conv1_bias)
        model.features[3].weight.copy_(source_conv2_weight)
        model.features[3].bias.copy_(source_conv2_bias)

        # The first linear layer can also be reused because adaptive pooling
        # keeps the hidden feature size at 32 * 7 * 7 = 1568.
        if "classifier.1.weight" in state_dict and "classifier.1.bias" in state_dict:
            model.classifier[1].weight.copy_(state_dict["classifier.1.weight"])
            model.classifier[1].bias.copy_(state_dict["classifier.1.bias"])


def build_model(
    model_name: str,
    hidden_dim: int = 256,
    dropout: float = 0.0,
    num_classes: int = 10,
) -> nn.Module:
    """Factory function used by the training script.

    Args:
        model_name: either "mlp" or "cnn"
        hidden_dim: used only by the MLP baseline
    """
    normalized_name = model_name.lower()

    if normalized_name == "mlp":
        return FashionMNISTMLP(hidden_dim=hidden_dim)
    if normalized_name == "cnn":
        return FashionMNISTCNN(dropout=dropout)
    if normalized_name == "transfer_cnn":
        return TransferLearningCNN(num_classes=num_classes, dropout=dropout)

    raise ValueError(
        f"Unknown model_name={model_name!r}. Expected 'mlp', 'cnn', or 'transfer_cnn'."
    )
