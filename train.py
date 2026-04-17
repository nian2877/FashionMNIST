"""Day 2 training script: real FashionMNIST data with a simple MLP baseline."""

from __future__ import annotations

from dataclasses import dataclass

from dataset import (
    CLASS_NAMES,
    DatasetDownloadError,
    MissingTorchDependencyError,
    make_fashion_mnist_dataloaders,
)


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration values used in the Day 2 training run."""

    data_root: str = "data"
    batch_size: int = 64
    epochs: int = 5
    learning_rate: float = 0.001
    hidden_dim: int = 256
    random_seed: int = 42


def _import_torch():
    """Import torch lazily and raise a clear error if it is missing."""
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise MissingTorchDependencyError(
            "PyTorch is not installed in this Python environment."
        ) from exc

    return torch, nn


def count_parameters(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def train_one_epoch(model, dataloader, criterion, optimizer, device) -> tuple[float, float]:
    """Train the model for one full pass over the training set."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in dataloader:
        # Move tensors to CPU or GPU.
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass: let the model predict logits for each class.
        logits = model(images)
        loss = criterion(logits, labels)

        # Clear old gradients before computing new ones.
        optimizer.zero_grad()

        # Backpropagation: compute gradients for every trainable parameter.
        loss.backward()

        # Update parameters using the gradients we just computed.
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        predictions = logits.argmax(dim=1)
        total_correct += (predictions == labels).sum().item()
        total_examples += labels.size(0)

    avg_loss = total_loss / total_examples
    accuracy = total_correct / total_examples
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device) -> tuple[float, float]:
    """Evaluate the model without changing parameters."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    torch, _ = _import_torch()
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            predictions = logits.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_examples += labels.size(0)

    avg_loss = total_loss / total_examples
    accuracy = total_correct / total_examples
    return avg_loss, accuracy


def print_dataset_overview(train_loader, test_loader, model, device, config: TrainingConfig) -> None:
    """Print shapes and metadata so beginners can see what flows through the code."""
    sample_images, sample_labels = next(iter(train_loader))
    sample_logits = model(sample_images.to(device)).cpu()

    print("=== Day 2: FashionMNIST training ===")
    print(f"Device:            {device}")
    print(f"Train samples:     {len(train_loader.dataset)}")
    print(f"Test samples:      {len(test_loader.dataset)}")
    print(f"Batch size:        {config.batch_size}")
    print(f"Epochs:            {config.epochs}")
    print(f"Learning rate:     {config.learning_rate}")
    print(f"Image batch shape: {tuple(sample_images.shape)}")
    print(f"Label batch shape: {tuple(sample_labels.shape)}")
    print(f"Logits shape:      {tuple(sample_logits.shape)}")
    print(f"Model parameters:  {count_parameters(model)}")
    print()
    print("Class index mapping:")
    for index, class_name in enumerate(CLASS_NAMES):
        print(f"{index}: {class_name}")
    print()


def run_training(config: TrainingConfig | None = None) -> None:
    config = config or TrainingConfig()
    torch, nn = _import_torch()
    from model import FashionMNISTMLP

    torch.manual_seed(config.random_seed)

    # Use GPU if available. CPU is also completely fine for FashionMNIST.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = make_fashion_mnist_dataloaders(
        batch_size=config.batch_size,
        root=config.data_root,
    )
    model = FashionMNISTMLP(hidden_dim=config.hidden_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    print_dataset_overview(train_loader, test_loader, model, device, config)
    print("Training loop reminder:")
    print("1. Read one batch of images and labels from the DataLoader")
    print("2. Send images through the model to get logits")
    print("3. Use CrossEntropyLoss to compare logits and labels")
    print("4. optimizer.zero_grad() clears old gradients")
    print("5. loss.backward() computes new gradients")
    print("6. optimizer.step() updates the parameters")
    print()

    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        test_loss, test_acc = evaluate(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
        )
        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.2%} | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.2%}"
        )

    print()
    print("Training finished.")
    print("You now have a complete real-image classification baseline.")


if __name__ == "__main__":
    try:
        run_training()
    except (MissingTorchDependencyError, DatasetDownloadError) as exc:
        print(exc)
