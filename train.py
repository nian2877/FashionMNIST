"""Day 3 training script: FashionMNIST with both MLP and CNN experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from dataset import (
    CLASS_NAMES,
    DatasetDownloadError,
    MissingTorchDependencyError,
    make_fashion_mnist_dataloaders,
)


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration values used in the Day 3 training run."""

    data_root: str = "data"
    output_root: str = "results"
    batch_size: int = 64
    epochs: int = 5
    learning_rate: float = 0.001
    mlp_hidden_dim: int = 256
    run_baseline_mlp: bool = True
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
    """Count the total number of trainable parameters."""
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


def print_dataset_overview(
    train_loader,
    test_loader,
    model,
    device,
    config: TrainingConfig,
    model_name: str,
) -> None:
    """Print shapes and metadata so beginners can see what flows through the code."""
    sample_images, sample_labels = next(iter(train_loader))
    sample_logits = model(sample_images.to(device)).cpu()

    print(f"=== Day 3: FashionMNIST training ({model_name.upper()}) ===")
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


def create_output_dir(config: TrainingConfig) -> Path:
    """Create the directory that will store training artifacts."""
    output_dir = Path(config.output_root) / "day3"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_history(output_dir: Path, model_name: str, history: list[dict]) -> Path:
    """Save epoch-by-epoch metrics to a JSON file."""
    history_path = output_dir / f"{model_name}_history.json"
    history_path.write_text(
        json.dumps(history, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return history_path


def save_summary(output_dir: Path, summary: dict) -> Path:
    """Save a concise summary of the whole Day 3 experiment."""
    summary_path = output_dir / "day3_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary_path


def train_single_model(
    model_name: str,
    config: TrainingConfig,
    train_loader,
    test_loader,
    device,
):
    """Train one model and return its metrics history plus the trained model."""
    torch, nn = _import_torch()
    from model import build_model

    # Reset the seed before each experiment so the comparison is more fair.
    torch.manual_seed(config.random_seed)

    model = build_model(
        model_name=model_name,
        hidden_dim=config.mlp_hidden_dim,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    print_dataset_overview(train_loader, test_loader, model, device, config, model_name)
    print("Training loop reminder:")
    print("1. Read one batch of images and labels from the DataLoader")
    print("2. Send images through the model to get logits")
    print("3. Use CrossEntropyLoss to compare logits and labels")
    print("4. optimizer.zero_grad() clears old gradients")
    print("5. loss.backward() computes new gradients")
    print("6. optimizer.step() updates the parameters")
    print()

    history = []
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
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
        }
        history.append(epoch_record)
        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.2%} | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.2%}"
        )

    print()
    print(f"{model_name.upper()} training finished.")
    return model, history


def save_model_weights(output_dir: Path, model_name: str, model) -> Path:
    """Save the trained model parameters."""
    torch, _ = _import_torch()
    model_path = output_dir / f"{model_name}_weights.pt"
    torch.save(model.state_dict(), model_path)
    return model_path


def print_comparison_report(summary: dict) -> None:
    """Print a short teaching-oriented comparison between MLP and CNN."""
    print("=== Comparison Summary ===")
    for model_name, metrics in summary["final_metrics"].items():
        print(
            f"{model_name.upper():>4} | "
            f"train_acc={metrics['train_acc']:.2%} | "
            f"test_acc={metrics['test_acc']:.2%} | "
            f"parameters={metrics['parameters']}"
        )

    if "mlp" in summary["final_metrics"] and "cnn" in summary["final_metrics"]:
        gap = (
            summary["final_metrics"]["cnn"]["test_acc"]
            - summary["final_metrics"]["mlp"]["test_acc"]
        )
        print(f"CNN - MLP test accuracy gap: {gap:.2%}")
    print()


def run_training(config: TrainingConfig | None = None) -> None:
    """Run the Day 3 experiment.

    By default we train:
    - an MLP baseline from Day 2
    - a CNN from Day 3

    This makes it easier to see why CNN is better suited for images.
    """
    config = config or TrainingConfig()
    torch, _ = _import_torch()

    torch.manual_seed(config.random_seed)

    # Use GPU if available. CPU is also completely fine for FashionMNIST.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = make_fashion_mnist_dataloaders(
        batch_size=config.batch_size,
        root=config.data_root,
    )

    model_order = []
    if config.run_baseline_mlp:
        model_order.append("mlp")
    model_order.append("cnn")

    output_dir = create_output_dir(config)
    summary = {
        "config": {
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "learning_rate": config.learning_rate,
            "mlp_hidden_dim": config.mlp_hidden_dim,
            "run_baseline_mlp": config.run_baseline_mlp,
        },
        "final_metrics": {},
        "artifacts": {},
    }

    for model_name in model_order:
        model, history = train_single_model(
            model_name=model_name,
            config=config,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
        )
        history_path = save_history(output_dir, model_name, history)
        weight_path = save_model_weights(output_dir, model_name, model)
        final_epoch = history[-1]

        summary["final_metrics"][model_name] = {
            "train_loss": final_epoch["train_loss"],
            "train_acc": final_epoch["train_acc"],
            "test_loss": final_epoch["test_loss"],
            "test_acc": final_epoch["test_acc"],
            "parameters": count_parameters(model),
        }
        summary["artifacts"][model_name] = {
            "history_json": str(history_path),
            "weights_path": str(weight_path),
        }

    summary_path = save_summary(output_dir, summary)

    print_comparison_report(summary)
    print("Training finished.")
    print(f"Results saved under: {output_dir}")
    print(f"Summary file: {summary_path}")


if __name__ == "__main__":
    try:
        run_training()
    except (MissingTorchDependencyError, DatasetDownloadError) as exc:
        print(exc)
