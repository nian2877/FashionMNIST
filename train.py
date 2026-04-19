"""Day 4 experiment script: hyperparameter comparison and error analysis.

Day 3 introduced CNN.
Day 4 asks a deeper question:
"How do learning rate, batch size, and dropout affect the model?"

This file runs a small but complete experiment suite and saves:
- per-epoch histories
- training curves
- model weights
- misclassified sample images
- an overall summary JSON
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from dataset import (
    CLASS_NAMES,
    DatasetDownloadError,
    MissingTorchDependencyError,
    make_fashion_mnist_dataloaders,
)


@dataclass(frozen=True)
class TrainingConfig:
    """Global configuration for the Day 4 experiment suite."""

    data_root: str = "data"
    output_root: str = "results"
    epochs: int = 3
    train_limit: int | None = 12000
    test_limit: int | None = 2000
    mlp_hidden_dim: int = 256
    random_seed: int = 42
    error_example_count: int = 12
    experiment_names: tuple[str, ...] | None = None


@dataclass(frozen=True)
class ExperimentSpec:
    """One experiment in the Day 4 comparison suite."""

    name: str
    category: str
    learning_rate: float
    batch_size: int
    dropout: float
    epochs: int
    note: str


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


def _import_plotting():
    """Import matplotlib lazily because plotting is only needed for analysis outputs."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for Day 4 plots. Please install matplotlib."
        ) from exc
    return plt


def count_parameters(model) -> int:
    """Count the total number of trainable parameters."""
    return sum(parameter.numel() for parameter in model.parameters())


def get_experiment_suite(config: TrainingConfig) -> list[ExperimentSpec]:
    """Build the default Day 4 experiment suite."""
    experiments = [
        ExperimentSpec(
            name="baseline",
            category="baseline",
            learning_rate=0.001,
            batch_size=64,
            dropout=0.0,
            epochs=config.epochs,
            note="Baseline CNN configuration used as the reference point.",
        ),
        ExperimentSpec(
            name="lr_low",
            category="learning_rate",
            learning_rate=0.0003,
            batch_size=64,
            dropout=0.0,
            epochs=config.epochs,
            note="Lower learning rate usually learns more slowly but can be stable.",
        ),
        ExperimentSpec(
            name="lr_high",
            category="learning_rate",
            learning_rate=0.003,
            batch_size=64,
            dropout=0.0,
            epochs=config.epochs,
            note="Higher learning rate may learn faster but can overshoot.",
        ),
        ExperimentSpec(
            name="batch_32",
            category="batch_size",
            learning_rate=0.001,
            batch_size=32,
            dropout=0.0,
            epochs=config.epochs,
            note="Smaller batch gives noisier gradients and more updates per epoch.",
        ),
        ExperimentSpec(
            name="batch_128",
            category="batch_size",
            learning_rate=0.001,
            batch_size=128,
            dropout=0.0,
            epochs=config.epochs,
            note="Larger batch gives smoother gradients but fewer updates per epoch.",
        ),
        ExperimentSpec(
            name="dropout_30",
            category="dropout",
            learning_rate=0.001,
            batch_size=64,
            dropout=0.3,
            epochs=config.epochs,
            note="Dropout removes part of the hidden activations during training.",
        ),
    ]

    if config.experiment_names is None:
        return experiments

    wanted = set(config.experiment_names)
    filtered = [experiment for experiment in experiments if experiment.name in wanted]
    if not filtered:
        raise ValueError(
            f"No experiments matched config.experiment_names={config.experiment_names!r}."
        )
    return filtered


def train_one_epoch(model, dataloader, criterion, optimizer, device) -> tuple[float, float]:
    """Train the model for one full pass over the training set."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
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


def create_output_dir(config: TrainingConfig) -> Path:
    """Create the directory that will store Day 4 artifacts."""
    output_dir = Path(config.output_root) / "day4"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def create_experiment_dir(output_dir: Path, experiment: ExperimentSpec) -> Path:
    """Create one subdirectory per experiment."""
    experiment_dir = output_dir / experiment.name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir


def print_experiment_intro(
    train_loader,
    test_loader,
    model,
    device,
    experiment: ExperimentSpec,
    config: TrainingConfig,
) -> None:
    """Print experiment settings and a quick tensor-shape overview."""
    sample_images, sample_labels = next(iter(train_loader))
    sample_logits = model(sample_images.to(device)).cpu()

    print(f"=== Day 4 Experiment: {experiment.name} ===")
    print(f"Category:          {experiment.category}")
    print(f"Note:              {experiment.note}")
    print(f"Device:            {device}")
    print(f"Train samples:     {len(train_loader.dataset)}")
    print(f"Test samples:      {len(test_loader.dataset)}")
    print(f"Batch size:        {experiment.batch_size}")
    print(f"Epochs:            {experiment.epochs}")
    print(f"Learning rate:     {experiment.learning_rate}")
    print(f"Dropout:           {experiment.dropout}")
    print(f"Image batch shape: {tuple(sample_images.shape)}")
    print(f"Label batch shape: {tuple(sample_labels.shape)}")
    print(f"Logits shape:      {tuple(sample_logits.shape)}")
    print(f"Model parameters:  {count_parameters(model)}")
    print(f"Subset train/test: {config.train_limit} / {config.test_limit}")
    print()


def save_json(path: Path, data: dict | list) -> Path:
    """Save a dict or list as readable JSON."""
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def save_history_plot(experiment_dir: Path, history: list[dict], title: str) -> Path:
    """Save loss/accuracy curves for one experiment."""
    plt = _import_plotting()

    epochs = [record["epoch"] for record in history]
    train_loss = [record["train_loss"] for record in history]
    test_loss = [record["test_loss"] for record in history]
    train_acc = [record["train_acc"] for record in history]
    test_acc = [record["test_acc"] for record in history]

    figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

    axes[0].plot(epochs, train_loss, marker="o", label="train_loss")
    axes[0].plot(epochs, test_loss, marker="o", label="test_loss")
    axes[0].set_title(f"{title} Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, train_acc, marker="o", label="train_acc")
    axes[1].plot(epochs, test_acc, marker="o", label="test_acc")
    axes[1].set_title(f"{title} Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    figure.tight_layout()
    plot_path = experiment_dir / "curves.png"
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)
    return plot_path


def collect_misclassified_examples(model, dataloader, device, max_examples: int) -> list[dict]:
    """Collect a few wrong predictions for qualitative analysis."""
    torch, _ = _import_torch()
    model.eval()
    examples = []

    with torch.no_grad():
        for images, labels in dataloader:
            logits = model(images.to(device))
            probabilities = torch.softmax(logits, dim=1).cpu()
            predictions = probabilities.argmax(dim=1)

            mismatches = predictions != labels
            if mismatches.any():
                mismatch_indices = mismatches.nonzero(as_tuple=False).flatten().tolist()
                for index in mismatch_indices:
                    examples.append(
                        {
                            "image": images[index].cpu(),
                            "true_label": int(labels[index].item()),
                            "pred_label": int(predictions[index].item()),
                            "confidence": float(probabilities[index, predictions[index]].item()),
                        }
                    )
                    if len(examples) >= max_examples:
                        return examples

    return examples


def save_error_samples_plot(experiment_dir: Path, examples: list[dict], title: str) -> Path | None:
    """Plot several misclassified images with true/predicted labels."""
    if not examples:
        return None

    plt = _import_plotting()
    columns = 4
    rows = (len(examples) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(12, 3 * rows))

    if rows == 1:
        axes = [axes]

    flat_axes = []
    for row_axes in axes:
        if isinstance(row_axes, (list, tuple)):
            flat_axes.extend(row_axes)
        else:
            flat_axes.extend(row_axes.tolist())

    for axis, example in zip(flat_axes, examples):
        image = example["image"].squeeze(0).numpy()
        true_name = CLASS_NAMES[example["true_label"]]
        pred_name = CLASS_NAMES[example["pred_label"]]
        axis.imshow(image, cmap="gray")
        axis.set_title(
            f"T: {true_name}\nP: {pred_name}\nConf: {example['confidence']:.2f}",
            fontsize=9,
        )
        axis.axis("off")

    for axis in flat_axes[len(examples):]:
        axis.axis("off")

    figure.suptitle(title)
    figure.tight_layout()
    plot_path = experiment_dir / "misclassified_examples.png"
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)
    return plot_path


def save_experiment_comparison_plot(output_dir: Path, summary: dict) -> Path:
    """Plot test accuracy comparison across all Day 4 experiments."""
    plt = _import_plotting()

    names = list(summary["experiments"].keys())
    accuracies = [
        summary["experiments"][name]["final_metrics"]["test_acc"] for name in names
    ]

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(names, accuracies)
    axis.set_title("Day 4 Experiment Comparison (Test Accuracy)")
    axis.set_xlabel("Experiment")
    axis.set_ylabel("Test Accuracy")
    axis.set_ylim(0.0, 1.0)
    axis.tick_params(axis="x", rotation=30)

    for index, accuracy in enumerate(accuracies):
        axis.text(index, accuracy + 0.01, f"{accuracy:.2%}", ha="center")

    figure.tight_layout()
    plot_path = output_dir / "comparison_test_accuracy.png"
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)
    return plot_path


def train_experiment(
    experiment: ExperimentSpec,
    config: TrainingConfig,
    device,
    output_dir: Path,
) -> dict:
    """Train one CNN experiment and save all of its artifacts."""
    torch, nn = _import_torch()
    from model import build_model

    torch.manual_seed(config.random_seed)

    train_loader, test_loader = make_fashion_mnist_dataloaders(
        batch_size=experiment.batch_size,
        root=config.data_root,
        train_limit=config.train_limit,
        test_limit=config.test_limit,
        subset_seed=config.random_seed,
    )

    model = build_model(
        model_name="cnn",
        hidden_dim=config.mlp_hidden_dim,
        dropout=experiment.dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=experiment.learning_rate)

    experiment_dir = create_experiment_dir(output_dir, experiment)
    print_experiment_intro(train_loader, test_loader, model, device, experiment, config)

    history = []
    for epoch in range(1, experiment.epochs + 1):
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

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
        }
        history.append(record)
        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.2%} | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.2%}"
        )

    errors = collect_misclassified_examples(
        model=model,
        dataloader=test_loader,
        device=device,
        max_examples=config.error_example_count,
    )

    history_json = save_json(experiment_dir / "history.json", history)
    errors_json = save_json(
        experiment_dir / "misclassified_examples.json",
        [
            {
                "true_label": example["true_label"],
                "true_name": CLASS_NAMES[example["true_label"]],
                "pred_label": example["pred_label"],
                "pred_name": CLASS_NAMES[example["pred_label"]],
                "confidence": example["confidence"],
            }
            for example in errors
        ],
    )
    curve_plot = save_history_plot(
        experiment_dir=experiment_dir,
        history=history,
        title=experiment.name,
    )
    error_plot = save_error_samples_plot(
        experiment_dir=experiment_dir,
        examples=errors,
        title=f"Misclassified Samples - {experiment.name}",
    )
    weights_path = experiment_dir / "cnn_weights.pt"
    torch.save(model.state_dict(), weights_path)

    final_epoch = history[-1]
    experiment_summary = {
        "spec": asdict(experiment),
        "final_metrics": {
            "train_loss": final_epoch["train_loss"],
            "train_acc": final_epoch["train_acc"],
            "test_loss": final_epoch["test_loss"],
            "test_acc": final_epoch["test_acc"],
            "parameters": count_parameters(model),
        },
        "artifacts": {
            "history_json": str(history_json),
            "curves_png": str(curve_plot),
            "weights_path": str(weights_path),
            "misclassified_json": str(errors_json),
            "misclassified_png": str(error_plot) if error_plot is not None else None,
        },
    }
    save_json(experiment_dir / "summary.json", experiment_summary)

    print()
    print(f"Experiment {experiment.name} finished.")
    print()
    return experiment_summary


def print_day4_summary(summary: dict) -> None:
    """Print a compact comparison table for all experiments."""
    print("=== Day 4 Comparison Summary ===")
    ranking = sorted(
        summary["experiments"].items(),
        key=lambda item: item[1]["final_metrics"]["test_acc"],
        reverse=True,
    )
    for name, record in ranking:
        metrics = record["final_metrics"]
        spec = record["spec"]
        print(
            f"{name:>10} | "
            f"category={spec['category']:<13} | "
            f"lr={spec['learning_rate']:<7g} | "
            f"batch={spec['batch_size']:<3d} | "
            f"dropout={spec['dropout']:<4.1f} | "
            f"test_acc={metrics['test_acc']:.2%}"
        )
    print()
    print(f"Best experiment: {summary['best_experiment']}")
    print()


def run_training(config: TrainingConfig | None = None) -> None:
    """Run the Day 4 experiment suite.

    Day 4 is no longer just one training run.
    It is an experiment-and-analysis stage where we compare:
    - learning rate
    - batch size
    - dropout
    """
    config = config or TrainingConfig()
    torch, _ = _import_torch()

    torch.manual_seed(config.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = create_output_dir(config)
    experiments = get_experiment_suite(config)

    summary = {
        "config": asdict(config),
        "experiments": {},
        "best_experiment": None,
        "artifacts": {},
    }

    for experiment in experiments:
        experiment_summary = train_experiment(
            experiment=experiment,
            config=config,
            device=device,
            output_dir=output_dir,
        )
        summary["experiments"][experiment.name] = experiment_summary

    best_name = max(
        summary["experiments"],
        key=lambda name: summary["experiments"][name]["final_metrics"]["test_acc"],
    )
    summary["best_experiment"] = best_name
    comparison_plot = save_experiment_comparison_plot(output_dir, summary)
    summary["artifacts"]["comparison_plot"] = str(comparison_plot)

    summary_path = save_json(output_dir / "day4_summary.json", summary)
    print_day4_summary(summary)
    print("Day 4 experiments finished.")
    print(f"Results saved under: {output_dir}")
    print(f"Summary file: {summary_path}")


if __name__ == "__main__":
    try:
        run_training()
    except (MissingTorchDependencyError, DatasetDownloadError) as exc:
        print(exc)
