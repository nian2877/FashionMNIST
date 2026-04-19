"""Day 5: transfer learning on a small real-image folder dataset.

Now that torchvision/timm are available, Day 5 prefers an official pretrained
image backbone such as ResNet18. We still keep the earlier custom-CNN route
as an offline-friendly fallback.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from transfer_dataset import DatasetStructureError, create_transfer_dataloaders


class MissingPretrainedCheckpointError(RuntimeError):
    """Raised when no usable source checkpoint can be found for transfer learning."""


class TransferBackboneError(RuntimeError):
    """Raised when the requested transfer-learning backbone cannot be created."""


@dataclass(frozen=True)
class TransferConfig:
    """Configuration for the Day 5 transfer-learning workflow."""

    dataset_root: str = "data/transfer_real"
    output_root: str = "results"
    image_size: int = 224
    batch_size: int = 16
    frozen_epochs: int = 2
    finetune_epochs: int = 2
    head_learning_rate: float = 0.001
    finetune_learning_rate: float = 0.0003
    dropout: float = 0.3
    random_seed: int = 42
    backbone_name: str = "torchvision_resnet18"
    use_pretrained_weights: bool = True
    pretrained_checkpoint: str | None = None
    error_example_count: int = 12


def _import_torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for Day 5 transfer learning.") from exc
    return torch, nn


def _import_plotting():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for Day 5 plots.") from exc
    return plt


def find_default_pretrained_checkpoint() -> Path | None:
    """Find a suitable source checkpoint from earlier days."""
    candidates = [
        Path("results/day4/baseline/cnn_weights.pt"),
        Path("results/day3/cnn_weights.pt"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_pretrained_checkpoint(config: TransferConfig) -> Path:
    """Resolve which checkpoint to use as the transfer-learning source."""
    if config.pretrained_checkpoint is not None:
        path = Path(config.pretrained_checkpoint)
        if not path.exists():
            raise MissingPretrainedCheckpointError(
                f"Configured pretrained checkpoint does not exist: {path}"
            )
        return path

    auto_path = find_default_pretrained_checkpoint()
    if auto_path is None:
        raise MissingPretrainedCheckpointError(
            "No pretrained CNN checkpoint was found.\n"
            "Expected one of these files to exist:\n"
            "- results/day4/baseline/cnn_weights.pt\n"
            "- results/day3/cnn_weights.pt"
        )
    return auto_path


def get_backbone_normalization(config: TransferConfig) -> tuple[tuple[float, float, float] | None, tuple[float, float, float] | None]:
    """Return the normalization that matches the selected backbone."""
    if config.backbone_name in {"torchvision_resnet18", "timm_resnet18"}:
        return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    return None, None


def get_classifier_module(model, backbone_name: str):
    """Return the classification head module for the current backbone."""
    if backbone_name in {"torchvision_resnet18", "timm_resnet18"}:
        return model.fc
    if backbone_name == "custom_cnn":
        return model.classifier
    raise TransferBackboneError(f"Unknown backbone_name={backbone_name!r}")


def set_backbone_trainable(model, backbone_name: str, trainable: bool) -> None:
    """Freeze or unfreeze the feature backbone while keeping the classifier trainable."""
    if trainable:
        for parameter in model.parameters():
            parameter.requires_grad = True
        return

    for parameter in model.parameters():
        parameter.requires_grad = False

    classifier = get_classifier_module(model, backbone_name)
    for parameter in classifier.parameters():
        parameter.requires_grad = True


def build_transfer_model(config: TransferConfig, num_classes: int):
    """Build the selected transfer-learning model."""
    torch, nn = _import_torch()

    if config.backbone_name == "torchvision_resnet18":
        try:
            from torchvision import models
        except ImportError as exc:
            raise TransferBackboneError(
                "torchvision is required for backbone_name='torchvision_resnet18'."
            ) from exc

        try:
            weights = (
                models.ResNet18_Weights.DEFAULT if config.use_pretrained_weights else None
            )
            model = models.resnet18(weights=weights)
        except Exception as exc:
            raise TransferBackboneError(
                "Could not create torchvision ResNet18 with pretrained weights.\n"
                "If this is the first run, make sure the machine can download weights,\n"
                "or set use_pretrained_weights=False."
            ) from exc

        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=config.dropout),
            nn.Linear(in_features, num_classes),
        )
        source_description = (
            "torchvision.models.resnet18(ResNet18_Weights.DEFAULT)"
            if config.use_pretrained_weights
            else "torchvision.models.resnet18(weights=None)"
        )
        return model, source_description

    if config.backbone_name == "timm_resnet18":
        try:
            import timm
        except ImportError as exc:
            raise TransferBackboneError(
                "timm is required for backbone_name='timm_resnet18'."
            ) from exc

        try:
            model = timm.create_model(
                "resnet18",
                pretrained=config.use_pretrained_weights,
                num_classes=num_classes,
            )
        except Exception as exc:
            raise TransferBackboneError(
                "Could not create timm ResNet18 with pretrained weights.\n"
                "If this is the first run, make sure the machine can download weights,\n"
                "or set use_pretrained_weights=False."
            ) from exc

        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=config.dropout),
            nn.Linear(in_features, num_classes),
        )
        source_description = (
            "timm.create_model('resnet18', pretrained=True)"
            if config.use_pretrained_weights
            else "timm.create_model('resnet18', pretrained=False)"
        )
        return model, source_description

    if config.backbone_name == "custom_cnn":
        checkpoint_path = resolve_pretrained_checkpoint(config)
        from model import build_model, load_fashion_cnn_pretrained_weights

        model = build_model(
            model_name="transfer_cnn",
            dropout=config.dropout,
            num_classes=num_classes,
        )
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        load_fashion_cnn_pretrained_weights(model, state_dict)
        return model, f"custom_cnn checkpoint: {checkpoint_path}"

    raise TransferBackboneError(
        f"Unknown backbone_name={config.backbone_name!r}. "
        "Expected 'torchvision_resnet18', 'timm_resnet18', or 'custom_cnn'."
    )


def count_parameters(model) -> int:
    """Count the total number of trainable parameters."""
    return sum(parameter.numel() for parameter in model.parameters())


def count_trainable_parameters(model) -> int:
    """Count only the parameters that currently require gradients."""
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )


def save_json(path: Path, data: dict | list) -> Path:
    """Save dict/list data as readable JSON."""
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def train_one_epoch(model, dataloader, criterion, optimizer, device) -> tuple[float, float]:
    """Train for one epoch."""
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

    return total_loss / total_examples, total_correct / total_examples


def evaluate(model, dataloader, criterion, device) -> tuple[float, float]:
    """Evaluate without updating parameters."""
    torch, _ = _import_torch()
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

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

    return total_loss / total_examples, total_correct / total_examples


def collect_misclassified_examples(model, dataloader, device, class_names, max_examples: int) -> list[dict]:
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
                    pred_label = int(predictions[index].item())
                    true_label = int(labels[index].item())
                    examples.append(
                        {
                            "image": images[index].cpu(),
                            "true_label": true_label,
                            "true_name": class_names[true_label],
                            "pred_label": pred_label,
                            "pred_name": class_names[pred_label],
                            "confidence": float(probabilities[index, pred_label].item()),
                        }
                    )
                    if len(examples) >= max_examples:
                        return examples
    return examples


def save_phase_curves(output_dir: Path, history: list[dict]) -> Path:
    """Save one combined curve figure for both transfer-learning phases."""
    plt = _import_plotting()

    epochs = [record["epoch_index"] for record in history]
    train_loss = [record["train_loss"] for record in history]
    val_loss = [record["val_loss"] for record in history]
    train_acc = [record["train_acc"] for record in history]
    val_acc = [record["val_acc"] for record in history]

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, train_loss, marker="o", label="train_loss")
    axes[0].plot(epochs, val_loss, marker="o", label="val_loss")
    axes[0].set_title("Day 5 Loss Curves")
    axes[0].set_xlabel("Global Epoch Index")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, train_acc, marker="o", label="train_acc")
    axes[1].plot(epochs, val_acc, marker="o", label="val_acc")
    axes[1].set_title("Day 5 Accuracy Curves")
    axes[1].set_xlabel("Global Epoch Index")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    figure.tight_layout()
    plot_path = output_dir / "transfer_curves.png"
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)
    return plot_path


def save_error_samples_plot(
    output_dir: Path,
    examples: list[dict],
    normalize_mean: tuple[float, float, float] | None,
    normalize_std: tuple[float, float, float] | None,
) -> Path | None:
    """Save a grid of misclassified real-image samples."""
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
        image_tensor = example["image"].clone()
        if normalize_mean is not None and normalize_std is not None:
            mean_tensor = image_tensor.new_tensor(normalize_mean).view(3, 1, 1)
            std_tensor = image_tensor.new_tensor(normalize_std).view(3, 1, 1)
            image_tensor = image_tensor * std_tensor + mean_tensor
        image = image_tensor.permute(1, 2, 0).numpy().clip(0.0, 1.0)
        axis.imshow(image)
        axis.set_title(
            f"T: {example['true_name']}\nP: {example['pred_name']}\nConf: {example['confidence']:.2f}",
            fontsize=9,
        )
        axis.axis("off")

    for axis in flat_axes[len(examples):]:
        axis.axis("off")

    figure.suptitle("Day 5 Misclassified Validation Samples")
    figure.tight_layout()
    plot_path = output_dir / "misclassified_examples.png"
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)
    return plot_path


def print_dataset_overview(train_loader, val_loader, model, device, class_names, config: TransferConfig) -> None:
    """Print shapes and metadata for the transfer-learning run."""
    sample_images, sample_labels = next(iter(train_loader))
    sample_logits = model(sample_images.to(device)).cpu()

    print("=== Day 5: Transfer Learning ===")
    print(f"Device:             {device}")
    print(f"Train samples:      {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Batch size:         {config.batch_size}")
    print(f"Image size:         {config.image_size}")
    print(f"Frozen epochs:      {config.frozen_epochs}")
    print(f"Fine-tune epochs:   {config.finetune_epochs}")
    print(f"Backbone:           {config.backbone_name}")
    print(f"Pretrained:         {config.use_pretrained_weights}")
    print(f"Head LR:            {config.head_learning_rate}")
    print(f"Fine-tune LR:       {config.finetune_learning_rate}")
    print(f"Image batch shape:  {tuple(sample_images.shape)}")
    print(f"Label batch shape:  {tuple(sample_labels.shape)}")
    print(f"Logits shape:       {tuple(sample_logits.shape)}")
    print(f"Class names:        {class_names}")
    print(f"All parameters:     {count_parameters(model)}")
    print()


def create_output_dir(config: TransferConfig) -> Path:
    """Create Day 5 output directory."""
    output_dir = Path(config.output_root) / "day5_transfer"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_phase(
    phase_name: str,
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    epochs: int,
    start_epoch_index: int,
) -> list[dict]:
    """Run one transfer-learning phase and return its history."""
    history = []
    for local_epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_loss, val_acc = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        record = {
            "phase": phase_name,
            "phase_epoch": local_epoch,
            "epoch_index": start_epoch_index + local_epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        history.append(record)
        print(
            f"{phase_name:>8} | epoch {local_epoch:02d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.2%} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.2%}"
        )
    print()
    return history


def run_transfer_learning(config: TransferConfig | None = None) -> None:
    """Run the Day 5 transfer-learning workflow."""
    config = config or TransferConfig()
    torch, nn = _import_torch()
    from model import (
        build_model,
        load_fashion_cnn_pretrained_weights,
    )

    torch.manual_seed(config.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    normalize_mean, normalize_std = get_backbone_normalization(config)
    train_loader, val_loader, class_names = create_transfer_dataloaders(
        root=config.dataset_root,
        image_size=config.image_size,
        batch_size=config.batch_size,
        random_seed=config.random_seed,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
    )

    model, source_description = build_transfer_model(
        config=config,
        num_classes=len(class_names),
    )
    model = model.to(device)

    output_dir = create_output_dir(config)
    criterion = nn.CrossEntropyLoss()

    print_dataset_overview(train_loader, val_loader, model, device, class_names, config)
    print(f"Source backbone:     {source_description}")
    print()

    # Phase 1: freeze the backbone and only train the classifier head.
    set_backbone_trainable(model, config.backbone_name, trainable=False)
    print("Phase 1: backbone frozen, only classifier head is trainable.")
    print(f"Trainable parameters: {count_trainable_parameters(model)}")
    head_optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.head_learning_rate,
    )
    frozen_history = run_phase(
        phase_name="frozen",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=head_optimizer,
        device=device,
        epochs=config.frozen_epochs,
        start_epoch_index=0,
    )

    # Phase 2: unfreeze the full model and fine-tune everything.
    set_backbone_trainable(model, config.backbone_name, trainable=True)
    print("Phase 2: full network unfrozen, now fine-tuning all parameters.")
    print(f"Trainable parameters: {count_trainable_parameters(model)}")
    finetune_optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.finetune_learning_rate,
    )
    finetune_history = run_phase(
        phase_name="finetune",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=finetune_optimizer,
        device=device,
        epochs=config.finetune_epochs,
        start_epoch_index=len(frozen_history),
    )

    full_history = frozen_history + finetune_history
    misclassified = collect_misclassified_examples(
        model=model,
        dataloader=val_loader,
        device=device,
        class_names=class_names,
        max_examples=config.error_example_count,
    )

    history_path = save_json(output_dir / "transfer_history.json", full_history)
    error_json_path = save_json(
        output_dir / "misclassified_examples.json",
        [
            {
                "true_label": item["true_label"],
                "true_name": item["true_name"],
                "pred_label": item["pred_label"],
                "pred_name": item["pred_name"],
                "confidence": item["confidence"],
            }
            for item in misclassified
        ],
    )
    curve_path = save_phase_curves(output_dir, full_history)
    error_plot_path = save_error_samples_plot(
        output_dir=output_dir,
        examples=misclassified,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
    )
    weights_path = output_dir / "transfer_weights.pt"
    torch.save(model.state_dict(), weights_path)

    final_record = full_history[-1]
    summary = {
        "config": asdict(config),
        "dataset_root": str(config.dataset_root),
        "class_names": class_names,
        "source_backbone": source_description,
        "final_metrics": {
            "train_loss": final_record["train_loss"],
            "train_acc": final_record["train_acc"],
            "val_loss": final_record["val_loss"],
            "val_acc": final_record["val_acc"],
            "parameters": count_parameters(model),
        },
        "artifacts": {
            "history_json": str(history_path),
            "weights_path": str(weights_path),
            "curves_png": str(curve_path),
            "misclassified_json": str(error_json_path),
            "misclassified_png": str(error_plot_path) if error_plot_path else None,
        },
    }
    summary_path = save_json(output_dir / "day5_summary.json", summary)

    print("=== Day 5 Summary ===")
    print(f"Final train accuracy: {final_record['train_acc']:.2%}")
    print(f"Final val accuracy:   {final_record['val_acc']:.2%}")
    print(f"Results saved under:  {output_dir}")
    print(f"Summary file:         {summary_path}")


if __name__ == "__main__":
    run_transfer_learning()
