import json
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import ImageDraw
from torchvision.transforms import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def plot_learning_curves(history, output_path=None):
    """
    Plot train/val loss and val detection metrics from train.py history.
    """
    if not isinstance(history, dict):
        history_path = _resolve_path(history)
        with history_path.open() as file:
            history = json.load(file)

    records = history.get("epochs", [])
    if not records:
        raise ValueError("Training history has no epochs to plot.")

    epochs = [record["epoch"] for record in records]
    train_loss = [record["train"]["total_loss"] for record in records]
    val_loss = [record["val"]["total_loss"] for record in records]
    val_ap50 = [record["val"]["ap50"] for record in records]
    val_ap75 = [record["val"]["ap75"] for record in records]
    val_map = [record["val"]["map_50_95"] for record in records]
    best_epoch = history.get("best_epoch")
    best_val_map = history.get("best_val_map_50_95")

    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.5))

    axes[0].plot(epochs, train_loss, label="Train loss")
    axes[0].plot(epochs, val_loss, label="Val loss")
    axes[0].set_title("Train vs Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")

    axes[1].plot(epochs, val_ap50, label="AP50")
    axes[1].plot(epochs, val_ap75, label="AP75")
    axes[1].plot(epochs, val_map, label="mAP50:95")
    axes[1].set_title("Validation Detection Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Average Precision")
    axes[1].set_ylim(0.0, 1.0)

    if best_epoch is not None:
        best_label = f"Best epoch {best_epoch}"
        if best_val_map is not None:
            best_label += f" (mAP {best_val_map:.4f})"
        for axis in axes:
            axis.axvline(
                best_epoch,
                color="black",
                linestyle="--",
                linewidth=1.0,
                label=best_label,
            )

    for axis in axes:
        axis.legend()
        axis.grid(True, alpha=0.3)

    figure.tight_layout()

    if output_path is not None:
        output_path = _resolve_path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=150, bbox_inches="tight")

    return figure


def save_detection_visualization(
    image,
    target,
    detection,
    output_path,
    score_threshold=0.1,
):
    """Draw ground-truth and predicted pedestrian boxes on one image."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rendered_image = F.to_pil_image(
        image.detach().cpu().clamp(0.0, 1.0)
    )
    draw = ImageDraw.Draw(rendered_image)

    ground_truth_boxes = target["boxes"].detach().cpu()
    for box in ground_truth_boxes:
        x1, y1, x2, y2 = box.tolist()
        draw.rectangle(
            (x1, y1, x2, y2),
            outline="red",
            width=3,
        )
        draw.text(
            (x1, max(0.0, y1 - 12.0)),
            "GT pedestrian",
            fill="red",
        )

    predicted_boxes = detection["boxes"].detach().cpu()
    predicted_scores = detection["scores"].detach().cpu()

    for box, score in zip(predicted_boxes, predicted_scores):
        if float(score) < score_threshold:
            continue

        x1, y1, x2, y2 = box.tolist()
        draw.rectangle(
            (x1, y1, x2, y2),
            outline="lime",
            width=2,
        )
        draw.text(
            (x1, min(float(rendered_image.height - 12), y2 + 2.0)),
            f"Pred {float(score):.2f}",
            fill="lime",
        )

    rendered_image.save(output_path)
    return rendered_image
