import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import ImageDraw
from torchvision.transforms import functional as F
from torchvision.ops import box_iou

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def history_to_frame(history):
    """Flatten train.py / train_v2.py JSON history into one row per epoch."""
    if not isinstance(history, dict):
        history_path = _resolve_path(history)
        with history_path.open() as file:
            history = json.load(file)

    records = history.get("epochs", [])
    if not records:
        raise ValueError("Training history has no epochs to plot.")

    rows = []
    for record in records:
        row = {
            "epoch": record["epoch"],
            "is_best": record.get("is_best", False),
        }
        for split in ("train", "val"):
            values = record.get(split)
            if values is not None:
                row.update(
                    {
                        f"{split}_{name}": value
                        for name, value in values.items()
                    }
                )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_training_curves(
    history,
    loss_title="Training and validation loss",
    figsize=(14, 4.5),
    output_path=None,
):
    """Plot train/val loss and validation AP from a history dict or JSON path."""
    if not isinstance(history, dict):
        history_path = _resolve_path(history)
        with history_path.open() as file:
            history = json.load(file)

    history_frame = history_to_frame(history)
    best_epoch = history.get("best_epoch")

    figure, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].plot(
        history_frame["epoch"],
        history_frame["train_total_loss"],
        label="Train total loss",
    )
    axes[0].plot(
        history_frame["epoch"],
        history_frame["val_total_loss"],
        marker="o",
        label="Validation total loss",
    )
    if best_epoch is not None:
        axes[0].axvline(
            best_epoch,
            color="black",
            linestyle="--",
            alpha=0.7,
            label=f"Best epoch {best_epoch}",
        )
    axes[0].set(title=loss_title, xlabel="Epoch", ylabel="Loss")
    axes[0].legend()

    for column, label in (
        ("val_ap50", "AP50"),
        ("val_ap75", "AP75"),
        ("val_map_50_95", "mAP50:95"),
    ):
        axes[1].plot(
            history_frame["epoch"],
            history_frame[column],
            marker="o",
            label=label,
        )
    if best_epoch is not None:
        axes[1].axvline(
            best_epoch,
            color="black",
            linestyle="--",
            alpha=0.7,
        )
    axes[1].set(
        title="Validation detection metrics",
        xlabel="Epoch",
        ylabel="Score",
        ylim=(0, None),
    )
    axes[1].legend()
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


QUALITATIVE_CATEGORIES = (
    "correct_detection",
    "missed_pedestrian",
    "localization_error",
    "false_positive",
)

QUALITATIVE_CATEGORY_LABELS = {
    "correct_detection": "Correct detection",
    "missed_pedestrian": "Missed pedestrian",
    "localization_error": "Localization error",
    "false_positive": "False positive",
}


def classify_qualitative_case(
    target,
    detection,
    score_threshold=0.1,
    iou_threshold=0.5,
    localization_iou_threshold=0.1,
):
    """Assign one diagnostic category to a detection sample."""
    ground_truth_boxes = target["boxes"].detach().cpu().float()
    predicted_scores = detection["scores"].detach().cpu().float()
    predicted_boxes = detection["boxes"].detach().cpu().float()
    predicted_boxes = predicted_boxes[
        predicted_scores >= score_threshold
    ]

    ground_truth_count = len(ground_truth_boxes)
    prediction_count = len(predicted_boxes)
    best_iou = 0.0

    if ground_truth_count == 0:
        category = "false_positive" if prediction_count else None
    elif prediction_count == 0:
        category = "missed_pedestrian"
    else:
        best_iou = float(
            box_iou(
                predicted_boxes,
                ground_truth_boxes,
            ).max()
        )
        if best_iou >= iou_threshold:
            category = "correct_detection"
        elif best_iou >= localization_iou_threshold:
            category = "localization_error"
        else:
            category = "missed_pedestrian"

    return {
        "category": category,
        "ground_truth_count": ground_truth_count,
        "prediction_count": prediction_count,
        "best_iou": best_iou,
    }


class QualitativeGalleryCollector:
    """Collect deterministic, scene-diverse detection examples."""

    def __init__(
        self,
        output_directory,
        max_examples=8,
        score_threshold=0.1,
        iou_threshold=0.5,
        localization_iou_threshold=0.1,
    ):
        if max_examples < 1:
            raise ValueError("max_examples must be at least 1.")

        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.score_threshold = float(score_threshold)
        self.iou_threshold = float(iou_threshold)
        self.localization_iou_threshold = float(
            localization_iou_threshold
        )
        base, remainder = divmod(
            int(max_examples),
            len(QUALITATIVE_CATEGORIES),
        )
        self.quotas = {
            category: base + (index < remainder)
            for index, category in enumerate(
                QUALITATIVE_CATEGORIES
            )
        }
        self.records = []
        self.selected_scenes = set()

    @property
    def is_full(self):
        counts = {
            category: 0 for category in QUALITATIVE_CATEGORIES
        }
        for record in self.records:
            counts[record["category"]] += 1
        return all(
            counts[category] >= quota
            for category, quota in self.quotas.items()
        )

    def consider(self, image, target, detection, metadata):
        case = classify_qualitative_case(
            target=target,
            detection=detection,
            score_threshold=self.score_threshold,
            iou_threshold=self.iou_threshold,
            localization_iou_threshold=(
                self.localization_iou_threshold
            ),
        )
        category = case["category"]
        if category is None:
            return False

        category_count = sum(
            record["category"] == category
            for record in self.records
        )
        scene_name = metadata["scene_name"]
        if (
            category_count >= self.quotas[category]
            or scene_name in self.selected_scenes
        ):
            return False

        sample_token = metadata["sample_token"]
        filename = (
            f"{len(self.records):02d}_{category}_"
            f"{sample_token}.png"
        )
        save_detection_visualization(
            image=image,
            target=target,
            detection=detection,
            output_path=self.output_directory / filename,
            score_threshold=self.score_threshold,
        )

        self.records.append({
            **case,
            "category_label": QUALITATIVE_CATEGORY_LABELS[
                category
            ],
            "scene_name": scene_name,
            "sample_token": sample_token,
            "filename": filename,
        })
        self.selected_scenes.add(scene_name)
        return True

    def save_manifest(self):
        manifest = {
            "selection": "deterministic_stratified",
            "score_threshold": self.score_threshold,
            "iou_threshold": self.iou_threshold,
            "localization_iou_threshold": (
                self.localization_iou_threshold
            ),
            "quotas": self.quotas,
            "examples": self.records,
        }
        manifest_path = self.output_directory / "gallery.json"
        with manifest_path.open("w") as file:
            json.dump(manifest, file, indent=2)
        return manifest_path


@torch.no_grad()
def generate_detection_gallery(
    model,
    dataloader,
    postprocessor,
    device,
    output_directory,
    max_examples=8,
    score_threshold=0.1,
    iou_threshold=0.5,
    progress_interval=50,
):
    """Run normal inference until a stratified gallery is full."""
    device = torch.device(device)
    model.eval()
    collector = QualitativeGalleryCollector(
        output_directory=output_directory,
        max_examples=max_examples,
        score_threshold=score_threshold,
        iou_threshold=iou_threshold,
    )

    for batch_index, batch in enumerate(dataloader, start=1):
        predictions = model(
            images=batch["images"].to(device),
            radar_points=batch["radar_points"].to(device),
            radar_padding_mask=batch[
                "radar_padding_mask"
            ].to(device),
        )
        detections = postprocessor(predictions)

        for sample_index, detection in enumerate(detections):
            collector.consider(
                image=batch["images"][sample_index],
                target=batch["targets"][sample_index],
                detection=detection,
                metadata=batch["metadata"][sample_index],
            )

        if (
            batch_index % progress_interval == 0
            or collector.is_full
            or batch_index == len(dataloader)
        ):
            print(
                "[qualitative] batch "
                f"{batch_index}/{len(dataloader)} | "
                f"examples {len(collector.records)}/"
                f"{max_examples}",
                flush=True,
            )

        if collector.is_full:
            break

    return collector.save_manifest()


def plot_detection_gallery(
    manifest_path,
    max_images=None,
    columns=2,
):
    """Render a qualitative gallery described by a JSON manifest."""
    manifest_path = Path(manifest_path)
    with manifest_path.open() as file:
        manifest = json.load(file)

    examples = manifest.get("examples", [])
    if max_images is not None:
        examples = examples[:max_images]
    if not examples:
        raise ValueError(
            f"Qualitative gallery has no examples: {manifest_path}"
        )

    rows = math.ceil(len(examples) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(7.5 * columns, 4.8 * rows),
        squeeze=False,
    )
    axes = axes.ravel()

    for axis, example in zip(axes, examples):
        image_path = manifest_path.parent / example["filename"]
        axis.imshow(plt.imread(image_path))
        axis.set_title(
            f"{example['category_label']} | "
            f"{example['scene_name']}\n"
            f"GT {example['ground_truth_count']} | "
            f"Pred {example['prediction_count']} | "
            f"best IoU {example['best_iou']:.2f}",
            fontsize=9,
        )
        axis.axis("off")

    for axis in axes[len(examples):]:
        axis.axis("off")

    figure.suptitle(
        "Red: ground truth | Green: prediction",
        fontsize=12,
    )
    figure.tight_layout()
    return figure
