import json
from pathlib import Path

import torch

from src.evaluation.metrics import PedestrianDetectionMetrics
from src.utils import select_device


RADAR_MODES = ("normal", "masked", "shuffled")


def load_detailed_results(*paths, missing_hint=None):
    """Load the first cached result that contains every Radar mode."""
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue

        with path.open() as file:
            candidate = json.load(file)

        if set(candidate.get("modes", {})) >= set(RADAR_MODES):
            return candidate, path

    raise FileNotFoundError(
        missing_hint
        or (
            "No detailed normal/masked/shuffled result was found. "
            "Run a detailed evaluation first."
        )
    )


def build_detection_metrics(evaluation_config):
    return PedestrianDetectionMetrics(
        iou_thresholds=evaluation_config["iou_thresholds"],
        report_iou_threshold=evaluation_config[
            "report_iou_threshold"
        ],
        report_score_threshold=evaluation_config[
            "report_score_threshold"
        ],
    )


def synchronize_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def resolve_device(device, evaluation_config):
    if device is None or device == "auto":
        return select_device(evaluation_config["device"])
    return torch.device(device)



def apply_radar_mode(
    radar_points,
    radar_padding_mask,
    mode,
    seed=42,
):
    """
    Create one Radar inference ablation without changing the batch.
    """
    if mode not in RADAR_MODES:
        raise ValueError(f"Unknown Radar mode: {mode}")

    points = radar_points.clone()
    padding_mask = radar_padding_mask.clone()

    if mode == "normal":
        return points, padding_mask

    if mode == "masked":
        padding_mask.fill_(True) # set all padding mask to True, block all radar points
        return points, padding_mask

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    # shuffle the radar points (u, v) in each batch
    for batch_index in range(points.shape[0]):
        valid_indices = torch.where(
            ~padding_mask[batch_index]
        )[0]
        count = valid_indices.numel()
        if count < 2:
            continue

        order = torch.randperm(count, generator=generator)
        if torch.equal(order, torch.arange(count)):
            order = torch.roll(order, shifts=1)

        # only the first two dimensions (u, v) are shuffled
        shuffled_indices = valid_indices[order]
        points[batch_index, valid_indices, :2] = points[
            batch_index,
            shuffled_indices,
            :2,
        ]

    return points, padding_mask
