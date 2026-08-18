import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from src.data_loader.nuscenes_front_loader import (
    NuScenesFrontLoader,
    collate_fn,
    load_scene_names,
)
from src.evaluation.metrics import PedestrianDetectionMetrics
from src.evaluation.visualization import (
    QualitativeGalleryCollector,
    generate_detection_gallery,
)
from src.model.detector import CameraRadarDetector
from src.model.postprocess import CenterNetPostProcessor
from src.utils import resolve_path, select_device


RADAR_MODES = ("normal", "masked", "shuffled")


def load_detailed_results(*paths):
    """Load the first cached result containing every V1 Radar mode."""
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue

        with path.open() as file:
            candidate = json.load(file)

        if set(candidate.get("modes", {})) >= set(RADAR_MODES):
            return candidate, path

    raise FileNotFoundError(
        "No detailed V1 normal/masked/shuffled result was found. "
        "Run run_v1_detailed_evaluation() first."
    )


def apply_radar_mode(
    radar_points,
    radar_padding_mask,
    mode,
    seed=42,
):
    """Create one Radar inference ablation without changing the batch."""
    if mode not in RADAR_MODES:
        raise ValueError(f"Unknown Radar mode: {mode}")

    points = radar_points.clone()
    padding_mask = radar_padding_mask.clone()

    if mode == "normal":
        return points, padding_mask

    if mode == "masked":
        padding_mask.fill_(True)
        return points, padding_mask

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

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

        shuffled_indices = valid_indices[order]
        points[batch_index, valid_indices, :2] = points[
            batch_index,
            shuffled_indices,
            :2,
        ]

    return points, padding_mask


def _build_detection_metrics(evaluation_config):
    return PedestrianDetectionMetrics(
        iou_thresholds=evaluation_config["iou_thresholds"],
        report_iou_threshold=evaluation_config[
            "report_iou_threshold"
        ],
        report_score_threshold=evaluation_config[
            "report_score_threshold"
        ],
    )


def _synchronize_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _resolve_device(device, evaluation_config):
    if device is None or device == "auto":
        return select_device(evaluation_config["device"])
    return torch.device(device)


def _build_runtime(
    config,
    checkpoint,
    device,
    sample_limit,
):
    evaluation_config = config["evaluation"]
    dataset_config = config.get("dataset", {})
    manifest = dataset_config.get(
        "scene_manifests",
        {},
    ).get(evaluation_config["split"])
    scene_names = (
        load_scene_names(resolve_path(manifest))
        if manifest
        else None
    )

    dataset = NuScenesFrontLoader(
        dataroot=resolve_path(evaluation_config["dataroot"]),
        split=evaluation_config["split"],
        image_size=config["image_size"],
        radar_channels=tuple(config["radar"]["channels"]),
        nsweeps=config["radar"]["nsweeps"],
        camera_channel=config["camera_channel"],
        class_name=config["class_name"],
        version=dataset_config.get("version", "v1.0-mini"),
        available_scenes_only=dataset_config.get(
            "available_scenes_only",
            False,
        ),
        scene_names=scene_names,
    )

    if sample_limit is not None:
        dataset = Subset(
            dataset,
            range(min(sample_limit, len(dataset))),
        )

    dataloader = DataLoader(
        dataset,
        batch_size=evaluation_config["batch_size"],
        shuffle=False,
        num_workers=evaluation_config["num_workers"],
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )

    model = CameraRadarDetector(image_size=config["image_size"])
    load_result = model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            "V1 checkpoint did not load strictly: "
            f"missing={load_result.missing_keys}, "
            f"unexpected={load_result.unexpected_keys}"
        )
    model = model.to(device).eval()

    postprocessor = CenterNetPostProcessor(
        image_size=config["image_size"],
        score_threshold=evaluation_config["score_threshold"],
        top_k=evaluation_config["top_k"],
    )
    return dataset, dataloader, model, postprocessor


@torch.no_grad()
def _run_evaluation_mode(
    model,
    dataloader,
    postprocessor,
    evaluation_config,
    device,
    mode,
    visualization_directory,
    max_visualizations,
    progress_interval,
):
    metrics = _build_detection_metrics(evaluation_config)
    elapsed_seconds = 0.0
    valid_pairs = 0
    null_mass_sum = 0.0
    normalized_entropy_sum = 0.0
    observed_samples = 0
    gallery_collector = None

    if visualization_directory is not None and mode == "normal":
        gallery_collector = QualitativeGalleryCollector(
            output_directory=visualization_directory,
            max_examples=max_visualizations,
            score_threshold=evaluation_config[
                "visualization_score_threshold"
            ],
            iou_threshold=evaluation_config[
                "report_iou_threshold"
            ],
        )

    for batch_index, batch in enumerate(dataloader, start=1):
        radar_points, radar_padding_mask = apply_radar_mode(
            batch["radar_points"],
            batch["radar_padding_mask"],
            mode,
            seed=42 + batch_index,
        )

        _synchronize_device(device)
        started_at = time.perf_counter()
        predictions = model(
            images=batch["images"].to(device),
            radar_points=radar_points.to(device),
            radar_padding_mask=radar_padding_mask.to(device),
            return_attention=True,
        )
        detections = postprocessor(predictions)
        _synchronize_device(device)
        elapsed_seconds += time.perf_counter() - started_at
        metrics.update(detections, batch["targets"])

        attention = predictions["attention_weights"].detach()
        batch_size, _, query_count, _ = attention.shape
        valid_key_counts = (
            (~radar_padding_mask).sum(dim=1) + 1
        )
        valid_pairs += int((valid_key_counts * query_count).sum())
        null_mass_sum += float(
            attention[..., 0].mean(dim=(1, 2)).sum()
        )
        entropy = -(
            attention.clamp_min(1e-12).log() * attention
        ).sum(dim=-1).mean(dim=(1, 2))
        entropy_denominator = (
            valid_key_counts.float().log().clamp_min(1.0)
        ).to(entropy.device)
        normalized_entropy_sum += float(
            (entropy / entropy_denominator).sum()
        )
        observed_samples += batch_size

        if gallery_collector is not None:
            for sample_index, detection in enumerate(detections):
                gallery_collector.consider(
                    image=batch["images"][sample_index],
                    target=batch["targets"][sample_index],
                    detection=detection,
                    metadata=batch["metadata"][sample_index],
                )

        if (
            batch_index % progress_interval == 0
            or batch_index == len(dataloader)
        ):
            print(
                f"[{mode}] batch {batch_index}/"
                f"{len(dataloader)}",
                flush=True,
            )

    number_of_samples = len(dataloader.dataset)
    result = metrics.compute()
    result["attention_diagnostics"] = {
        "valid_global_pairs": valid_pairs,
        "mean_valid_global_pairs_per_sample": (
            valid_pairs / number_of_samples
        ),
        "mean_null_attention_mass": (
            null_mass_sum / observed_samples
        ),
        "mean_normalized_attention_entropy": (
            normalized_entropy_sum / observed_samples
        ),
    }
    result["latency_ms_per_sample"] = (
        elapsed_seconds * 1000.0 / number_of_samples
    )

    if gallery_collector is not None:
        gallery_manifest_path = gallery_collector.save_manifest()
        result["qualitative_gallery"] = {
            "manifest": str(gallery_manifest_path),
            "examples": len(gallery_collector.records),
            "complete": gallery_collector.is_full,
        }
    else:
        result["qualitative_gallery"] = None

    return result


def generate_v1_qualitative_gallery(
    config,
    checkpoint_path,
    output_directory,
    device=None,
    sample_limit=None,
    max_visualizations=8,
    progress_interval=50,
):
    """Generate only the V1 normal-mode qualitative gallery."""
    evaluation_config = config["evaluation"]
    device = _resolve_device(device, evaluation_config)
    checkpoint = torch.load(
        resolve_path(checkpoint_path),
        map_location="cpu",
        weights_only=False,
    )
    _, dataloader, model, postprocessor = _build_runtime(
        config=config,
        checkpoint=checkpoint,
        device=device,
        sample_limit=sample_limit,
    )
    return generate_detection_gallery(
        model=model,
        dataloader=dataloader,
        postprocessor=postprocessor,
        device=device,
        output_directory=resolve_path(output_directory),
        max_examples=max_visualizations,
        score_threshold=evaluation_config[
            "visualization_score_threshold"
        ],
        iou_threshold=evaluation_config[
            "report_iou_threshold"
        ],
        progress_interval=progress_interval,
    )


def run_v1_detailed_evaluation(
    config,
    checkpoint_path,
    output_directory,
    device=None,
    sample_limit=None,
    max_visualizations=8,
    progress_interval=50,
):
    """Run normal/masked/shuffled V1 evaluation and save JSON."""
    evaluation_config = config["evaluation"]
    device = _resolve_device(device, evaluation_config)
    checkpoint_path = resolve_path(checkpoint_path)
    output_directory = resolve_path(output_directory)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    dataset, dataloader, model, postprocessor = _build_runtime(
        config=config,
        checkpoint=checkpoint,
        device=device,
        sample_limit=sample_limit,
    )

    output_directory.mkdir(parents=True, exist_ok=True)
    visualization_directory = output_directory / "visualizations"
    modes = {}

    for mode in RADAR_MODES:
        modes[mode] = _run_evaluation_mode(
            model=model,
            dataloader=dataloader,
            postprocessor=postprocessor,
            evaluation_config=evaluation_config,
            device=device,
            mode=mode,
            visualization_directory=(
                visualization_directory
                if mode == "normal"
                else None
            ),
            max_visualizations=max_visualizations,
            progress_interval=progress_interval,
        )

    metric_names = (
        "ap50",
        "ap75",
        "map_50_95",
        "map_small",
        "map_medium",
        "map_large",
        "precision",
        "recall",
    )
    results = {
        "split": evaluation_config["split"],
        "samples": len(dataset),
        "checkpoint": str(checkpoint_path),
        "modes": modes,
        "deltas": {
            "normal_minus_masked": {
                name: modes["normal"][name]
                - modes["masked"][name]
                for name in metric_names
            },
            "normal_minus_shuffled": {
                name: modes["normal"][name]
                - modes["shuffled"][name]
                for name in metric_names
            },
        },
    }

    metrics_path = output_directory / "metrics.json"
    with metrics_path.open("w") as file:
        json.dump(results, file, indent=2)

    return results, metrics_path
