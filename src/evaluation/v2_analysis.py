import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryPrecision,
    BinaryRecall,
)

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
from src.model.detector_v2 import CameraRadarDetector
from src.model.postprocess import CenterNetPostProcessor
from src.utils import resolve_path, select_device


RADAR_MODES = ("normal", "masked", "shuffled")


def load_detailed_results(*paths):
    """Load the first cached result containing every Radar mode."""
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue

        with path.open() as file:
            candidate = json.load(file)

        if set(candidate.get("modes", {})) >= set(RADAR_MODES):
            return candidate, path

    raise FileNotFoundError(
        "No detailed normal/masked/shuffled result was found. "
        "Run run_v2_detailed_evaluation() first."
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


def _empty_fusion_totals():
    names = (
        "occupied_windows",
        "local_radar_points",
        "local_attention_pairs",
        "global_attention_pairs",
    )
    return {
        scale: {name: 0 for name in names}
        for scale in ("s4", "s8", "s16")
    }


def _accumulate_fusion_diagnostics(totals, diagnostics):
    for scale, values in diagnostics.items():
        for name in totals[scale]:
            totals[scale][name] += int(values[name])


def _finalize_fusion_diagnostics(totals, number_of_samples):
    results = {}

    for scale, values in totals.items():
        occupied = values["occupied_windows"]
        local_pairs = values["local_attention_pairs"]
        global_pairs = values["global_attention_pairs"]

        results[scale] = {
            **values,
            "mean_occupied_windows_per_sample": (
                occupied / number_of_samples
            ),
            "mean_local_radar_points_per_sample": (
                values["local_radar_points"]
                / number_of_samples
            ),
            "mean_radar_points_per_occupied_window": (
                values["local_radar_points"] / occupied
                if occupied
                else 0.0
            ),
            "mean_attention_pairs_per_occupied_window": (
                local_pairs / occupied
                if occupied
                else 0.0
            ),
            "local_to_global_pair_ratio": (
                local_pairs / global_pairs
                if global_pairs
                else 0.0
            ),
        }

    return results


def _update_subset_metrics(
    supported_metrics,
    unsupported_metrics,
    detections,
    targets,
    supported,
):
    supported_detections = []
    supported_targets = []
    unsupported_detections = []
    unsupported_targets = []

    for detection, target, has_support in zip(
        detections,
        targets,
        supported.tolist(),
    ):
        if has_support:
            supported_detections.append(detection)
            supported_targets.append(target)
        else:
            unsupported_detections.append(detection)
            unsupported_targets.append(target)

    if supported_detections:
        supported_metrics.update(
            supported_detections,
            supported_targets,
        )
    if unsupported_detections:
        unsupported_metrics.update(
            unsupported_detections,
            unsupported_targets,
        )

    return len(supported_detections), len(unsupported_detections)


def _relevance_threshold_curve(probabilities, targets):
    thresholds = torch.linspace(0.0, 1.0, 101)
    predictions = (
        probabilities[:, None] >= thresholds[None, :]
    )
    positives = targets[:, None].bool()

    true_positives = (predictions & positives).sum(dim=0).float()
    false_positives = (
        predictions & ~positives
    ).sum(dim=0).float()
    false_negatives = (
        ~predictions & positives
    ).sum(dim=0).float()

    precision = true_positives / (
        true_positives + false_positives
    ).clamp_min(1.0)
    recall = true_positives / (
        true_positives + false_negatives
    ).clamp_min(1.0)
    f1 = (
        2.0
        * precision
        * recall
        / (precision + recall).clamp_min(1e-12)
    )
    best_index = int(f1.argmax())

    return {
        "thresholds": thresholds.tolist(),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "best_f1_threshold": float(thresholds[best_index]),
        "best_f1": float(f1[best_index]),
    }


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

    fusion = config["model"]["fusion"]
    model = CameraRadarDetector(
        image_size=config["image_size"],
        window_size=fusion["window_size"],
        vertical_neighbor_windows=fusion[
            "vertical_neighbor_windows"
        ],
        window_batch_bucket_size=fusion[
            "window_batch_bucket_size"
        ],
        dropout=fusion["dropout"],
        freeze_camera=True,
    )
    load_result = model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            "V2 checkpoint did not load strictly: "
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
    supported_metrics = _build_detection_metrics(
        evaluation_config
    )
    unsupported_metrics = _build_detection_metrics(
        evaluation_config
    )

    supported_samples = 0
    unsupported_samples = 0
    fusion_totals = _empty_fusion_totals()
    elapsed_seconds = 0.0
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

    relevance_precision = BinaryPrecision(threshold=0.5)
    relevance_recall = BinaryRecall(threshold=0.5)
    relevance_auroc = BinaryAUROC()
    relevance_probabilities = []
    relevance_targets = []

    for batch_index, batch in enumerate(dataloader, start=1):
        radar_points, radar_padding_mask = apply_radar_mode(
            batch["radar_points"],
            batch["radar_padding_mask"],
            mode,
            seed=42 + batch_index,
        )
        original_padding_mask = batch["radar_padding_mask"]
        supported = (
            (batch["radar_relevance_targets"] == 1.0)
            & ~batch["radar_relevance_ignore_mask"]
            & ~original_padding_mask
        ).any(dim=1)

        _synchronize_device(device)
        started_at = time.perf_counter()
        predictions = model(
            images=batch["images"].to(device),
            radar_points=radar_points.to(device),
            radar_padding_mask=radar_padding_mask.to(device),
            return_diagnostics=True,
        )
        detections = postprocessor(predictions)
        _synchronize_device(device)
        elapsed_seconds += time.perf_counter() - started_at

        metrics.update(detections, batch["targets"])
        added_supported, added_unsupported = (
            _update_subset_metrics(
                supported_metrics,
                unsupported_metrics,
                detections,
                batch["targets"],
                supported,
            )
        )
        supported_samples += added_supported
        unsupported_samples += added_unsupported
        _accumulate_fusion_diagnostics(
            fusion_totals,
            predictions["fusion_diagnostics"],
        )

        if mode == "normal":
            valid = (
                ~original_padding_mask
                & ~batch["radar_relevance_ignore_mask"]
            )
            if bool(valid.any()):
                probabilities = torch.sigmoid(
                    predictions[
                        "radar_relevance_logits"
                    ].detach().cpu()
                )[valid]
                targets = batch[
                    "radar_relevance_targets"
                ][valid].long()
                relevance_precision.update(
                    probabilities,
                    targets,
                )
                relevance_recall.update(
                    probabilities,
                    targets,
                )
                relevance_auroc.update(
                    probabilities,
                    targets,
                )
                relevance_probabilities.append(probabilities)
                relevance_targets.append(targets)

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
    result["radar_supported"] = {
        "samples": supported_samples,
        "metrics": (
            supported_metrics.compute()
            if supported_samples
            else None
        ),
    }
    result["radar_unsupported"] = {
        "samples": unsupported_samples,
        "metrics": (
            unsupported_metrics.compute()
            if unsupported_samples
            else None
        ),
    }
    result["fusion_diagnostics"] = (
        _finalize_fusion_diagnostics(
            fusion_totals,
            number_of_samples,
        )
    )
    result["local_latency_ms_per_sample"] = (
        elapsed_seconds * 1000.0 / number_of_samples
    )

    if mode == "normal" and relevance_probabilities:
        all_probabilities = torch.cat(relevance_probabilities)
        all_targets = torch.cat(relevance_targets)
        result["radar_relevance"] = {
            "labelled_points": int(all_targets.numel()),
            "precision_at_0.5": float(
                relevance_precision.compute()
            ),
            "recall_at_0.5": float(
                relevance_recall.compute()
            ),
            "auroc": float(relevance_auroc.compute()),
            "threshold_curve": _relevance_threshold_curve(
                all_probabilities,
                all_targets,
            ),
        }
    else:
        result["radar_relevance"] = None

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


def generate_v2_qualitative_gallery(
    config,
    checkpoint_path,
    output_directory,
    device=None,
    sample_limit=None,
    max_visualizations=8,
    progress_interval=50,
):
    """Generate only the V2 normal-mode qualitative gallery."""
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


def run_v2_detailed_evaluation(
    config,
    checkpoint_path,
    output_directory,
    device=None,
    sample_limit=None,
    max_visualizations=10,
    progress_interval=50,
):
    """Run normal/masked/shuffled V2 evaluation and save JSON."""
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
    visualization_directory = (
        output_directory / "visualizations"
    )
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
