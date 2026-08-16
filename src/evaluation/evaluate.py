import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data_loader.nuscenes_front_loader import NuScenesFrontLoader, collate_fn
from src.evaluation.metrics import PedestrianDetectionMetrics
from src.evaluation.visualization import save_detection_visualization
from src.model.detector import CameraRadarDetector
from src.model.postprocess import CenterNetPostProcessor


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def select_device(requested_device):
    if requested_device != "auto":
        return torch.device(requested_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_evaluation(config=None):
    if config is None:
        config = load_config()

    evaluation_config = config["evaluation"]
    device = select_device(evaluation_config["device"])
    checkpoint_path = resolve_path(evaluation_config["checkpoint"])
    output_directory = resolve_path(evaluation_config["output_dir"])

    dataset = NuScenesFrontLoader(
        dataroot=resolve_path(evaluation_config["dataroot"]),
        split=evaluation_config["split"],
        image_size=config["image_size"],
        radar_channels=tuple(config["radar"]["channels"]),
        nsweeps=config["radar"]["nsweeps"],
        camera_channel=config["camera_channel"],
        class_name=config["class_name"],
    )

    number_of_samples = evaluation_config["num_samples"]
    if number_of_samples is not None:
        number_of_samples = min(number_of_samples, len(dataset))
        dataset = Subset(dataset, range(number_of_samples))

    dataloader = DataLoader(
        dataset,
        batch_size=evaluation_config["batch_size"],
        shuffle=False,
        num_workers=evaluation_config["num_workers"],
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )

    model = CameraRadarDetector(
        image_size=config["image_size"],
        freeze_camera=True,
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    postprocessor = CenterNetPostProcessor(
        image_size=config["image_size"],
        score_threshold=evaluation_config["score_threshold"],
        top_k=evaluation_config["top_k"],
    )
    metrics = PedestrianDetectionMetrics(
        iou_thresholds=evaluation_config["iou_thresholds"],
        report_iou_threshold=evaluation_config["report_iou_threshold"],
        report_score_threshold=evaluation_config["report_score_threshold"],
    )

    output_directory.mkdir(parents=True, exist_ok=True)
    visualized_samples = 0

    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader, start=1):
            predictions = model(
                images=batch["images"].to(device),
                radar_points=batch["radar_points"].to(device),
                radar_padding_mask=batch["radar_padding_mask"].to(device),
            )
            detections = postprocessor(predictions)
            metrics.update(detections, batch["targets"])

            if evaluation_config["save_visualizations"]:
                for sample_index, detection in enumerate(detections):
                    if visualized_samples >= evaluation_config["max_visualizations"]:
                        break

                    sample_token = batch["metadata"][sample_index][
                        "sample_token"
                    ]
                    output_path = (
                        output_directory
                        / "visualizations"
                        / f"{visualized_samples:04d}_{sample_token}.png"
                    )
                    save_detection_visualization(
                        image=batch["images"][sample_index],
                        target=batch["targets"][sample_index],
                        detection=detection,
                        output_path=output_path,
                        score_threshold=evaluation_config[
                            "visualization_score_threshold"
                        ],
                    )
                    visualized_samples += 1

            print(f"Evaluated batch {batch_index}/{len(dataloader)}")

    results = metrics.compute()
    results.update(
        {
            "split": evaluation_config["split"],
            "samples": len(dataset),
            "checkpoint": str(checkpoint_path),
        }
    )

    metrics_path = output_directory / "metrics.json"
    with metrics_path.open("w") as file:
        json.dump(results, file, indent=2)

    print(f"AP50: {results['ap50']:.4f}")
    print(f"AP75: {results['ap75']:.4f}")
    print(f"mAP50:95: {results['map_50_95']:.4f}")
    print(
        f"Precision@{results['report_score_threshold']:.2f}: "
        f"{results['precision']:.4f}"
    )
    print(
        f"Recall@{results['report_score_threshold']:.2f}: "
        f"{results['recall']:.4f}"
    )
    print(f"Metrics: {metrics_path}")

    return results


if __name__ == "__main__":
    run_evaluation()
