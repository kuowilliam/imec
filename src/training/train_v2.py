import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import Subset


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import load_config
from src.evaluation.metrics import PedestrianDetectionMetrics
from src.model.detector_v2 import CameraRadarDetector
from src.model.loss_v2 import CenterNetLoss
from src.model.postprocess import CenterNetPostProcessor
from src.training.train_utils import (
    average_losses,
    build_dataloader,
    move_targets_to_device,
    save_checkpoint,
    save_history,
    should_validate,
)
from src.utils import resolve_path, select_device


def empty_loss_totals():
    return {
        "total_loss": 0.0,
        "heatmap_loss": 0.0,
        "box_size_loss": 0.0,
        "offset_loss": 0.0,
        "radar_relevance_loss": 0.0,
    }


def train_one_epoch(
    model,
    criterion,
    dataloader,
    optimizer,
    device,
    epoch=None,
    total_epochs=None,
    progress_interval=25,
):
    model.train()

    if progress_interval < 1:
        raise ValueError(
            "progress_interval must be at least 1."
        )

    running_losses = empty_loss_totals()
    started_at = time.perf_counter()

    for batch_index, batch in enumerate(
        dataloader,
        start=1,
    ):
        images = batch["images"].to(device)
        radar_points = batch["radar_points"].to(device)
        radar_padding_mask = batch[
            "radar_padding_mask"
        ].to(device)
        radar_relevance_targets = batch[
            "radar_relevance_targets"
        ].to(device)
        radar_relevance_ignore_mask = batch[
            "radar_relevance_ignore_mask"
        ].to(device)
        targets = move_targets_to_device(
            batch["targets"],
            device,
        )

        optimizer.zero_grad(set_to_none=True)

        predictions = model(
            images=images,
            radar_points=radar_points,
            radar_padding_mask=radar_padding_mask,
        )
        losses = criterion(
            predictions,
            targets,
            radar_relevance_targets=(
                radar_relevance_targets
            ),
            radar_relevance_ignore_mask=(
                radar_relevance_ignore_mask
            ),
            radar_padding_mask=radar_padding_mask,
        )
        total_loss = losses["total_loss"]

        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                "Non-finite training loss: "
                f"{float(total_loss.detach())}"
            )

        total_loss.backward()
        optimizer.step()

        for name in running_losses:
            running_losses[name] = (
                running_losses[name]
                + losses[name].detach()
            )

        if (
            batch_index % progress_interval == 0
            or batch_index == len(dataloader)
        ):
            epoch_label = (
                f"Epoch {epoch:03d}/{total_epochs:03d} | "
                if (
                    epoch is not None
                    and total_epochs is not None
                )
                else ""
            )
            elapsed = time.perf_counter() - started_at
            average_total = float(
                running_losses["total_loss"]
                / batch_index
            )
            average_radar = float(
                running_losses["radar_relevance_loss"]
                / batch_index
            )
            print(
                f"{epoch_label}batch {batch_index:04d}/"
                f"{len(dataloader):04d} | "
                f"total={average_total:.4f} | "
                f"radar={average_radar:.4f} | "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    return average_losses(
        running_losses,
        len(dataloader),
    )


@torch.no_grad()
def validate_one_epoch(
    model,
    criterion,
    postprocessor,
    metrics,
    dataloader,
    device,
):
    model.eval()
    metrics.reset()

    running_losses = empty_loss_totals()

    for batch in dataloader:
        images = batch["images"].to(device)
        radar_points = batch["radar_points"].to(device)
        radar_padding_mask = batch[
            "radar_padding_mask"
        ].to(device)
        radar_relevance_targets = batch[
            "radar_relevance_targets"
        ].to(device)
        radar_relevance_ignore_mask = batch[
            "radar_relevance_ignore_mask"
        ].to(device)
        targets = move_targets_to_device(
            batch["targets"],
            device,
        )

        predictions = model(
            images=images,
            radar_points=radar_points,
            radar_padding_mask=radar_padding_mask,
        )
        losses = criterion(
            predictions,
            targets,
            radar_relevance_targets=(
                radar_relevance_targets
            ),
            radar_relevance_ignore_mask=(
                radar_relevance_ignore_mask
            ),
            radar_padding_mask=radar_padding_mask,
        )

        if not torch.isfinite(losses["total_loss"]):
            raise FloatingPointError(
                "Non-finite validation loss: "
                f"{float(losses['total_loss'].detach())}"
            )

        for name in running_losses:
            running_losses[name] = (
                running_losses[name]
                + losses[name].detach()
            )

        detections = postprocessor(predictions)
        metrics.update(detections, targets)

    detection_metrics = metrics.compute()
    return (
        average_losses(
            running_losses,
            len(dataloader),
        ),
        detection_metrics,
    )


def main():
    random.seed(42)
    torch.manual_seed(42)

    # load config
    config = load_config()
    train_cfg = config["train"]
    evaluation_cfg = config["evaluation"]
    model_cfg = config["model"]
    fusion_cfg = model_cfg["fusion"]
    relevance_cfg = model_cfg["radar_relevance"]

    if fusion_cfg["scales"] != ["s4", "s8", "s16"]:
        raise ValueError(
            "V2 requires fusion scales [s4, s8, s16]."
        )

    device = select_device(train_cfg["device"])
    dataroot = resolve_path(train_cfg["dataroot"])
    checkpoint_path = resolve_path(
        train_cfg["checkpoint"]
    )
    history_path = resolve_path(train_cfg["history"])
    patience = train_cfg.get(
        "early_stopping_patience"
    )
    validation_interval = train_cfg.get(
        "validation_interval",
        1,
    )

    train_dataset, train_dataloader = build_dataloader(
        config=config,
        dataroot=dataroot,
        split=train_cfg.get("split", "mini_train"),
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg["num_workers"],
        shuffle=True,
        device=device,
        num_samples=train_cfg["num_samples"],
        apply_augmentation=True,
        frame_stride=train_cfg.get("frame_stride", 1),
    )
    val_dataset, val_dataloader = build_dataloader(
        config=config,
        dataroot=dataroot,
        split=train_cfg.get("val_split", "mini_val"),
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg["num_workers"],
        shuffle=False,
        device=device,
    )

    model = CameraRadarDetector(
        image_size=config["image_size"],
        window_size=fusion_cfg["window_size"],
        vertical_neighbor_windows=fusion_cfg[
            "vertical_neighbor_windows"
        ],
        window_batch_bucket_size=fusion_cfg[
            "window_batch_bucket_size"
        ],
        dropout=fusion_cfg["dropout"],
        freeze_camera=True,
    ).to(device)
    relevance_weight = (
        relevance_cfg["loss_weight"]
        if relevance_cfg["enabled"]
        else 0.0
    )
    criterion = CenterNetLoss(
        image_size=config["image_size"],
        radar_relevance_weight=relevance_weight,
    ).to(device)
    postprocessor = CenterNetPostProcessor(
        image_size=config["image_size"],
        score_threshold=(
            evaluation_cfg["score_threshold"]
        ),
        top_k=evaluation_cfg["top_k"],
    )
    metrics = PedestrianDetectionMetrics(
        iou_thresholds=(
            evaluation_cfg["iou_thresholds"]
        ),
        report_iou_threshold=(
            evaluation_cfg["report_iou_threshold"]
        ),
        report_score_threshold=(
            evaluation_cfg["report_score_threshold"]
        ),
    )

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    train_source = (
        train_dataset.dataset
        if isinstance(train_dataset, Subset)
        else train_dataset
    )
    print(f"Device: {device}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(
        f"Train scenes: {len(train_source.scene_names)}"
    )
    print(f"Val scenes: {len(val_dataset.scene_names)}")
    print(
        "Train augmentation enabled: "
        f"{train_source.augmentation is not None}"
    )
    print(
        "Train batches per epoch: "
        f"{len(train_dataloader)}"
    )
    print(
        "Val batches per epoch: "
        f"{len(val_dataloader)}"
    )
    print(
        "Trainable parameters: "
        f"{sum(parameter.numel() for parameter in trainable_parameters):,}"
    )

    best_val_map = float("-inf")
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_early = False
    history = {
        "best_epoch": None,
        "best_val_map_50_95": None,
        "stopped_early": False,
        "validation_interval": validation_interval,
        "epochs": [],
    }
    started_at = time.perf_counter()

    for epoch in range(1, train_cfg["epochs"] + 1):
        train_losses = train_one_epoch(
            model=model,
            criterion=criterion,
            dataloader=train_dataloader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_epochs=train_cfg["epochs"],
            progress_interval=train_cfg.get(
                "progress_interval",
                25,
            ),
        )
        record = {
            "epoch": epoch,
            "train": dict(train_losses),
            "val": None,
            "is_best": False,
        }

        run_validation = should_validate(
            epoch,
            train_cfg["epochs"],
            validation_interval,
        )
        if run_validation:
            val_losses, val_metrics = validate_one_epoch(
                model=model,
                criterion=criterion,
                postprocessor=postprocessor,
                metrics=metrics,
                dataloader=val_dataloader,
                device=device,
            )

            val_map = float(val_metrics["map_50_95"])
            val_ap50 = float(val_metrics["ap50"])
            val_ap75 = float(val_metrics["ap75"])
            is_best = val_map > best_val_map
            record["val"] = {
                **val_losses,
                "ap50": val_ap50,
                "ap75": val_ap75,
                "map_50_95": val_map,
            }
            record["is_best"] = is_best

            print(
                f"Epoch {epoch:03d}/"
                f"{train_cfg['epochs']:03d} | "
                "train_total="
                f"{train_losses['total_loss']:.4f} | "
                "train_radar="
                f"{train_losses['radar_relevance_loss']:.4f} | "
                "val_total="
                f"{val_losses['total_loss']:.4f} | "
                "val_radar="
                f"{val_losses['radar_relevance_loss']:.4f} | "
                f"val_AP50={val_ap50:.4f} | "
                f"val_AP75={val_ap75:.4f} | "
                f"val_mAP={val_map:.4f}"
                f"{' *' if is_best else ''}"
            )

            if is_best:
                best_val_map = val_map
                best_epoch = epoch
                epochs_without_improvement = 0
                history["best_epoch"] = best_epoch
                history["best_val_map_50_95"] = (
                    best_val_map
                )
                save_checkpoint(
                    path=checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    losses={
                        "train": train_losses,
                        "val": val_losses,
                    },
                    metrics={
                        "ap50": val_ap50,
                        "ap75": val_ap75,
                        "map_50_95": val_map,
                    },
                    config=config,
                )
            else:
                epochs_without_improvement += 1
        else:
            print(
                f"Epoch {epoch:03d}/"
                f"{train_cfg['epochs']:03d} | "
                "train_total="
                f"{train_losses['total_loss']:.4f} | "
                "train_radar="
                f"{train_losses['radar_relevance_loss']:.4f} | "
                "validation=skipped"
            )

        history["epochs"].append(record)
        save_history(history_path, history)

        if (
            run_validation
            and patience is not None
            and epochs_without_improvement >= patience
        ):
            stopped_early = True
            history["stopped_early"] = True
            save_history(history_path, history)
            print(
                f"Early stopping at epoch {epoch}: "
                "val mAP50:95 did not improve for "
                f"{patience} validation checks "
                f"(best {best_val_map:.4f} "
                f"at epoch {best_epoch})"
            )
            break

    elapsed = time.perf_counter() - started_at
    minutes, seconds = divmod(elapsed, 60)
    print(
        "Best val mAP50:95: "
        f"{best_val_map:.4f} at epoch {best_epoch}"
    )
    print(f"Stopped early: {stopped_early}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"History: {history_path}")
    print(
        f"Elapsed: {int(minutes)}m {seconds:.1f}s"
    )


if __name__ == "__main__":
    main()
