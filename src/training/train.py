import json
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.augmentation import build_augmentation
from src.config import load_config
from src.data_loader.nuscenes_front_loader import (
    NuScenesFrontLoader,
    collate_fn,
    load_scene_names,
)
from src.evaluation.metrics import PedestrianDetectionMetrics
from src.model.detector import CameraRadarDetector
from src.model.loss import CenterNetLoss
from src.model.postprocess import CenterNetPostProcessor
from src.utils import resolve_path, select_device


def move_targets_to_device(targets, device):
    return [
        {
            name: value.to(device)
            for name, value in target.items()
        }
        for target in targets
    ]


def empty_loss_totals():
    return {
        "total_loss": 0.0,
        "heatmap_loss": 0.0,
        "box_size_loss": 0.0,
        "offset_loss": 0.0,
    }


def average_losses(running_losses, number_of_batches):
    return {
        name: value / number_of_batches
        for name, value in running_losses.items()
    }


def should_validate(epoch, total_epochs, interval):
    if interval < 1:
        raise ValueError("validation_interval must be at least 1.")
    return epoch % interval == 0 or epoch == total_epochs


def build_dataloader(
    config,
    dataroot,
    split,
    batch_size,
    num_workers,
    shuffle,
    device,
    num_samples=None,
    apply_augmentation=False,
    frame_stride=1,
):
    dataset_config = config.get("dataset", {})
    manifest_path = dataset_config.get("scene_manifests", {}).get(split)
    scene_names = (
        load_scene_names(resolve_path(manifest_path))
        if manifest_path is not None
        else None
    )
    augmentation = None
    if apply_augmentation:
        augmentation = build_augmentation(
            config.get("augmentation"),
            image_size=config["image_size"],
        )

    dataset = NuScenesFrontLoader(
        dataroot=dataroot,
        split=split,
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
        augmentation=augmentation,
        scene_names=scene_names,
        frame_stride=frame_stride,
    )

    if num_samples is not None:
        dataset = Subset(dataset, range(min(num_samples, len(dataset))))

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    return dataset, dataloader


def train_one_epoch(model, criterion, dataloader, optimizer, device):
    model.train()

    running_losses = empty_loss_totals()

    for batch in dataloader:
        images = batch["images"].to(device)
        radar_points = batch["radar_points"].to(device)
        radar_padding_mask = batch["radar_padding_mask"].to(device)
        targets = move_targets_to_device(batch["targets"], device)

        optimizer.zero_grad(set_to_none=True)

        predictions = model(
            images=images,
            radar_points=radar_points,
            radar_padding_mask=radar_padding_mask,
        )
        losses = criterion(predictions, targets)
        total_loss = losses["total_loss"]

        # prevent NaN or inf loss, stop training
        if not torch.isfinite(total_loss):
            raise FloatingPointError(
                f"Non-finite training loss: {float(total_loss.detach())}"
            )

        total_loss.backward()
        optimizer.step()

        for name in running_losses:
            running_losses[name] += float(losses[name].detach())

    return average_losses(running_losses, len(dataloader))


@torch.no_grad()
def validate_one_epoch(model, criterion, postprocessor, metrics, dataloader, device):
    model.eval()
    metrics.reset()

    running_losses = empty_loss_totals()

    for batch in dataloader:
        images = batch["images"].to(device)
        radar_points = batch["radar_points"].to(device)
        radar_padding_mask = batch["radar_padding_mask"].to(device)
        targets = move_targets_to_device(batch["targets"], device)

        predictions = model(
            images=images,
            radar_points=radar_points,
            radar_padding_mask=radar_padding_mask,
        )
        losses = criterion(predictions, targets)

        if not torch.isfinite(losses["total_loss"]):
            raise FloatingPointError(
                f"Non-finite validation loss: {float(losses['total_loss'].detach())}"
            )

        for name in running_losses:
            running_losses[name] += float(losses[name].detach())

        detections = postprocessor(predictions)
        metrics.update(detections, targets)

    detection_metrics = metrics.compute()
    return average_losses(running_losses, len(dataloader)), detection_metrics


def save_checkpoint(path, model, optimizer, epoch, losses, metrics, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "losses": losses,
            "metrics": metrics,
            "config": config,
        },
        path,
    )


def save_history(path, history):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(history, file, indent=2)


def main():
    random.seed(42)
    torch.manual_seed(42)

    config = load_config()
    train_cfg = config["train"]
    evaluation_cfg = config["evaluation"]

    device = select_device(train_cfg["device"])
    dataroot = resolve_path(train_cfg["dataroot"])
    checkpoint_path = resolve_path(train_cfg["checkpoint"])
    history_path = resolve_path(train_cfg["history"])
    patience = train_cfg.get("early_stopping_patience")
    validation_interval = train_cfg.get("validation_interval", 1)

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
        freeze_camera=True,
    ).to(device)
    criterion = CenterNetLoss(image_size=config["image_size"]).to(device)
    postprocessor = CenterNetPostProcessor(
        image_size=config["image_size"],
        score_threshold=evaluation_cfg["score_threshold"],
        top_k=evaluation_cfg["top_k"],
    )
    metrics = PedestrianDetectionMetrics(
        iou_thresholds=evaluation_cfg["iou_thresholds"],
        report_iou_threshold=evaluation_cfg["report_iou_threshold"],
        report_score_threshold=evaluation_cfg["report_score_threshold"],
    )

    trainable_parameters = [ # filter out parameters that are not trainable
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
    print(f"Train scenes: {len(train_source.scene_names)}")
    print(f"Val scenes: {len(val_dataset.scene_names)}")
    print(f"Train augmentation enabled: {train_source.augmentation is not None}")
    print(f"Train batches per epoch: {len(train_dataloader)}")
    print(f"Val batches per epoch: {len(val_dataloader)}")
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
                f"Epoch {epoch:03d}/{train_cfg['epochs']:03d} | "
                f"train_total={train_losses['total_loss']:.4f} | "
                f"val_total={val_losses['total_loss']:.4f} | "
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
                history["best_val_map_50_95"] = best_val_map
                save_checkpoint(
                    path=checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    losses={"train": train_losses, "val": val_losses},
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
                f"Epoch {epoch:03d}/{train_cfg['epochs']:03d} | "
                f"train_total={train_losses['total_loss']:.4f} | "
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
                f"val mAP50:95 did not improve for {patience} "
                "validation checks "
                f"(best {best_val_map:.4f} at epoch {best_epoch})"
            )
            break

    elapsed = time.perf_counter() - started_at
    minutes, seconds = divmod(elapsed, 60)
    print(f"Best val mAP50:95: {best_val_map:.4f} at epoch {best_epoch}")
    print(f"Stopped early: {stopped_early}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"History: {history_path}")
    print(f"Elapsed: {int(minutes)}m {seconds:.1f}s")


if __name__ == "__main__":
    main()
