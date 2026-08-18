import json

import torch
from torch.utils.data import DataLoader, Subset

from src.augmentation import build_augmentation
from src.data_loader.nuscenes_front_loader import (
    NuScenesFrontLoader,
    collate_fn,
    load_scene_names,
)
from src.utils import resolve_path


def move_targets_to_device(targets, device):
    return [
        {
            name: value.to(device)
            for name, value in target.items()
        }
        for target in targets
    ]


def average_losses(running_losses, number_of_batches):
    return {
        name: float(value / number_of_batches)
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
