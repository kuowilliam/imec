import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import load_config
from src.data_loader.nuscenes_front_loader import NuScenesFrontLoader, collate_fn
from src.model.detector import CameraRadarDetector
from src.model.loss import CenterNetLoss
from src.utils import resolve_path, select_device


def move_targets_to_device(targets, device):
    return [
        {
            name: value.to(device)
            for name, value in target.items()
        }
        for target in targets
    ]


def train_one_epoch(model, criterion, dataloader, optimizer, device):
    model.train()

    running_losses = {
        "total_loss": 0.0,
        "heatmap_loss": 0.0,
        "box_size_loss": 0.0,
        "offset_loss": 0.0,
    }

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

    number_of_batches = len(dataloader)
    return {
        name: value / number_of_batches
        for name, value in running_losses.items()
    }


def save_checkpoint(path, model, optimizer, epoch, losses, config):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "losses": losses,
            "config": config,
        },
        path,
    )


def main():
    random.seed(42)
    torch.manual_seed(42)

    config = load_config()
    train_cfg = config["train"]

    device = select_device(train_cfg["device"])
    dataroot = resolve_path(train_cfg["dataroot"])
    checkpoint_path = resolve_path(train_cfg["checkpoint"])

    dataset = NuScenesFrontLoader(
        dataroot=dataroot,
        split="mini_train",
        image_size=config["image_size"],
        radar_channels=tuple(config["radar"]["channels"]),
        nsweeps=config["radar"]["nsweeps"],
        camera_channel=config["camera_channel"],
        class_name=config["class_name"],
    )

    number_of_samples = train_cfg["num_samples"]

    overfit_dataset = Subset(dataset, range(number_of_samples))
    dataloader = DataLoader(
        overfit_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )

    model = CameraRadarDetector(
        image_size=config["image_size"],
        freeze_camera=True,
    ).to(device)
    criterion = CenterNetLoss(image_size=config["image_size"]).to(device)

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

    print(f"Device: {device}")
    print(f"Overfit samples: {number_of_samples}")
    print(f"Batches per epoch: {len(dataloader)}")
    print(
        "Trainable parameters: "
        f"{sum(parameter.numel() for parameter in trainable_parameters):,}"
    )

    best_loss = float("inf")

    for epoch in range(1, train_cfg["epochs"] + 1):
        losses = train_one_epoch(
            model=model,
            criterion=criterion,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
        )

        print(
            f"Epoch {epoch:03d}/{train_cfg['epochs']:03d} | "
            f"total={losses['total_loss']:.4f} | "
            f"heatmap={losses['heatmap_loss']:.4f} | "
            f"size={losses['box_size_loss']:.4f} | "
            f"offset={losses['offset_loss']:.4f}"
        )

        if losses["total_loss"] < best_loss:
            best_loss = losses["total_loss"]
            save_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                losses=losses,
                config=config,
            )

    print(f"Best total loss: {best_loss:.4f}")
    print(f"Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
