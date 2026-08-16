from pathlib import Path
import sys

import numpy as np
import torch
from torchvision.transforms import functional as F

from PIL import Image
from torch.utils.data import Dataset

from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.geometry_utils import view_points

# Allow running this file directly: uv run nuscenes_front_loader.py
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import load_config
from src.data_loader.radar_loader import load_projected_radar

class NuScenesFrontLoader(Dataset):
    def __init__(
        self,
        dataroot,
        split="mini_train",
        image_size=(640, 360),
        radar_channels=("RADAR_FRONT",),
        nsweeps=5,
        camera_channel="CAM_FRONT",
        class_name="pedestrian",
        version="v1.0-mini",
    ):
        self.dataroot = Path(dataroot)
        self.split = split
        self.image_size = image_size
        self.radar_channels = radar_channels
        self.nsweeps = nsweeps
        self.camera_channel = camera_channel
        self.class_name = class_name

        self.nusc = NuScenes(
            version=version,
            dataroot=str(self.dataroot),
            verbose=False,
        )

        # split scenes
        split_scenes = set(create_splits_scenes()[split])
        self.scene_name_by_token = { # map scene token to scene name
            scene["token"]: scene["name"]
            for scene in self.nusc.scene
        }

        self.samples = [ # filter samples by split
            sample
            for sample in self.nusc.sample
            if self.scene_name_by_token[sample["scene_token"]] in split_scenes
        ]

    def _get_pedestrian_boxes(self, sample, original_size):
        """
        Get pedestrian boxes only and project them to the image.
        """
        camera_token = sample["data"][self.camera_channel]
        _, boxes, camera_intrinsic = self.nusc.get_sample_data(camera_token)

        original_w, original_h = original_size
        target_w, target_h = self.image_size

        # calculate scale for resizing
        scale_x = target_w / original_w
        scale_y = target_h / original_h

        pedestrian_boxes = []

        for box in boxes:
            if category_to_detection_name(box.name) != self.class_name:
                continue

            corners = box.corners()
            projected = view_points( # Project the 3D box corners to the image
                corners,
                np.asarray(camera_intrinsic),
                normalize=True,
            )

            x1 = projected[0].min()
            y1 = projected[1].min()
            x2 = projected[0].max()
            y2 = projected[1].max()

            # Clip to original image
            x1 = np.clip(x1, 0, original_w)
            y1 = np.clip(y1, 0, original_h)
            x2 = np.clip(x2, 0, original_w)
            y2 = np.clip(y2, 0, original_h)

            if x2 <= x1 or y2 <= y1:
                continue

            pedestrian_boxes.append([
                x1 * scale_x,
                y1 * scale_y,
                x2 * scale_x,
                y2 * scale_y,
            ])

        return np.asarray(pedestrian_boxes, dtype=np.float32).reshape(-1, 4)

    def __len__(self):
        return len(self.samples)

    def _to_tensor(self, image, boxes, radar_points):
        image = F.to_tensor(image)

        boxes = torch.as_tensor(
            boxes,
            dtype=torch.float32,
        )

        labels = torch.ones(
            len(boxes),
            dtype=torch.int64,
        )

        radar_points = torch.as_tensor(
            radar_points,
            dtype=torch.float32,
        )

        target = {
            "boxes": boxes,
            "labels": labels,
        }

        return image, radar_points, target

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Camera
        camera_token = sample["data"][self.camera_channel]
        camera_record = self.nusc.get("sample_data", camera_token)

        image_path = self.dataroot / camera_record["filename"]
        image = Image.open(image_path).convert("RGB")

        original_w, original_h = image.size
        target_w, target_h = self.image_size

        scale_x = target_w / original_w
        scale_y = target_h / original_h

        # Pedestrian boxes
        boxes = self._get_pedestrian_boxes(
            sample,
            (original_w, original_h),
        )

        # Radar
        radar_points = load_projected_radar(
            nusc=self.nusc,
            sample=sample,
            radar_channels=self.radar_channels,
            camera_channel=self.camera_channel,
            nsweeps=self.nsweeps,
        )

        # Scale radar image coordinates to resized image
        if len(radar_points) > 0:
            radar_points[:, 0] *= scale_x  # u
            radar_points[:, 1] *= scale_y  # v

        # Resize image
        image = image.resize(self.image_size)

        image, radar_points, target = self._to_tensor(image, boxes, radar_points)
        return {
            "image": image,
            "radar_points": radar_points,
            "target": target,
            "metadata": {
                "sample_token": sample["token"],
                "scene_name": self.scene_name_by_token[sample["scene_token"]],
            },
        }


def collate_fn(batch):
    # image shape is the same so can stack
    images = torch.stack([item["image"] for item in batch])

    # radar points shape is different so need to pad
    radar_list = [item["radar_points"] for item in batch]
    max_points = max(1, max(points.shape[0] for points in radar_list)) # get the max number of points

    batch_size = len(batch)
    feature_dim = radar_list[0].shape[-1]

    # create a tensor to store the radar points
    radar_points = torch.zeros(
        batch_size,
        max_points,
        feature_dim,
        dtype=torch.float32,
    )

    radar_padding_mask = torch.ones(
        batch_size,
        max_points,
        dtype=torch.bool,
    )

    for i, points in enumerate(radar_list):
        n = points.shape[0]

        if n == 0:
            continue

        radar_points[i, :n] = points
        radar_padding_mask[i, :n] = False

    return {
        "images": images,
        "radar_points": radar_points,
        "radar_padding_mask": radar_padding_mask,
        "targets": [item["target"] for item in batch],
        "metadata": [item["metadata"] for item in batch],
    }

if __name__ == "__main__":
    cfg = load_config()
    dataroot = Path(__file__).resolve().parents[2] / "v1.0-mini"

    common = dict(
        dataroot=dataroot,
        image_size=cfg["image_size"],
        camera_channel=cfg["camera_channel"],
        class_name=cfg["class_name"],
        radar_channels=tuple(cfg["radar"]["channels"]),
        nsweeps=cfg["radar"]["nsweeps"],
    )

    train_dataset = NuScenesFrontLoader(split="mini_train", **common)
    val_dataset = NuScenesFrontLoader(split="mini_val", **common)

    sample = train_dataset[0]
    print("Image:", sample["image"].shape)
    print("Radar points:", sample["radar_points"].shape)
    print("Boxes:", sample["target"]["boxes"].shape)
    print("Labels:", sample["target"]["labels"].shape)

    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,
    )
    batch = next(iter(train_loader))
    print("Batch size:", len(batch["images"]))

    for i in range(len(batch["images"])):
        valid_points = (~batch["radar_padding_mask"][i]).sum().item()

        print(
            i,
            "image:", batch["images"][i].shape,
            "radar:", batch["radar_points"][i].shape,
            "valid radar points:", valid_points,
            "boxes:", batch["targets"][i]["boxes"].shape,
        )
