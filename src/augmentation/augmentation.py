import torch
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as F


# Radar feature layout produced by radar_loader.py:
# [u, v, depth, RCS, vx_comp, vy_comp, time_lag]
RADAR_U_INDEX = 0
RADAR_VY_COMP_INDEX = 5


class SynchronizedAugmentation:
    """
    Photometric jitter only changes appearance.
    Horizontal flip updates:
    - image
    - bounding boxes
    - projected radar u
    - radar lateral velocity (vy_comp)
    """

    def __init__(self, image_size, photometric=None, horizontal_flip=None):
        self.image_width = image_size[0]
        self.image_height = image_size[1]
        self.color_jitter = None
        self.flip_probability = 0.0

        if photometric and photometric.get("enabled"):
            self.color_jitter = ColorJitter(
                brightness=photometric.get("brightness", 0.0),
                contrast=photometric.get("contrast", 0.0),
                saturation=photometric.get("saturation", 0.0),
                hue=photometric.get("hue", 0.0),
            )

        if horizontal_flip and horizontal_flip.get("enabled"):
            self.flip_probability = float(horizontal_flip.get("probability", 0.5))

    def __call__(self, image, boxes, radar_points):
        if self.color_jitter is not None:
            image = self.color_jitter(image)

        if self.flip_probability > 0.0 and torch.rand(1).item() < self.flip_probability:
            image, boxes, radar_points = self._horizontal_flip(
                image,
                boxes,
                radar_points,
            )

        return image, boxes, radar_points

    def _horizontal_flip(self, image, boxes, radar_points):
        image = F.hflip(image)

        if boxes.numel() > 0:
            flipped_boxes = boxes.clone()
            flipped_boxes[:, 0] = self.image_width - boxes[:, 2]
            flipped_boxes[:, 2] = self.image_width - boxes[:, 0]
            boxes = flipped_boxes

        if radar_points.numel() > 0:
            radar_points = radar_points.clone()
            radar_points[:, RADAR_U_INDEX] = (
                self.image_width - 1.0 - radar_points[:, RADAR_U_INDEX]
            )
            radar_points[:, RADAR_VY_COMP_INDEX] = -radar_points[:, RADAR_VY_COMP_INDEX]

        return image, boxes, radar_points


def build_augmentation(config, image_size):
    """
    use switch to enable or disable the augmentation
    """
    if not config:
        return None

    photometric = config.get("photometric") or {}
    horizontal_flip = config.get("horizontal_flip") or {}
    if not photometric.get("enabled") and not horizontal_flip.get("enabled"):
        return None

    return SynchronizedAugmentation(
        image_size=image_size,
        photometric=photometric,
        horizontal_flip=horizontal_flip,
    )
