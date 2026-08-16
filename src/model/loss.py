import math

import torch
import torch.nn.functional as F
from torch import nn


class CenterNetLoss(nn.Module):
    """
    Convert ground-truth bounding boxes into CenterNet-style training targets
    and compute the pedestrian detection losses.

    The ground-truth boxes are converted into:
    - a center heatmap,
    - bounding-box width and height targets,
    - sub-cell center offset targets
    """
    def __init__(
        self,
        image_size=(640, 360),
        min_gaussian_overlap=0.7,
        heatmap_alpha=2.0,
        heatmap_beta=4.0,
        box_size_weight=0.1,
        offset_weight=1.0,
    ):
        super().__init__()

        self.image_width = image_size[0]
        self.image_height = image_size[1]
        self.min_gaussian_overlap = min_gaussian_overlap
        self.heatmap_alpha = heatmap_alpha
        self.heatmap_beta = heatmap_beta
        self.box_size_weight = box_size_weight
        self.offset_weight = offset_weight

    @staticmethod
    def _gaussian_radius(height, width, min_overlap):
        """
        based on the width and height of the bounding box, calculate the radius of the gaussian
        with the constraint of the minimum overlap.
        """
        a1 = 1.0
        b1 = height + width
        c1 = width * height * (1.0 - min_overlap) / (1.0 + min_overlap)
        radius1 = (b1 + math.sqrt(max(0.0, b1**2 - 4.0 * a1 * c1))) / 2.0

        a2 = 4.0
        b2 = 2.0 * (height + width)
        c2 = (1.0 - min_overlap) * width * height
        radius2 = (b2 + math.sqrt(max(0.0, b2**2 - 4.0 * a2 * c2))) / 2.0

        a3 = 4.0 * min_overlap
        b3 = -2.0 * min_overlap * (height + width)
        c3 = (min_overlap - 1.0) * width * height
        radius3 = (b3 + math.sqrt(max(0.0, b3**2 - 4.0 * a3 * c3))) / 2.0

        return min(radius1, radius2, radius3)

    @staticmethod
    def _draw_gaussian(heatmap, center_x, center_y, radius):
        """Draw one 2D Gaussian on a single-class heatmap in place."""
        diameter = 2 * radius + 1
        sigma = diameter / 6.0

        coordinates = torch.arange(
            -radius,
            radius + 1,
            device=heatmap.device,
            dtype=heatmap.dtype,
        )
        grid_y, grid_x = torch.meshgrid(
            coordinates,
            coordinates,
            indexing="ij",
        )
        gaussian = torch.exp(-(grid_x.square() + grid_y.square()) / (2.0 * sigma**2))

        height, width = heatmap.shape
        left = min(center_x, radius)
        right = min(width - center_x - 1, radius)
        top = min(center_y, radius)
        bottom = min(height - center_y - 1, radius)

        heatmap_patch = heatmap[
            center_y - top:center_y + bottom + 1,
            center_x - left:center_x + right + 1,
        ]
        gaussian_patch = gaussian[
            radius - top:radius + bottom + 1,
            radius - left:radius + right + 1,
        ]

        torch.maximum(
            heatmap_patch,
            gaussian_patch,
            out=heatmap_patch,
        )

    def build_targets(self, targets, output_size, device, dtype):
        """
        - GT bboxes -> get center, width, height, offset
        - project to output resolution
        convert into:
            - heatmap: center of gaussian
            - box_size: width and height
            - offset: sub-cell center offset
            - regression_mask: which grid cell contains the object
        """
        output_height, output_width = output_size
        batch_size = len(targets)

        # create empty heatmap and box_size, offset tensors
        heatmap = torch.zeros(
            batch_size,
            1,
            output_height,
            output_width,
            device=device,
            dtype=dtype,
        )
        box_size = torch.zeros(
            batch_size,
            2,
            output_height,
            output_width,
            device=device,
            dtype=dtype,
        )
        offset = torch.zeros_like(box_size)
        regression_mask = torch.zeros(
            batch_size,
            output_height,
            output_width,
            device=device,
            dtype=torch.bool,
        )

        # calculate the scale
        scale_x = output_width / self.image_width
        scale_y = output_height / self.image_height

        for batch_index, target in enumerate(targets):
            boxes = target["boxes"].to(device=device, dtype=dtype)

            for box in boxes: # for each bbox in the image
                if not torch.isfinite(box).all():
                    continue

                x1, y1, x2, y2 = box
                x1 = x1.clamp(0.0, float(self.image_width))
                x2 = x2.clamp(0.0, float(self.image_width))
                y1 = y1.clamp(0.0, float(self.image_height))
                y2 = y2.clamp(0.0, float(self.image_height))

                box_width = x2 - x1
                box_height = y2 - y1

                if box_width <= 0 or box_height <= 0:
                    continue

                center_x = ((x1 + x2) / 2.0) * scale_x
                center_y = ((y1 + y2) / 2.0) * scale_y

                center_x = center_x.clamp(0.0, output_width - 1e-4)
                center_y = center_y.clamp(0.0, output_height - 1e-4)

                center_int_x = int(torch.floor(center_x).item())
                center_int_y = int(torch.floor(center_y).item())

                # for calculate the radius of gaussian
                output_box_width = float((box_width * scale_x).item())
                output_box_height = float((box_height * scale_y).item())
                radius = max(
                    0,
                    int(
                        self._gaussian_radius(
                            output_box_height,
                            output_box_width,
                            self.min_gaussian_overlap,
                        )
                    ),
                )

                self._draw_gaussian(
                    heatmap[batch_index, 0],
                    center_int_x,
                    center_int_y,
                    radius,
                )
                box_size[batch_index, :, center_int_y, center_int_x] = torch.stack([box_width, box_height])
                offset[batch_index, :, center_int_y, center_int_x] = torch.stack([center_x - center_int_x, center_y - center_int_y])
                regression_mask[batch_index, center_int_y, center_int_x] = True

        return {
            "heatmap": heatmap,
            "box_size": box_size,
            "offset": offset,
            "regression_mask": regression_mask,
        }

    def _heatmap_loss(self, logits, target):
        """CenterNet penalty-reduced focal loss, calculated from logits."""
        probabilities = torch.sigmoid(logits)
        positive_mask = target.eq(1.0).to(logits.dtype)
        negative_mask = target.lt(1.0).to(logits.dtype)
        # close to the center of the gaussian does not need to be penalized as much
        negative_weights = (1.0 - target).pow(self.heatmap_beta) 

        positive_loss = (
            F.logsigmoid(logits)
            * (1.0 - probabilities).pow(self.heatmap_alpha)
            * positive_mask
        )
        negative_loss = (
            F.logsigmoid(-logits)
            * probabilities.pow(self.heatmap_alpha)
            * negative_weights
            * negative_mask
        )

        number_of_objects = positive_mask.sum()

        if number_of_objects.item() == 0:
            return -negative_loss.sum()

        return -(positive_loss.sum() + negative_loss.sum()) / number_of_objects

    @staticmethod
    def _regression_l1_loss(prediction, target, regression_mask):
        """Calculate L1 loss only at ground-truth object centers."""
        number_of_objects = regression_mask.sum()

        if number_of_objects.item() == 0: # if there are no objects, return 0
            return prediction.sum() * 0.0

        expanded_mask = regression_mask.unsqueeze(1).expand_as(prediction)

        return F.l1_loss( # only calculate the loss at the object centers, other cells are ignored
            prediction[expanded_mask],
            target[expanded_mask],
            reduction="sum",
        ) / number_of_objects

    def forward(self, predictions, targets):
        heatmap_logits = predictions["heatmap_logits"]
        box_size_prediction = predictions["box_size"]
        offset_prediction = predictions["offset"]

        output_size = heatmap_logits.shape[-2:]
        dense_targets = self.build_targets(
            targets=targets,
            output_size=output_size,
            device=heatmap_logits.device,
            dtype=heatmap_logits.dtype,
        )

        heatmap_loss = self._heatmap_loss(
            heatmap_logits,
            dense_targets["heatmap"],
        )
        box_size_loss = self._regression_l1_loss(
            box_size_prediction,
            dense_targets["box_size"],
            dense_targets["regression_mask"],
        )
        offset_loss = self._regression_l1_loss(
            offset_prediction,
            dense_targets["offset"],
            dense_targets["regression_mask"],
        )

        total_loss = (
            heatmap_loss
            + self.box_size_weight * box_size_loss
            + self.offset_weight * offset_loss
        )

        return {
            "total_loss": total_loss,
            "heatmap_loss": heatmap_loss,
            "box_size_loss": box_size_loss,
            "offset_loss": offset_loss,
        }
