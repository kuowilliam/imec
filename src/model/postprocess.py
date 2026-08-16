import torch
import torch.nn.functional as F
from torch import nn


class CenterNetPostProcessor(nn.Module):
    """
    decode the output of model: heatmap, box_size, offset
    back to its original output image-space bbox

    Input dictionary: heatmap_logits, box_size, offset
    Output: list of detection dictionaries, each containing boxes, scores, and labels
    """
    def __init__(
        self,
        image_size=(640, 360),
        score_threshold=0.1,
        top_k=100,
        local_maximum_kernel=3,
    ):
        super().__init__()

        self.image_width = image_size[0]
        self.image_height = image_size[1]
        self.score_threshold = score_threshold # only keep the confidence scores above this threshold
        self.top_k = top_k # keep the top k detections
        self.local_maximum_kernel = local_maximum_kernel # kernel size for local maximum pooling

    @staticmethod
    def _gather_at_indices(feature_map, indices):
        """
        Find the top K cells in the heatmap and get the indices
        use this indices to fetch values from offset and box_size
        Form bounding boxes
        """
        batch_size, channels, _, _ = feature_map.shape
        flattened = feature_map.flatten(2)
        gather_indices = indices.unsqueeze(1).expand(
            batch_size,
            channels,
            indices.shape[1],
        )
        return flattened.gather(2, gather_indices).transpose(1, 2)

    @torch.no_grad()
    def forward(self, predictions):
        heatmap_logits = predictions["heatmap_logits"]
        box_size_map = predictions["box_size"]
        offset_map = predictions["offset"]

        probabilities = torch.sigmoid(heatmap_logits)

        padding = self.local_maximum_kernel // 2
        local_maxima = F.max_pool2d( # find the local maxima in the heatmap
            probabilities,
            kernel_size=self.local_maximum_kernel,
            stride=1,
            padding=padding,
        )
        peak_heatmap = probabilities * probabilities.eq(local_maxima) # peak suppression

        batch_size, _, output_height, output_width = peak_heatmap.shape
        number_of_candidates = min(self.top_k, output_height * output_width)

        scores, indices = torch.topk( # get the top k scores and indices
            peak_heatmap.flatten(1),
            k=number_of_candidates,
            dim=1,
        )

        # convert the flattened index back to (x, y) coordinates.
        center_y_indices = torch.div(indices, output_width, rounding_mode="floor")
        center_x_indices = indices % output_width

        offsets = self._gather_at_indices(offset_map, indices) # get the offset values
        box_sizes = self._gather_at_indices(box_size_map, indices) # get the box size values

        stride_x = self.image_width / output_width
        stride_y = self.image_height / output_height

        center_x = (
            center_x_indices.to(offsets.dtype) + offsets[..., 0]
        ) * stride_x
        center_y = (
            center_y_indices.to(offsets.dtype) + offsets[..., 1]
        ) * stride_y

        box_width = box_sizes[..., 0]
        box_height = box_sizes[..., 1]

        # recover the bbox using center and box size
        x1 = (center_x - box_width / 2.0).clamp(0.0, self.image_width)
        y1 = (center_y - box_height / 2.0).clamp(0.0, self.image_height)
        x2 = (center_x + box_width / 2.0).clamp(0.0, self.image_width)
        y2 = (center_y + box_height / 2.0).clamp(0.0, self.image_height)

        all_boxes = torch.stack([x1, y1, x2, y2], dim=-1)
        detections = []

        for batch_index in range(batch_size):
            keep = (
                (scores[batch_index] >= self.score_threshold)
                & (box_width[batch_index] > 0.0)
                & (box_height[batch_index] > 0.0)
            )

            kept_scores = scores[batch_index][keep]
            kept_boxes = all_boxes[batch_index][keep]
            kept_labels = torch.ones(
                kept_scores.shape[0],
                dtype=torch.int64,
                device=kept_scores.device,
            )

            detections.append(
                {
                    "boxes": kept_boxes,
                    "scores": kept_scores,
                    "labels": kept_labels,
                }
            )

        return detections
