import torch
import torch.nn.functional as F

from src.model.loss import (
    CenterNetLoss as CenterNetDetectionLoss,
)


class CenterNetLoss(CenterNetDetectionLoss):
    """
    V2 CenterNet detection loss with balanced point-level
    Radar relevance supervision.
    """

    def __init__(
        self,
        image_size=(640, 360),
        min_gaussian_overlap=0.7,
        heatmap_alpha=2.0,
        heatmap_beta=4.0,
        box_size_weight=0.1,
        offset_weight=1.0,
        radar_relevance_weight=0.1,
    ):
        super().__init__(
            image_size=image_size,
            min_gaussian_overlap=min_gaussian_overlap,
            heatmap_alpha=heatmap_alpha,
            heatmap_beta=heatmap_beta,
            box_size_weight=box_size_weight,
            offset_weight=offset_weight,
        )

        self.radar_relevance_weight = (
            radar_relevance_weight
        )

    @staticmethod
    def _balanced_radar_relevance_loss(
        logits,
        targets,
        ignore_mask,
        padding_mask,
    ):
        targets = targets.to(device=logits.device, dtype=logits.dtype)
        ignore_mask = ignore_mask.to(device=logits.device, dtype=torch.bool)
        padding_mask = padding_mask.to(device=logits.device, dtype=torch.bool)

        valid = ~ignore_mask & ~padding_mask
        # if no valid points, return 0 loss
        if not bool(valid.any()):
            return logits.sum() * 0.0

        point_losses = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )
        # define positive and negative points
        positive = valid & targets.eq(1.0)
        negative = valid & targets.eq(0.0)

        group_losses = []

        # average the loss for positive points
        if bool(positive.any()):
            group_losses.append(
                point_losses[positive].mean()
            )

        # average the loss for negative points
        if bool(negative.any()):
            group_losses.append(
                point_losses[negative].mean()
            )

        if not group_losses:
            return logits.sum() * 0.0

        # final average the losses for positive and negative points
        return torch.stack(group_losses).mean()

    def forward(
        self,
        predictions,
        targets,
        radar_relevance_targets,
        radar_relevance_ignore_mask,
        radar_padding_mask,
    ):
        # use the original detection loss
        detection_losses = super().forward(
            predictions,
            targets,
        )

        radar_relevance_loss = (
            self._balanced_radar_relevance_loss(
                logits=predictions[
                    "radar_relevance_logits"
                ],
                targets=radar_relevance_targets,
                ignore_mask=radar_relevance_ignore_mask,
                padding_mask=radar_padding_mask,
            )
        )

        return {
            **detection_losses,
            "total_loss": (
                detection_losses["total_loss"]
                + self.radar_relevance_weight
                * radar_relevance_loss
            ),
            "radar_relevance_loss": (
                radar_relevance_loss
            ),
        }
