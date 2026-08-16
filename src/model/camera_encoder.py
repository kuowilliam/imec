import torch
from torch import nn

import timm


class CameraEncoder(nn.Module):
    """
    Frozen DINOv3 ConvNeXt-Tiny camera feature extractor.
    Input:
        images: [B, 3, H, W]
    Output:
        Multi-scale camera feature maps at strides 4, 8, 16, and 32.
    """

    FEATURE_NAMES = ("s4", "s8", "s16", "s32")

    def __init__(self, freeze=True):
        super().__init__()

        self.backbone = timm.create_model(
            "convnext_tiny.dinov3_lvd1689m",
            pretrained=True,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        self.frozen = freeze

        data_cfg = timm.data.resolve_model_data_config(self.backbone)
        self.register_buffer(
            "image_mean",
            torch.tensor(data_cfg["mean"]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor(data_cfg["std"]).view(1, 3, 1, 1),
            persistent=False,
        )

        channels = self.backbone.feature_info.channels()
        # create a dictionary of output channels for each feature map
        self.out_channels = dict(zip(self.FEATURE_NAMES, channels))

        if self.frozen:
            self._freeze_backbone()

    def _freeze_backbone(self):
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.backbone.eval()

    def train(self, mode=True):
        """
        Keep the frozen backbone in evaluation mode even when the full
        detector is switched to training mode.
        """
        super().train(mode)

        if self.frozen:
            self.backbone.eval()

        return self

    def forward(self, images):
        images = (images - self.image_mean) / self.image_std

        if self.frozen:
            with torch.no_grad():
                features = self.backbone(images)
        else:
            features = self.backbone(images)

        return dict(zip(self.FEATURE_NAMES, features))