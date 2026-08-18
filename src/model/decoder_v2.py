import torch.nn.functional as F
from torch import nn


class ConvBlock(nn.Module):
    """
    helper class for convolution with group norm and gelu
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()

        padding = kernel_size // 2 

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )

    def forward(self, features):
        return self.block(features)


class PredictionHead(nn.Module):
    """
    final prediction head using small convolutional head for one CenterNet prediction target.
    """

    def __init__(self, in_channels, out_channels, hidden_channels=64):
        super().__init__()

        self.layers = nn.Sequential(
            ConvBlock(in_channels, hidden_channels),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )

    def forward(self, features):
        return self.layers(features)


class PedestrianDetectionDecoder(nn.Module):
    """
    decode multi-scale fused features into CenterNet-style predictions.
    - using top-down FPN pathway from fused s16 to fused s8 and s4.
    - using center-net style prediction head.
    """
    def __init__(
        self,
        fused_s16_channels=256,
        fused_s8_channels=192,
        fused_s4_channels=96,
        decoder_channels=128,
    ):
        super().__init__()

        self.s16_projection = nn.Conv2d( # project fused s16 to decoder 128 channels
            fused_s16_channels,
            decoder_channels,
            kernel_size=1,
        )
        self.s8_lateral = nn.Conv2d( # also convert fused s8 to decoder 128 channels
            fused_s8_channels,
            decoder_channels,
            kernel_size=1,
        )
        self.s8_refine = ConvBlock(
            decoder_channels,
            decoder_channels,
        )

        # Continue the top-down pathway from s8 to s4.
        self.s4_lateral = nn.Conv2d(
            fused_s4_channels,
            decoder_channels,
            kernel_size=1,
        )
        self.s4_refine = ConvBlock(
            decoder_channels,
            decoder_channels,
        )

        self.heatmap_head = PredictionHead(
            decoder_channels,
            out_channels=1,
        )
        self.box_size_head = PredictionHead(
            decoder_channels,
            out_channels=2,
        )
        self.offset_head = PredictionHead(
            decoder_channels,
            out_channels=2,
        )

        # Start with a low foreground probability. This prevents the large
        # number of background cells from dominating early training.
        nn.init.constant_(self.heatmap_head.layers[-1].bias, -2.19)

    def forward(self, fused_s16, fused_s8, fused_s4):

        decoded_s16 = self.s16_projection(fused_s16) # fused s16 to 128 channels

        # upsample s16 to s8
        decoded_s8 = F.interpolate(
            decoded_s16,
            size=fused_s8.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        decoded_s8 = self.s8_refine(
            decoded_s8
            + self.s8_lateral(fused_s8)
        )

        decoded_s4 = F.interpolate(
            decoded_s8,
            size=fused_s4.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        decoded_s4 = self.s4_refine(
            decoded_s4
            + self.s4_lateral(fused_s4)
        )

        # final prediction heads
        return {
            "heatmap_logits": self.heatmap_head(decoded_s4),
            "box_size": self.box_size_head(decoded_s4),
            "offset": self.offset_head(decoded_s4),
        }
