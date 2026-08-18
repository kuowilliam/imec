from torch import nn

from src.model.camera_encoder import CameraEncoder
from src.model.decoder_v2 import PedestrianDetectionDecoder
from src.model.point_window_fusion import (
    MultiScaleCameraRadarFusion,
)
from src.model.radar_encoder_v2 import RadarPointEncoder


class CameraRadarDetector(nn.Module):
    """
    V2 camera-radar pedestrian detector.

    Camera:
        image -> camera encoder -> s4, s8, s16

    Radar:
        points -> point encoder
        -> tokens, positions, relevance logits

    Fusion:
        independent local point-window attention
        at s4, s8, and s16

    Decoder:
        fused s4, s8, s16
        -> heatmap, box size, offset
    """

    def __init__(
        self,
        image_size=(640, 360),
        token_dim=256,
        decoder_channels=128,
        window_size=5,
        vertical_neighbor_windows=1,
        window_batch_bucket_size=32,
        dropout=0.1,
        freeze_camera=True,
    ):
        super().__init__()

        self.camera_encoder = CameraEncoder(
            freeze=freeze_camera,
        )

        self.radar_encoder = RadarPointEncoder(
            image_size=image_size,
            token_dim=token_dim,
        )

        camera_channels = self.camera_encoder.out_channels

        self.fusion = MultiScaleCameraRadarFusion(
            camera_channels=camera_channels,
            radar_token_dim=token_dim,
            window_size=window_size,
            vertical_neighbor_windows=(
                vertical_neighbor_windows
            ),
            window_batch_bucket_size=(
                window_batch_bucket_size
            ),
            dropout=dropout,
        )

        self.decoder = PedestrianDetectionDecoder(
            fused_s16_channels=(
                self.fusion.out_channels["s16"]
            ),
            fused_s8_channels=(
                self.fusion.out_channels["s8"]
            ),
            fused_s4_channels=(
                self.fusion.out_channels["s4"]
            ),
            decoder_channels=decoder_channels,
        )

    def forward(
        self,
        images,
        radar_points,
        radar_padding_mask,
        return_diagnostics=False,
        return_attention=False,
    ):
        camera_features = self.camera_encoder(images)

        radar_output = self.radar_encoder(
            radar_points,
            radar_padding_mask,
        )

        fusion_output = self.fusion(
            camera_features=camera_features,
            radar_tokens=radar_output["tokens"],
            radar_positions=radar_output["positions"],
            radar_padding_mask=radar_output["padding_mask"],
            relevance_logits=radar_output[
                "relevance_logits"
            ],
            return_attention=return_attention,
        )

        predictions = self.decoder(
            fused_s16=fusion_output["features"]["s16"],
            fused_s8=fusion_output["features"]["s8"],
            fused_s4=fusion_output["features"]["s4"],
        )

        predictions["radar_relevance_logits"] = (
            radar_output["relevance_logits"]
        )

        if return_diagnostics or return_attention:
            predictions["fusion_diagnostics"] = (
                fusion_output["diagnostics"]
            )

        if return_attention:
            predictions["attention_records"] = (
                fusion_output["attention_records"]
            )

        return predictions
