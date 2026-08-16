from torch import nn

from src.model.camera_encoder import CameraEncoder
from src.model.decoder import PedestrianDetectionDecoder
from src.model.fusion import CameraRadarFusion
from src.model.radar_encoder import RadarPointEncoder


class CameraRadarDetector(nn.Module):
    """
    full pipeline of a camera and radar pedestrian detector.

    camera head: input image -> camera encoder -> camera features s4, s8, s16
    radar head: input radar points -> radar encoder -> radar tokens
    
    fusion: camera features s16 and radar tokens -> fused features
    decoder: fused features -> heatmap_logits, box_size, offset
    """

    def __init__(
        self,
        image_size=(640, 360),
        token_dim=256,
        num_heads=8,
        ffn_dim=512,
        decoder_channels=128,
        dropout=0.1,
        freeze_camera=True,
    ):
        super().__init__()

        self.camera_encoder = CameraEncoder(freeze=freeze_camera)

        self.radar_encoder = RadarPointEncoder(
            image_size=image_size,
            token_dim=token_dim,
        )

        camera_channels = self.camera_encoder.out_channels

        self.fusion = CameraRadarFusion(
            camera_channels=camera_channels["s16"],
            token_dim=token_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )

        self.decoder = PedestrianDetectionDecoder(
            fused_channels=token_dim,
            camera_s8_channels=camera_channels["s8"],
            camera_s4_channels=camera_channels["s4"],
            decoder_channels=decoder_channels,
        )

    def forward(
        self,
        images,
        radar_points,
        radar_padding_mask,
        return_attention=False,
    ):
        camera_features = self.camera_encoder(images)

        radar_output = self.radar_encoder(
            radar_points,
            radar_padding_mask,
        )

        fusion_output = self.fusion(
            camera_feature=camera_features["s16"],
            radar_tokens=radar_output["tokens"],
            radar_padding_mask=radar_output["padding_mask"],
            return_attention=return_attention,
        )

        predictions = self.decoder(
            fused_s16=fusion_output["feature_map"],
            camera_s8=camera_features["s8"],
            camera_s4=camera_features["s4"],
        )

        if return_attention:
            predictions["attention_weights"] = fusion_output[
                "attention_weights"
            ]

        return predictions
