import torch
from torch import nn


class RadarPointEncoder(nn.Module):
    """
    Encode projected Radar points into tokens for cross-attention.

    Input:
        radar_points: [B, N, 7]
        radar_padding_mask: [B, N]
            False = valid Radar point
            True = padding

    Output:
        tokens: [B, N + 1, token_dim]
        padding_mask: [B, N + 1]

    The additional token (+1) is a learned null Radar token.
    """

    def __init__(self, image_size=(640, 360), token_dim=256,):
        super().__init__()

        self.image_width = image_size[0]
        self.image_height = image_size[1]
        self.token_dim = token_dim

        # physical radar features:
        # depth, RCS, vx, vy, time lag
        self.feature_mlp = nn.Sequential(
            nn.Linear(5, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, token_dim),
        )

        # Image-plane position:
        # normalized u, normalized v
        self.position_mlp = nn.Sequential(
            nn.Linear(2, 64),
            nn.GELU(),
            nn.Linear(64, token_dim),
        )

        self.output_norm = nn.LayerNorm(token_dim)

        # This represents no useful radar information.
        self.null_token = nn.Parameter(torch.zeros(1, 1, token_dim))

    def _normalize_points(self, radar_points):
        u = radar_points[..., 0]
        v = radar_points[..., 1]
        depth = radar_points[..., 2]
        rcs = radar_points[..., 3]
        vx = radar_points[..., 4]
        vy = radar_points[..., 5]
        time_lag = radar_points[..., 6]

        # normalize image positions to -1, 1
        u = 2.0 * u / max(self.image_width - 1, 1) - 1.0
        v = 2.0 * v / max(self.image_height - 1, 1) - 1.0

        # same physical ranges used during the earlier EDA/raster work.
        depth = depth.clamp(0.0, 250.0) / 250.0
        rcs = (rcs.clamp(-10.0, 50.0) + 10.0) / 60.0
        vx = vx.clamp(-20.0, 20.0) / 20.0
        vy = vy.clamp(-20.0, 20.0) / 20.0
        time_lag = (
            time_lag.clamp(-0.1, 0.5) + 0.1
        ) / 0.6

        positions = torch.stack([u, v], dim=-1)
        physical_features = torch.stack([depth, rcs, vx, vy, time_lag], dim=-1)

        return physical_features, positions

    def forward(self, radar_points, radar_padding_mask):

        physical_features, positions = self._normalize_points(radar_points)

        feature_tokens = self.feature_mlp(physical_features)
        position_tokens = self.position_mlp(positions)

        # combine features and positions
        radar_tokens = self.output_norm(
            feature_tokens + position_tokens
        )

        # Padded input rows can become non-zero due to MLP biases.
        # so extra zero them
        radar_tokens = radar_tokens.masked_fill(
            radar_padding_mask.unsqueeze(-1),
            0.0,
        )

        batch_size = radar_points.shape[0]

        null_tokens = self.null_token.expand(batch_size, -1, -1)

        # Null token is always valid.
        null_mask = torch.zeros(
            batch_size,
            1,
            dtype=torch.bool,
            device=radar_padding_mask.device,
        )

        # combine null tokens and radar tokens, also add to padding mask
        tokens = torch.cat([null_tokens, radar_tokens], dim=1)
        padding_mask = torch.cat([null_mask, radar_padding_mask], dim=1)

        return {
            "tokens": tokens,
            "padding_mask": padding_mask,
        }