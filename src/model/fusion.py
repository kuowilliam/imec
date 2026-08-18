import torch
from torch import nn


class CameraRadarFusion(nn.Module):
    """
    Fuse camera feature map with Radar tokens.

    Camera features are used as queries.
    Radar tokens are used as keys and values.
    """

    def __init__(
        self,
        camera_channels=384,
        token_dim=256,
        num_heads=8,
        ffn_dim=512,
        dropout=0.1,
    ):
        super().__init__()

        self.token_dim = token_dim

        # Convert the s16 camera feature channels from 384 to 256.
        self.camera_projection = nn.Conv2d(
            camera_channels,
            token_dim,
            kernel_size=1,
        )

        # camera position encoding
        # Encode normalized 2D camera feature-map positions.
        self.camera_position_mlp = nn.Sequential(
            nn.Linear(2, 64),
            nn.GELU(),
            nn.Linear(64, token_dim),
        )

        # normalize layer
        self.camera_query_norm = nn.LayerNorm(token_dim)
        self.radar_norm = nn.LayerNorm(token_dim)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.attention_dropout = nn.Dropout(dropout)

        self.ffn_norm = nn.LayerNorm(token_dim)

        self.ffn = nn.Sequential(
            nn.Linear(token_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, token_dim),
            nn.Dropout(dropout),
        )

        self.output_norm = nn.LayerNorm(token_dim)

    def _make_camera_positions(self, height, width, device, dtype):
        """
        Create a normalized 2D position (x, y) for each cell in the camera feature map.
        """
        y = ((torch.arange(height, device=device, dtype=dtype) + 0.5) / height * 2.0 - 1.0)
        x = ((torch.arange(width, device=device, dtype=dtype) + 0.5) / width * 2.0 - 1.0)

        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        positions = torch.stack([grid_x, grid_y], dim=-1)

        return positions.reshape(1, height * width, 2)

    def forward(
        self,
        camera_feature,
        radar_tokens,
        radar_padding_mask,
        return_attention=False,
    ):
        batch_size, _, height, width = camera_feature.shape

        # transform camera feature map to token dimension
        camera_feature = self.camera_projection(camera_feature)
        camera_tokens = camera_feature.flatten(2).transpose(1, 2) # flatten to tokens

        camera_positions = self._make_camera_positions( # get camera positions
            height=height,
            width=width,
            device=camera_tokens.device,
            dtype=camera_tokens.dtype,
        )

        # encode camera positions to tokens
        camera_position_tokens = self.camera_position_mlp(camera_positions)

        # merge camera tokens and camera position tokens
        camera_queries = self.camera_query_norm(camera_tokens + camera_position_tokens)

        radar_keys_values = self.radar_norm(radar_tokens)


        # cross-attention
        radar_update, attention_weights = self.cross_attention(
            query=camera_queries,
            key=radar_keys_values,
            value=radar_keys_values,
            key_padding_mask=radar_padding_mask,
            need_weights=return_attention,
            average_attn_weights=False,
        )

        # residual connection
        fused_tokens = (
            camera_tokens
            + self.attention_dropout(radar_update)
        )

        # Per-token nonlinear refinement.
        fused_tokens = (
            fused_tokens
            + self.ffn(self.ffn_norm(fused_tokens))
        )

        fused_tokens = self.output_norm(fused_tokens)
        fused_feature_map = (
            fused_tokens
            .transpose(1, 2)
            .reshape(batch_size, self.token_dim, height, width)
        )

        return {
            "feature_map": fused_feature_map,
            "attention_weights": attention_weights,
        }
