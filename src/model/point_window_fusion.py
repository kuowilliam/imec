import math

import torch
import torch.nn.functional as F
from torch import nn


class PointWindowCrossAttention(nn.Module):
    """Fuse sparse radar tokens into occupied local camera windows only."""

    def __init__(
        self,
        camera_channels,
        radar_token_dim,
        fusion_dim,
        num_heads,
        window_size=5,
        vertical_neighbor_windows=1, # number of windows to include above and below the current window
        window_batch_bucket_size=32,
        ffn_dim=None,
        dropout=0.1,
    ):
        super().__init__()

        if fusion_dim % num_heads != 0:
            raise ValueError(
                "fusion_dim must be divisible by num_heads."
            )

        self.fusion_dim = fusion_dim
        self.window_size = window_size
        self.vertical_neighbor_windows = vertical_neighbor_windows
        self.window_batch_bucket_size = window_batch_bucket_size

        self.camera_projection = (
            nn.Identity()
            if camera_channels == fusion_dim
            else nn.Conv2d(
                camera_channels,
                fusion_dim,
                kernel_size=1,
            )
        )

        self.radar_projection = nn.Linear(
            radar_token_dim,
            fusion_dim,
        )

        self.camera_position_mlp = nn.Sequential(
            nn.Linear(2, 64),
            nn.GELU(),
            nn.Linear(64, fusion_dim),
        )

        self.camera_query_norm = nn.LayerNorm(fusion_dim)
        self.radar_norm = nn.LayerNorm(fusion_dim)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.attention_dropout = nn.Dropout(dropout)

        hidden_dim = ffn_dim or 2 * fusion_dim

        self.ffn_norm = nn.LayerNorm(fusion_dim)
        self.ffn = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, fusion_dim),
            nn.Dropout(dropout),
        )

        self.output_norm = nn.LayerNorm(fusion_dim)

    @staticmethod
    def _round_up_to_multiple(value, multiple):
        return (
            (value + multiple - 1)
            // multiple
            * multiple
        )

    @staticmethod
    def _next_power_of_two(value):
        return 1 << (value - 1).bit_length()

    @staticmethod
    def _make_camera_positions(
        padded_height,
        padded_width,
        original_height,
        original_width,
        device,
        dtype,
    ):
        """
        Create a normalized 2D position (x, y) for each camera feature cell.
        """
        y = (
            (
                torch.arange(
                    padded_height,
                    device=device,
                    dtype=dtype,
                )
                + 0.5
            )
            / original_height
            * 2.0
            - 1.0
        )

        x = (
            (
                torch.arange(
                    padded_width,
                    device=device,
                    dtype=dtype,
                )
                + 0.5
            )
            / original_width
            * 2.0
            - 1.0
        )

        grid_y, grid_x = torch.meshgrid(
            y,
            x,
            indexing="ij",
        )

        return torch.stack(
            [grid_x, grid_y],
            dim=-1,
        )

    def _partition_windows(self, features):
        """Partition a feature map into non-overlapping local windows."""
        batch_size, channels, height, width = features.shape
        window_size = self.window_size

        window_rows = math.ceil(height / window_size)
        window_columns = math.ceil(width / window_size)

        padded_height = window_rows * window_size
        padded_width = window_columns * window_size

        padded = F.pad(
            features,
            (
                0,
                padded_width - width,
                0,
                padded_height - height,
            ),
        )

        windows = (
            padded
            .permute(0, 2, 3, 1)
            .reshape(
                batch_size,
                window_rows,
                window_size,
                window_columns,
                window_size,
                channels,
            )
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(
                batch_size
                * window_rows
                * window_columns,
                window_size * window_size,
                channels,
            )
        )

        return (
            windows,
            window_rows,
            window_columns,
            padded_height,
            padded_width,
        )

    def _merge_windows(
        self,
        windows,
        batch_size,
        window_rows,
        window_columns,
        height,
        width,
    ):
        """Merge local windows and crop padding back to the input size."""
        window_size = self.window_size

        features = (
            windows
            .reshape(
                batch_size,
                window_rows,
                window_columns,
                window_size,
                window_size,
                self.fusion_dim,
            )
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(
                batch_size,
                window_rows * window_size,
                window_columns * window_size,
                self.fusion_dim,
            )
            .permute(0, 3, 1, 2)
        )

        return features[:, :, :height, :width]

    def forward(
        self,
        camera_feature, # for specific resolution feature map, ex s16
        radar_tokens,
        radar_positions,
        radar_padding_mask,
        relevance_logits,
        return_attention=False,
    ):
        batch_size, _, height, width = camera_feature.shape

        camera_base = self.camera_projection(camera_feature)
        # partition the camera feature map into windows
        camera_windows, window_rows, window_columns, padded_height, padded_width = self._partition_windows(camera_base)

        # get the coordinates of the entire padded feature map
        camera_positions = self._make_camera_positions(
            padded_height=padded_height,
            padded_width=padded_width,
            original_height=height,
            original_width=width,
            device=camera_base.device,
            dtype=camera_base.dtype,
        )

        position_map = (
            camera_positions
            .permute(2, 0, 1)
            .unsqueeze(0)
            .expand(batch_size, -1, -1, -1)
        )

        position_windows, _, _, _, _ = (
            self._partition_windows(position_map)
        )
        position_windows = position_windows[:, :, :2]

        # project the radar tokens to the fusion dimension
        radar_tokens = self.radar_projection(radar_tokens)

        occupied_indices = []
        local_token_lists = []
        local_relevance_lists = []
        local_point_counts = []

        valid_point_count = int(
            (~radar_padding_mask).sum().item()
        )

        global_attention_pairs = (
            height * width * valid_point_count
        )

        # process one image at a time
        for batch_index in range(batch_size):
            valid = ~radar_padding_mask[batch_index] # valid radar points

            if not bool(valid.any()):
                continue # leave if no valid radar points

            sample_tokens = radar_tokens[batch_index, valid]
            sample_positions = radar_positions[batch_index, valid]
            sample_relevance = relevance_logits[batch_index, valid]

            x_cells = torch.floor(
                (sample_positions[:, 0] + 1.0)
                * 0.5
                * width
            ).long().clamp(0, width - 1)

            y_cells = torch.floor(
                (sample_positions[:, 1] + 1.0)
                * 0.5
                * height
            ).long().clamp(0, height - 1)

            # get the position of the point in the window
            point_window_x = torch.div(x_cells, self.window_size, rounding_mode="floor")
            point_window_y = torch.div(y_cells, self.window_size, rounding_mode="floor")

            window_to_points = {} # map from window index to point indices

            for point_index, (base_x, base_y) in enumerate(
                zip(
                    point_window_x.tolist(),
                    point_window_y.tolist(),
                )
            ):
                # get the range of windows to include
                first_y = max(0, base_y - self.vertical_neighbor_windows)
                last_y = min(window_rows - 1, base_y + self.vertical_neighbor_windows)

                for destination_y in range(# add above and below the current window
                    first_y,
                    last_y + 1,
                ):
                    local_window_index = (
                        destination_y * window_columns
                        + base_x
                    )

                    window_to_points.setdefault(
                        local_window_index,
                        [],
                    ).append(point_index)

            for local_window_index, point_indices in sorted(
                window_to_points.items()
            ):
                flat_index = (
                    batch_index
                    * window_rows
                    * window_columns
                    + local_window_index
                )

                point_index_tensor = torch.tensor(
                    point_indices,
                    device=camera_base.device,
                    dtype=torch.long,
                )

                occupied_indices.append(flat_index)

                local_token_lists.append(
                    sample_tokens[point_index_tensor]
                )

                local_relevance_lists.append(
                    sample_relevance[point_index_tensor]
                )

                local_point_counts.append(
                    len(point_indices)
                )

        diagnostics = {
            "occupied_windows": len(occupied_indices),
            "local_radar_points": sum(local_point_counts),
            "local_attention_pairs": (
                self.window_size
                * self.window_size
                * sum(local_point_counts)
            ),
            "global_attention_pairs": global_attention_pairs,
            "attention_window_capacity": 0,
            "attention_point_capacity": 0,
        }

        if not occupied_indices:
            return {
                "feature_map": camera_base,
                "diagnostics": diagnostics,
                "attention_records": (
                    [] if return_attention else None
                ),
            }

        # get the camera window that contains the radar points
        occupied_index_tensor = torch.tensor(
            occupied_indices,
            device=camera_base.device,
            dtype=torch.long,
        )
        occupied_camera = camera_windows[occupied_index_tensor]
        occupied_positions = position_windows[occupied_index_tensor]

        number_of_occupied = len(occupied_indices)

        window_capacity = self._round_up_to_multiple(
            number_of_occupied,
            self.window_batch_bucket_size,
        )

        point_capacity = self._next_power_of_two(
            max(local_point_counts)
        )

        diagnostics["attention_window_capacity"] = (
            window_capacity
        )

        diagnostics["attention_point_capacity"] = (
            point_capacity
        )

        attention_camera = camera_windows.new_zeros(
            window_capacity,
            occupied_camera.shape[1],
            self.fusion_dim,
        )

        attention_positions = position_windows.new_zeros(
            window_capacity,
            occupied_positions.shape[1],
            occupied_positions.shape[2],
        )

        attention_camera[:number_of_occupied] = occupied_camera # copy the occupied camera windows to the attention camera
        attention_positions[:number_of_occupied] = occupied_positions # copy the occupied positions to the attention positions

        # build the camera queries
        camera_queries = self.camera_query_norm(
            attention_camera
            + self.camera_position_mlp(
                attention_positions
            )
        )

        radar_keys = radar_tokens.new_zeros(
            window_capacity,
            point_capacity,
            self.fusion_dim,
        )
        relevance = relevance_logits.new_zeros(
            window_capacity,
            point_capacity,
        )

        local_padding_mask = torch.ones(
            window_capacity,
            point_capacity,
            dtype=torch.bool,
            device=camera_base.device,
        )

        for index, (tokens, logits) in enumerate(
            zip(
                local_token_lists,
                local_relevance_lists,
            )
        ):
            count = tokens.shape[0]

            radar_keys[index, :count] = tokens
            relevance[index, :count] = logits
            local_padding_mask[index, :count] = False

        if window_capacity > number_of_occupied:
            local_padding_mask[
                number_of_occupied:,
                0,
            ] = False

        radar_keys = self.radar_norm(radar_keys)
        # time the relevance to get the radar values
        radar_values = radar_keys * torch.sigmoid(relevance).unsqueeze(-1)

        # build the radar keys and values, perform the cross-attention
        radar_update, attention_weights = (
            self.cross_attention(
                query=camera_queries,
                key=radar_keys,
                value=radar_values,
                key_padding_mask=local_padding_mask,
                need_weights=return_attention,
                average_attn_weights=False,
            )
        )
        # camera_attention + radar_update + ffn
        fused = attention_camera + self.attention_dropout(radar_update)
        fused = fused + self.ffn(self.ffn_norm(fused))

        fused = self.output_norm(fused)

        fused = fused[:number_of_occupied]

        fused_windows = camera_windows.clone()

        fused_windows[
            occupied_index_tensor
        ] = fused

        # merge the windows back into the feature map
        fused_feature_map = self._merge_windows(
            windows=fused_windows,
            batch_size=batch_size,
            window_rows=window_rows,
            window_columns=window_columns,
            height=height,
            width=width,
        )

        attention_records = None

        if return_attention:
            attention_records = []

            for index, (
                flat_index,
                point_count,
            ) in enumerate(
                zip(
                    occupied_indices,
                    local_point_counts,
                )
            ):
                attention_records.append(
                    {
                        "flat_window_index": flat_index,
                        "local_point_count": point_count,
                        "weights": attention_weights[
                            index,
                            :,
                            :,
                            :point_count,
                        ],
                    }
                )

        return {
            "feature_map": fused_feature_map,
            "diagnostics": diagnostics,
            "attention_records": attention_records,
        }


class MultiScaleCameraRadarFusion(nn.Module):
    """Apply point-window fusion at s4, s8, and s16."""

    SCALE_SETTINGS = {
        "s4": {
            "fusion_dim": 96,
            "num_heads": 4,
            "ffn_dim": 192,
        },
        "s8": {
            "fusion_dim": 192,
            "num_heads": 8,
            "ffn_dim": 384,
        },
        "s16": {
            "fusion_dim": 256,
            "num_heads": 8,
            "ffn_dim": 512,
        },
    }

    def __init__(
        self,
        camera_channels,
        radar_token_dim=256,
        window_size=5,
        vertical_neighbor_windows=1,
        window_batch_bucket_size=32,
        dropout=0.1,
    ):
        super().__init__()

        self.blocks = nn.ModuleDict()

        for scale, settings in self.SCALE_SETTINGS.items():
            self.blocks[scale] = PointWindowCrossAttention(
                camera_channels=camera_channels[scale],
                radar_token_dim=radar_token_dim,
                fusion_dim=settings["fusion_dim"],
                num_heads=settings["num_heads"],
                window_size=window_size,
                vertical_neighbor_windows=(
                    vertical_neighbor_windows
                ),
                window_batch_bucket_size=(
                    window_batch_bucket_size
                ),
                ffn_dim=settings["ffn_dim"],
                dropout=dropout,
            )

        self.out_channels = {
            scale: settings["fusion_dim"]
            for scale, settings
            in self.SCALE_SETTINGS.items()
        }

    def forward(
        self,
        camera_features,
        radar_tokens,
        radar_positions,
        radar_padding_mask,
        relevance_logits,
        return_attention=False,
    ):
        fused_features = {}
        diagnostics = {}
        attention_records = {}

        for scale, block in self.blocks.items():
            output = block(
                camera_feature=camera_features[scale],
                radar_tokens=radar_tokens,
                radar_positions=radar_positions,
                radar_padding_mask=radar_padding_mask,
                relevance_logits=relevance_logits,
                return_attention=return_attention,
            )

            fused_features[scale] = output[
                "feature_map"
            ]

            diagnostics[scale] = output[
                "diagnostics"
            ]

            if return_attention:
                attention_records[scale] = output[
                    "attention_records"
                ]

        return {
            "features": fused_features,
            "diagnostics": diagnostics,
            "attention_records": (
                attention_records
                if return_attention
                else None
            ),
        }