import math

import torch

def normalize_radar_features(depth, rcs, speed, time_lag):
    depth = depth.clamp(0, 250) / 250
    rcs = (rcs.clamp(-10, 50) + 10) / 60
    speed = speed.clamp(0, 20) / 20
    time_lag = (time_lag.clamp(-0.1, 0.5) + 0.1) / 0.6

    return depth, rcs, speed, time_lag


def rasterize_radar( radar_points, image_size=(640, 360)):
    """
    Convert sparse radar points [N, 7] into dense radar maps [C, H, W].

    multiple points in same pixed handle by:
    - choosing the point with closet timestamp
    - timestamp is the same choose the nearest point in depth
        Channels:
        0: occupancy
        1: point count
        2: selected depth
        3: selected RCS
        4: selected compensated speed magnitude
        5: selected time lag
    """
    # create radar map with the size of the image
    width, height = image_size
    radar_map = torch.zeros((6, height, width), dtype=torch.float32) 

    if len(radar_points) == 0: # need to be able to handle empty radar points
        return radar_map

    u = radar_points[:, 0].long()
    v = radar_points[:, 1].long()

    depth = radar_points[:, 2]
    rcs = radar_points[:, 3]
    vx = radar_points[:, 4]
    vy = radar_points[:, 5]
    time_lag = radar_points[:, 6]

    speed = torch.sqrt(vx**2 + vy**2)

    # Safety check
    valid = (
        (u >= 0)
        & (u < width)
        & (v >= 0)
        & (v < height)
    )

    u = u[valid]
    v = v[valid]
    depth = depth[valid]
    rcs = rcs[valid]
    speed = speed[valid]
    time_lag = time_lag[valid]

    # Group radar points by image pixel
    pixel_groups = {}

    for i in range(len(u)):
        pixel = (int(v[i]), int(u[i]))

        if pixel not in pixel_groups:
            pixel_groups[pixel] = []

        pixel_groups[pixel].append(i)

    for (y, x), indices in pixel_groups.items(): # one pixel each group
        point_count = len(indices)

        # Choose:
        # 1. smallest absolute time lag
        # 2. nearest depth if time lag is tied
        selected_idx = min(
            indices,
            key=lambda i: (
                abs(float(time_lag[i])),
                float(depth[i]),
            ),
        )

        selected_depth = depth[selected_idx]
        selected_rcs = rcs[selected_idx]
        selected_speed = speed[selected_idx]
        selected_time_lag = time_lag[selected_idx]

        # normalize
        selected_depth, selected_rcs, selected_speed, selected_time_lag = normalize_radar_features(selected_depth, selected_rcs, selected_speed, selected_time_lag)
        
        # write into radar map
        radar_map[0, y, x] = 1.0
        radar_map[1, y, x] = math.log1p(min(point_count, 10)) / math.log1p(10)
        radar_map[2, y, x] = selected_depth
        radar_map[3, y, x] = selected_rcs
        radar_map[4, y, x] = selected_speed
        radar_map[5, y, x] = selected_time_lag

    return radar_map
