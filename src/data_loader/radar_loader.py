"""
Since my goal is 2D front-camera detection, I choose CAM_FRONT as the reference frame.

Processing the radar data from the nuScenes dataset.
Steps:
- For a given sample
↓
- Select the RADAR_FRONT data closest to this timestamp
↓
- Aggregate the nearest 5 radar sweeps
↓
- Transform radar points from their original coordinate frame to the CAM_FRONT 3D coordinate frame
↓
- Use camera intrinsics to project the 3D radar points onto image pixels (u, v)
↓
- Keep only points that actually fall within the CAM_FRONT image frame
↓
- Obtain an array with shape [N, 7]
"""

import numpy as np
from pathlib import Path
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.geometry_utils import view_points

def load_projected_radar(
    nusc,
    sample,
    radar_channels=("RADAR_FRONT",), # allow adding more radar channels
    camera_channel="CAM_FRONT",
    nsweeps=5, # default 5 sweeps
    min_distance=1.0,
):
    camera_token = sample["data"][camera_channel]
    camera_record = nusc.get("sample_data", camera_token) #get camera record

    # get camera intrinsics for projection
    calibration = nusc.get( "calibrated_sensor", camera_record["calibrated_sensor_token"],)
    camera_intrinsic = np.asarray(calibration["camera_intrinsic"], dtype=np.float32)
    
    all_points = []
    for radar_channel in radar_channels:
        radar, time_lags = RadarPointCloud.from_file_multisweep(
            nusc=nusc,
            sample_rec=sample,
            chan=radar_channel,
            ref_chan=camera_channel,
            nsweeps=nsweeps,
            min_distance=min_distance,
        )

        points = radar.points
        # do projection from 3D to 2D
        pixels = view_points(
            points[:3],
            camera_intrinsic,
            normalize=True,
        )
        depth = points[2]

        valid = ( # filter out points that are outside the image
            (depth > min_distance)
            & (pixels[0] >= 0)
            & (pixels[0] < camera_record["width"])
            & (pixels[1] >= 0)
            & (pixels[1] < camera_record["height"])
        )

        if not np.any(valid):
            continue

        features = np.stack(
            [
                pixels[0, valid],
                pixels[1, valid],
                depth[valid],
                points[5, valid],   # RCS, radar reflection strength
                points[8, valid],   # compensated vx
                points[9, valid],   # compensated vy
                time_lags[0, valid],
            ],
            axis=1,
        )

        all_points.append(features)

    if not all_points:
        return np.empty((0, 7), dtype=np.float32)

    return np.concatenate(all_points, axis=0).astype(np.float32)