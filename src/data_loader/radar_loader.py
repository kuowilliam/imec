"""
Radar helpers in the CAM_FRONT frame.

load_projected_radar:
aggregate nsweeps, project to the image, return [N, 7]
(u, v, depth, RCS, vx_comp, vy_comp, time_lag).

build_radar_relevance_targets:
label each 3D point as pedestrian (1), clutter (0),
or ignore if it sits in the 10% box margin.
"""

import numpy as np
from nuscenes.eval.detection.utils import category_to_detection_name
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.geometry_utils import points_in_box

def load_projected_radar(
    nusc,
    sample,
    radar_channels=("RADAR_FRONT",), # allow adding more radar channels
    camera_channel="CAM_FRONT",
    nsweeps=5, # default 5 sweeps
    min_distance=1.0,
    class_name="pedestrian",
    ignore_margin=0.10,
):
    """
    Load radar points and project them to the image.

    return relavance by using bbox from the annotation.
    """
    camera_token = sample["data"][camera_channel]
    camera_record = nusc.get("sample_data", camera_token) #get camera record

    # get the pedestrian bboxes from the camera frame.
    _, camera_boxes, _ = nusc.get_sample_data(camera_token)
    pedestrian_boxes = [box for box in camera_boxes if category_to_detection_name(box.name) == class_name]

    # get camera intrinsics for projection
    calibration = nusc.get( "calibrated_sensor", camera_record["calibrated_sensor_token"],)
    camera_intrinsic = np.asarray(calibration["camera_intrinsic"], dtype=np.float32)
    
    all_points = []
    all_camera_xyz = [] # save all coordinates of the 3D camera points.
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
        all_camera_xyz.append(
            points[:3, valid].T.astype(np.float32)
        )

    if not all_points:
        return (
            np.empty((0, 7), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=bool),
        )

    radar_features = np.concatenate(
        all_points,
        axis=0,
    ).astype(np.float32)

    camera_xyz = np.concatenate(
        all_camera_xyz,
        axis=0,
    ).astype(np.float32)

    relevance_targets, relevance_ignore_mask = (
        build_radar_relevance_targets(
            camera_xyz=camera_xyz,
            pedestrian_boxes=pedestrian_boxes,
            ignore_margin=ignore_margin,
        )
    )

    return (
        radar_features,
        relevance_targets,
        relevance_ignore_mask,
    )

def build_radar_relevance_targets(
    camera_xyz,
    pedestrian_boxes,
    ignore_margin=0.10,
):
    """
    determine if this radar point is in the pedestrian box
    - in the box: 1.0
    - out of the box: 0.0
    - ignore: True (if the point is at blurry region)

    input: 3D camera points(camera front), pedestrian boxes
    """
    camera_xyz = np.asarray(camera_xyz, dtype=np.float32)

    num_points = camera_xyz.shape[0]

    targets = np.zeros(num_points, dtype=np.float32)
    ignore_mask = np.zeros(num_points, dtype=bool)

    if num_points == 0 or not pedestrian_boxes:
        return targets, ignore_mask

    # points_in_box expects [3, N]
    points = camera_xyz.T

    positive = np.zeros(num_points, dtype=bool)
    expanded = np.zeros(num_points, dtype=bool)

    for box in pedestrian_boxes:
        positive |= points_in_box(
            box,
            points,
            wlh_factor=1.0,
        )

        expanded |= points_in_box(
            box,
            points,
            wlh_factor=1.0 + ignore_margin,
        )

    targets[positive] = 1.0

    # points in the expanded box but not in the original box
    ignore_mask = expanded & ~positive

    return targets, ignore_mask