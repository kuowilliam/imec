# Radar Relevance Data Path Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce aligned per-point pedestrian relevance targets and carry them through the existing nuScenes loader and collate function without changing the detector yet.

**Architecture:** Radar returns are already transformed to the current `CAM_FRONT` coordinate frame by `RadarPointCloud.from_file_multisweep`. A pure helper labels each valid return against camera-frame pedestrian 3D boxes, with a fixed 10% enlarged ignore boundary. The loader keeps the existing seven model features and carries targets and ignore masks as separate tensors.

**Tech Stack:** Python 3.12, NumPy, PyTorch, nuScenes devkit

**Spec:** `docs/superpowers/specs/2026-08-17-radar-fusion-v2-design.md`

## Global Constraints

- Keep radar model features ordered as `[u, v, depth, RCS, vx_comp, vy_comp, time_lag]`.
- Use `1` for relevant, `0` for irrelevant, and a separate boolean ignore mask.
- Enlarge each pedestrian box by exactly 10% about its centre for the ignore boundary.
- Keep `True = padding` for `radar_padding_mask`.
- Do not modify the model, loss, training loop, configuration, or dependencies in this sub-project.
- Do not add a permanent test suite or Ruff configuration; use focused executable assertions.

---

### Task 1: Add the pure radar relevance labelling helper

**Files:**
- Modify: `src/data_loader/radar_loader.py`

**Interfaces:**
- Consumes: `camera_xyz: np.ndarray[N, 3]` and nuScenes camera-frame `Box` objects.
- Produces: `build_radar_relevance_targets(camera_xyz, pedestrian_boxes, ignore_margin=0.10) -> tuple[np.ndarray, np.ndarray]`, where the arrays are relevance targets and ignore flags with shape `[N]`.

- [ ] **Step 1: Run a focused assertion before the helper exists**

```bash
uv run python - <<'PY'
from src.data_loader.radar_loader import build_radar_relevance_targets
print(build_radar_relevance_targets)
PY
```

Expected: import failure because `build_radar_relevance_targets` does not exist.

- [ ] **Step 2: Add the geometry import and helper**

Change the geometry import and add this function above `load_projected_radar`:

```python
from nuscenes.utils.geometry_utils import points_in_box, view_points


def build_radar_relevance_targets(
    camera_xyz,
    pedestrian_boxes,
    ignore_margin=0.10,
):
    camera_xyz = np.asarray(camera_xyz, dtype=np.float32)
    if camera_xyz.ndim != 2 or camera_xyz.shape[1] != 3:
        raise ValueError("camera_xyz must have shape [N, 3].")
    if ignore_margin < 0.0:
        raise ValueError("ignore_margin must be non-negative.")

    num_points = camera_xyz.shape[0]
    targets = np.zeros(num_points, dtype=np.float32)
    ignore_mask = np.zeros(num_points, dtype=bool)
    if num_points == 0 or not pedestrian_boxes:
        return targets, ignore_mask

    points = camera_xyz.T
    positive = np.zeros(num_points, dtype=bool)
    expanded = np.zeros(num_points, dtype=bool)

    for box in pedestrian_boxes:
        positive |= points_in_box(box, points, wlh_factor=1.0)
        expanded |= points_in_box(
            box,
            points,
            wlh_factor=1.0 + ignore_margin,
        )

    targets[positive] = 1.0
    ignore_mask = expanded & ~positive
    return targets, ignore_mask
```

- [ ] **Step 3: Verify positive, ignore-boundary, negative, and empty cases**

```bash
uv run python - <<'PY'
import numpy as np
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
from src.data_loader.radar_loader import build_radar_relevance_targets

box = Box(
    center=[0.0, 0.0, 10.0],
    size=[2.0, 2.0, 2.0],
    orientation=Quaternion(),
)
points = np.asarray([
    [0.0, 0.0, 10.0],
    [1.04, 0.0, 10.0],
    [3.0, 0.0, 10.0],
], dtype=np.float32)

targets, ignored = build_radar_relevance_targets(points, [box])
assert targets.tolist() == [1.0, 0.0, 0.0]
assert ignored.tolist() == [False, True, False]

empty_targets, empty_ignored = build_radar_relevance_targets(
    np.empty((0, 3), dtype=np.float32),
    [box],
)
assert empty_targets.shape == (0,)
assert empty_ignored.shape == (0,)
print("radar relevance helper: PASS")
PY
```

Expected: `radar relevance helper: PASS`.

- [ ] **Step 4: Review and commit only the helper**

```bash
git diff --check
git add src/data_loader/radar_loader.py
git commit -m "Add radar relevance target helper"
```

---

### Task 2: Generate relevance labels with projected radar features

**Files:**
- Modify: `src/data_loader/radar_loader.py`
- Modify: `src/data_loader/nuscenes_front_loader.py`

**Interfaces:**
- Consumes: current sample, current camera channel, radar channels, and class name.
- Produces: `load_projected_radar(...) -> tuple[features, targets, ignore_mask]`, with aligned shapes `[N, 7]`, `[N]`, and `[N]`.

- [ ] **Step 1: In `load_projected_radar`, get camera-frame pedestrian boxes once**

Add `class_name="pedestrian"` and `ignore_margin=0.10` parameters. Import `category_to_detection_name`, then immediately after loading `camera_record` use:

```python
_, camera_boxes, _ = nusc.get_sample_data(camera_token)
pedestrian_boxes = [
    box
    for box in camera_boxes
    if category_to_detection_name(box.name) == class_name
]
```

- [ ] **Step 2: Accumulate aligned camera-frame XYZ coordinates**

Initialize `all_camera_xyz = []` next to `all_points = []`. After applying `valid`, append:

```python
all_camera_xyz.append(points[:3, valid].T.astype(np.float32))
```

For the empty case, return three aligned empty arrays:

```python
return (
    np.empty((0, 7), dtype=np.float32),
    np.empty((0,), dtype=np.float32),
    np.empty((0,), dtype=bool),
)
```

For the non-empty case, replace the existing return with:

```python
features = np.concatenate(all_points, axis=0).astype(np.float32)
camera_xyz = np.concatenate(all_camera_xyz, axis=0).astype(np.float32)
targets, ignore_mask = build_radar_relevance_targets(
    camera_xyz,
    pedestrian_boxes,
    ignore_margin=ignore_margin,
)
return features, targets, ignore_mask
```

- [ ] **Step 3: Unpack the new return values in `NuScenesFrontLoader.__getitem__`**

```python
radar_points, radar_relevance_targets, radar_relevance_ignore_mask = (
    load_projected_radar(
        nusc=self.nusc,
        sample=sample,
        radar_channels=self.radar_channels,
        camera_channel=self.camera_channel,
        nsweeps=self.nsweeps,
        class_name=self.class_name,
    )
)
```

Convert both arrays to tensors in `_to_tensor`, and return sample keys:

```python
"radar_relevance_targets": radar_relevance_targets,
"radar_relevance_ignore_mask": radar_relevance_ignore_mask,
```

The augmentation call does not need these arrays because horizontal flipping changes radar `(u, v)` but does not change whether the physical return lies inside its original 3D box.

- [ ] **Step 4: Verify one real sample remains aligned**

Use the existing loader construction in `src/data_loader/nuscenes_front_loader.py`, then assert:

```python
sample = dataset[0]
num_points = sample["radar_points"].shape[0]
assert sample["radar_points"].shape[1] == 7
assert sample["radar_relevance_targets"].shape == (num_points,)
assert sample["radar_relevance_ignore_mask"].shape == (num_points,)
assert set(sample["radar_relevance_targets"].unique().tolist()) <= {0.0, 1.0}
print("single-sample radar relevance: PASS")
```

- [ ] **Step 5: Commit the aligned sample interface**

```bash
git diff --check
git add src/data_loader/radar_loader.py src/data_loader/nuscenes_front_loader.py
git commit -m "Generate pedestrian relevance labels for radar points"
```

---

### Task 3: Pad relevance labels in `collate_fn`

**Files:**
- Modify: `src/data_loader/nuscenes_front_loader.py`

**Interfaces:**
- Consumes: variable-length sample tensors from Task 2.
- Produces: batch tensors `radar_relevance_targets: [B, N]` and `radar_relevance_ignore_mask: [B, N]` aligned with `radar_points` and `radar_padding_mask`.

- [ ] **Step 1: Allocate batch tensors beside `radar_padding_mask`**

```python
radar_relevance_targets = torch.zeros(
    batch_size,
    max_points,
    dtype=torch.float32,
)
radar_relevance_ignore_mask = torch.ones(
    batch_size,
    max_points,
    dtype=torch.bool,
)
```

Padding starts ignored so it can never contribute to the later relevance loss.

- [ ] **Step 2: Copy valid targets and ignore masks in the existing loop**

```python
radar_relevance_targets[i, :n] = batch[i]["radar_relevance_targets"]
radar_relevance_ignore_mask[i, :n] = batch[i][
    "radar_relevance_ignore_mask"
]
```

- [ ] **Step 3: Add both tensors to the returned batch dictionary**

```python
"radar_relevance_targets": radar_relevance_targets,
"radar_relevance_ignore_mask": radar_relevance_ignore_mask,
```

- [ ] **Step 4: Verify batch alignment and padding invariants**

After obtaining one batch from the existing `DataLoader`, run:

```python
assert batch["radar_points"].shape[:2] == batch["radar_padding_mask"].shape
assert batch["radar_relevance_targets"].shape == batch["radar_padding_mask"].shape
assert batch["radar_relevance_ignore_mask"].shape == batch["radar_padding_mask"].shape
assert torch.all(
    batch["radar_relevance_ignore_mask"][batch["radar_padding_mask"]]
)
print("collated radar relevance: PASS")
```

- [ ] **Step 5: Commit the complete relevance data path**

```bash
git diff --check
git add src/data_loader/nuscenes_front_loader.py
git commit -m "Collate radar relevance supervision"
```

At this checkpoint the detector remains unchanged. The next sub-project begins only after inspecting real positive/ignored/negative counts and projected overlays from this data path.
