# V2 Relevance-Supervised Multi-Scale Radar Fusion

Date: 2026-08-17
Status: Proposed for review
Baseline commit: `6d04792` (`main`)
Implementation branch: `feature/v2_architecture`

## 1. Purpose

V1 proves that the camera-radar detector trains and generalizes beyond the
nuScenes mini split, but it does not yet prove that radar contributes useful
information. The current architecture gives the detector several easy camera
shortcuts:

1. A learned null radar token is always available to cross-attention.
2. The s16 fusion block has a camera residual, so its radar update can collapse
   toward zero.
3. The decoder receives camera-only s8 and s4 lateral features after the single
   s16 fusion block.
4. The final objective supervises only 2D detection. The radar encoder has no
   direct objective that teaches it which sparse radar returns are relevant to
   pedestrians.

Removing every fallback would force noisy or missing radar into the detector.
V2 instead makes radar use conditional: relevant radar should affect all
decoder resolutions, empty radar regions should preserve the camera path
exactly, and irrelevant returns should be suppressed toward that path.

## 2. Goals

- Preserve the point-based radar representation; do not rasterize radar.
- Replace global camera-to-radar attention with true local-window attention.
- Fuse radar at s4, s8, and s16 so the decoder has no raw camera-only lateral
  shortcut.
- Remove the learned null token as an unrestricted attention shortcut.
- Bypass fusion exactly when a local window has no radar points.
- Directly supervise radar-point relevance using nuScenes annotations.
- Suppress clutter through relevance-weighted radar values.
- Preserve camera residuals so missing or unreliable radar cannot corrupt the
  primary modality.
- Produce measurable evidence of radar use with normal, masked, and shuffled
  radar evaluations.
- Report sparsity and runtime honestly; a boolean mask over dense global
  attention is not sufficient to claim sparse computation.

## 3. Non-goals

- Replacing the DINOv3 ConvNeXt-Tiny camera encoder.
- Unfreezing the camera backbone.
- Switching the CenterNet-style 2D detection head or localization loss.
- Adding LiDAR or additional radar sensors.
- Adding an SSM/Mamba block.
- Reproducing HRFuser, TransCAR, or CRAFT end to end.
- Introducing camera blackout or modality dropout in the first V2 experiment.
- Optimizing an accelerator-specific attention kernel.

The AP50-to-AP75 localization gap remains a separate issue. V2 may improve
localization through better spatial fusion, but it is not presented as a
replacement for later box-regression improvements.

## 4. Research basis and adaptations

### 4.1 TransCAR

Reference: <https://arxiv.org/abs/2305.00397>

TransCAR treats each sparse radar point independently, encodes radar attributes
and radar position with separate MLPs, adds the two embeddings, and uses the
result as a cross-attention token. This directly supports retaining the current
`RadarPointEncoder` design.

TransCAR also prevents object queries from attending to distant radar points
with a Query-Radar spatial mask. V2 adapts this locality principle from BEV 3D
object-query distance to projected image-plane windows.

This is an adaptation, not a reproduction: V2 camera feature cells are dense
2D queries, and locality is defined by projected `(u, v)` rather than 3D query
centres.

### 4.2 HRFuser

Reference: <https://arxiv.org/abs/2206.15157>

HRFuser uses multi-window cross-attention repeatedly at multiple feature
resolutions. Its secondary modalities are projected into dense feature maps and
partitioned into corresponding windows.

V2 adopts the local-window and multi-resolution principles, but radar remains a
sparse token set. Radar tokens are assigned to camera windows from their
projected coordinates instead of being rasterized and passed through a
secondary CNN branch.

The resulting module is therefore described as HRFuser-inspired point-window
cross-attention, not as an HRFuser implementation.

### 4.3 CRAFT

Reference: <https://arxiv.org/abs/2209.06535>

CRAFT uses point-based radar features, proposal-conditioned association, a
zero-attention fallback, and an auxiliary point-wise classification loss that
predicts whether a radar point belongs to a 3D object. Proposals without useful
radar preserve their image prediction.

V2 adopts the direct radar-relevance supervision and conditional fallback
principles. It does not adopt CRAFT's PointNet++, image proposals, polar
association, deformable attention, or 3D detection heads.

### 4.4 CRF-Net

Reference: <https://arxiv.org/abs/2005.07431>

CRF-Net's BlackIn training occasionally removes the camera input to counter
camera dominance. This is reserved as a later experiment because pedestrian
radar support is incomplete. It is not part of the initial V2 implementation.

## 5. Architecture

### 5.1 Overview

```text
Camera image                         Projected radar points
     |                                      |
DINOv3 ConvNeXt-Tiny                Shared point MLP encoder
     |                                      |
 s4, s8, s16                     tokens, positions, relevance logits
     |                           /             |              \
     +--> s4 point-window fusion               |               |
     +----------> s8 point-window fusion ------+               |
     +----------------> s16 point-window fusion ---------------+
                         |
                 fused s4, s8, s16
                         |
                CenterNet/FPN decoder
                         |
              heatmap, size, offset
```

The radar encoder is shared across scales. Each scale has its own camera
projection, radar projection, local cross-attention, FFN, and normalization
parameters because feature dimensions and semantics differ by resolution.

### 5.2 Radar point encoder

The seven loader features remain:

```text
[u, v, camera_depth, RCS, vx_comp, vy_comp, time_lag]
```

The existing encoder continues to create each point token from:

```text
feature_mlp(depth, RCS, vx_comp, vy_comp, time_lag)
+ position_mlp(normalized_u, normalized_v)
```

The learned null token is removed. The encoder returns:

```text
tokens:            [B, N, 256]
positions:         [B, N, 2]   # normalized projected u, v
padding_mask:      [B, N]
relevance_logits:  [B, N]
```

The relevance head is a small shared MLP applied to each valid radar token. It
does not use camera features, so the auxiliary task directly supervises the
radar representation.

### 5.3 Radar relevance target

Training uses nuScenes 3D pedestrian annotations as training-only supervision.
A radar return is positive when its point, transformed into the current
CAM_FRONT coordinate frame, lies inside a visible pedestrian 3D box for the
current sample. It is negative when it lies outside every visible pedestrian
box. Padding points are ignored.

Because accumulated sweeps and radar projection contain spatial uncertainty,
each pedestrian box is also enlarged by 10% about its centre along all three
box axes. Points inside the original box are positive; points outside the
original box but inside the enlarged box are ignored; all other valid points
are negative. The 10% ignore margin is fixed for the first V2 experiment. The
same label is shared by all three fusion scales.

This auxiliary label does not change the final task or inference input. The
deployed model still consumes camera and radar and produces only 2D pedestrian
detections.

### 5.4 Relevance loss

The total objective becomes:

```text
L_total = L_heatmap + 0.1 * L_size + L_offset
          + lambda_relevance * L_relevance
```

`L_relevance` is computed only over non-padding, non-ignored radar points. To
avoid an all-negative solution, positive and negative point losses are averaged
separately and then given equal weight when both groups are present. If a batch
contains only one group, only the available group contributes.

The initial `lambda_relevance` is `0.1`. It is a single documented design
constant for the first V2 run, not a hyperparameter sweep.

### 5.5 Point-window construction

Each camera feature map is partitioned into non-overlapping `5 x 5` feature-cell
windows. Five cells are chosen instead of HRFuser's seven because the current
640 x 360 feature sizes and sparse point count produce less padding and smaller
high-resolution attention groups.

Approximate image-plane coverage is:

```text
s4:  20 x 20 image pixels per base window
s8:  40 x 40 image pixels per base window
s16: 80 x 80 image pixels per base window
```

Radar `(u, v)` coordinates are scaled to the current feature-map resolution and
assigned to their corresponding base window. To account for weak radar height
information, a camera window may also consume radar points from its immediately
adjacent vertical windows. Horizontal expansion is not used initially because
radar azimuth is more reliable than elevation and unconstrained horizontal
expansion would increase cross-object mixing.

Feature-map edges may produce smaller windows. The implementation pads camera
features only for partitioning, tracks the original height and width, and crops
the merged result back to the exact original shape.

### 5.6 True sparse local computation

The implementation does not pass a full-scene boolean mask to global
`nn.MultiheadAttention`. Instead it:

1. Partitions camera tokens into windows.
2. Assigns valid radar tokens to each window neighbourhood.
3. Selects only camera windows with at least one local radar point.
4. Pads radar keys only within the selected window batch.
5. Runs cross-attention on the selected windows.
6. Scatters fused windows back into the feature map.
7. Leaves empty windows exactly equal to that scale's camera base feature. For
   s4 and s8 this is the original camera feature; for s16 it is the existing
   384-to-256 camera projection. Normalization and FFN operations are applied
   only inside occupied windows, not globally after scattering.

This makes the amount of attention work depend on occupied windows and local
radar counts rather than `H * W * N_radar`.

### 5.7 Relevance-weighted fusion

For radar token `r_j`, relevance probability is:

```text
p_j = sigmoid(relevance_logit_j)
```

Cross-attention uses radar tokens as keys and relevance-weighted radar tokens as
values:

```text
Q = camera_window
K = radar_window
V = p * radar_window
radar_update = MultiHeadAttention(Q, K, V)
fused_window = camera_window + dropout(radar_update)
```

The camera residual remains mandatory. There is no independent learned fusion
gate because such a gate could collapse to zero without supervision. The
supervised relevance probability controls the magnitude of radar information.

If a window contains radar points but all are predicted irrelevant, its update
approaches zero and the residual preserves the camera feature. If a window has
no radar points, attention is not called and the camera feature is returned
directly.

### 5.8 Scale-specific dimensions

To limit high-resolution cost, the fusion dimensions follow the existing
camera/decoder widths instead of expanding every scale to 256 channels:

```text
s4:   96 channels, 4 heads
s8:  192 channels, 8 heads
s16: 256 channels, 8 heads
```

The shared 256-dimensional radar token is projected separately to each scale.
The s16 camera map is projected from 384 to 256 channels; s8 and s4 retain their
current widths.

### 5.9 Decoder

The decoder interface changes from:

```text
fused_s16, camera_s8, camera_s4
```

to:

```text
fused_s16, fused_s8, fused_s4
```

The top-down FPN additions remain. This preserves the current CenterNet head and
isolates V2 changes to radar supervision and fusion. All decoder inputs have
passed through a conditional radar fusion block, while camera residuals keep
each representation valid in empty or irrelevant radar regions.

## 6. Missing-radar and failure behaviour

| Condition | Expected behaviour |
| --- | --- |
| Sample has no radar points | All scales equal their camera paths; no attention call |
| A window has no local radar | That window bypasses attention exactly |
| Window contains clutter only | Relevance probabilities suppress radar values |
| Window contains useful return | Local attention injects radar at s4, s8, and s16 |
| Projection is vertically uncertain | Adjacent vertical windows provide tolerance |
| Radar relevance head is uncertain | Camera residual limits the damage |

No architectural design can both guarantee radar use and permit fallback using
only the final detection loss. Direct relevance supervision is therefore a
required V2 component, not an optional diagnostic.

## 7. Interfaces and files

### `src/data_loader/radar_loader.py`

- Retain projected camera-frame point coordinates long enough to build radar
  relevance targets.
- Return the existing seven model features plus relevance labels and ignore
  masks.
- Keep the model feature ordering stable.

### `src/data_loader/nuscenes_front_loader.py`

- Add relevance labels to each sample target.
- Resize and augment projected positions and 2D boxes as before.
- Ensure horizontal flip preserves relevance labels unchanged.

### `src/data_loader/nuscenes_front_loader.py::collate_fn`

- Pad relevance labels and relevance-ignore masks alongside radar points.
- Preserve the existing `True = padding` convention.

### `src/model/radar_encoder.py`

- Remove the learned null token.
- Return normalized point positions.
- Add the per-point relevance head.

### `src/model/fusion.py`

- Replace global `CameraRadarFusion` with a reusable
  `PointWindowCrossAttention` block.
- Add a `MultiScaleCameraRadarFusion` wrapper for s4, s8, and s16.
- Return optional per-scale diagnostics: occupied-window count, local pair
  count, and attention weights for selected windows.

### `src/model/detector.py`

- Apply shared radar encoding once.
- Fuse s4, s8, and s16 with separate scale blocks.
- Pass only fused feature maps to the decoder.
- Return relevance logits and diagnostics when requested.

### `src/model/decoder.py`

- Rename camera lateral inputs to fused lateral inputs.
- Keep the FPN topology and prediction heads unchanged.

### `src/model/loss.py`

- Add balanced point-relevance loss.
- Include `radar_relevance_loss` in the returned loss dictionary.
- Preserve existing CenterNet target generation and detection weights.

### `src/training/train.py`

- Accumulate and report relevance loss.
- Save relevance-loss history in the existing JSON format.
- Keep checkpoint selection based on validation mAP50:95.

### `src/evaluation/evaluate.py`

- Support normal, masked-radar, and shuffled-position evaluation modes through
  internal evaluation functions, while the default command remains unchanged.
- Report radar-use deltas and supported/unsupported subsets without changing
  the standard detection metric definitions.

### `src/config/config.yaml`

- Keep one configuration file.
- Add V2 fusion and relevance settings.
- Use separate V2 checkpoint, history, and output paths so baseline artifacts
  cannot be overwritten.

## 8. Configuration

The intended V2 configuration section is:

```yaml
model:
  fusion:
    scales: [s4, s8, s16]
    window_size: 5
    vertical_neighbor_windows: 1
    dropout: 0.1
  radar_relevance:
    enabled: true
    loss_weight: 0.1

train:
  checkpoint: checkpoints/v2_local_multiscale_best.pt
  history: outputs/v2_local_multiscale_history.json

evaluation:
  checkpoint: checkpoints/v2_local_multiscale_best.pt
  output_dir: outputs/v2_local_multiscale_val
```

No second YAML file and no `--config` command-line option are introduced.

## 9. Evaluation design

All comparisons use the fixed scene-disjoint protocol already committed on
`main`:

```text
136 train scenes, frame_stride = 2
34 validation scenes, all keyframes
same seed, batch size, learning rate, early stopping, and metrics
```

### 9.1 Required runs

1. V1 baseline: global s16 fusion.
2. V2 normal radar.
3. V2 with every radar point masked at inference.
4. V2 with projected radar positions shuffled within each sample at inference.

Masked and shuffled runs use the same trained V2 checkpoint. They diagnose
model reliance and spatial association but are not replacements for a separately
trained camera-only baseline.

### 9.2 Required metrics

- AP50, AP75, and mAP50:95.
- Precision and recall at the documented operating threshold.
- Small, medium, and large AP where supported by the evaluator.
- Radar relevance precision, recall, and AUROC on valid labelled points.
- Normal-minus-masked and normal-minus-shuffled detection deltas.
- Results split by samples/objects with and without labelled radar support.
- Mean occupied windows per scale.
- Mean real radar points and local attention pairs per occupied window.
- End-to-end latency using the existing local machine, labelled as a local
  measurement rather than an edge-device benchmark.

### 9.3 Success criteria

V2 is considered to use radar meaningfully when all of the following hold:

- Normal radar outperforms masked radar on radar-supported validation cases.
- Shuffling radar positions reduces performance relative to normal radar,
  showing that spatial association matters.
- Unsupported cases do not materially regress relative to masked radar.
- Relevance predictions separate positive and negative radar returns better
  than chance.
- The full validation mAP does not regress relative to the fixed V1 baseline.
- Measured local attention pairs are substantially below the equivalent global
  all-pairs count.

If normal, masked, and shuffled performance are indistinguishable, the model
still does not use radar even if attention weights appear non-zero.

## 10. Risks and mitigations

### Sparse positive relevance labels

Pedestrians often lack radar returns, so positives may be rare. Balanced
positive/negative loss aggregation prevents the trivial all-negative solution.

### Noisy accumulated-sweep labels

Past-sweep returns from moving objects can be displaced. The relevance target
uses an ignore boundary around boxes, and the relevance loss remains auxiliary
rather than replacing detection supervision.

### Window-boundary misses

Vertical neighbouring windows are included because radar height is weak. If
visual diagnostics show horizontal boundary failures, shifted or neighbouring
horizontal windows become a later isolated experiment.

### High-resolution cost

s4 fusion uses true occupied-window batching and a 96-dimensional attention
space. A full-scene dense attention mask is explicitly rejected.

### Forced noise injection

Camera residuals remain at every scale, and relevance-weighted values suppress
clutter. The design never requires a non-empty window to contribute a minimum
radar magnitude.

### Camera shortcut remains mathematically possible

Any residual multimodal model can learn to ignore an update. The relevance
loss, all-scale fusion, and ablation protocol make this behaviour less likely
and directly measurable rather than pretending it is structurally impossible.

## 11. Implementation sequence

Implementation proceeds in small verified steps after this design is approved:

1. Add and verify radar relevance targets in the data path.
2. Update collation and augmentation invariants.
3. Update `RadarPointEncoder` outputs and relevance head; remove null token.
4. Implement one-scale point-window partition, occupied-window attention,
   merge, and exact empty-window bypass.
5. Generalize the block to s4, s8, and s16 with scale-specific dimensions.
6. Update detector and decoder interfaces.
7. Add relevance loss and training history fields.
8. Add radar normal/masked/shuffled evaluation and diagnostics.
9. Run a small overfit/smoke experiment before the full 136-scene training.
10. Train V2 and compare it to the fixed V1 baseline.

Each implementation step will be reviewed and run before moving to the next;
the full model will not be rewritten in a single change.
