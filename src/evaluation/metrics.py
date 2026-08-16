import torch


def box_iou(boxes1, boxes2):
    """Calculate pairwise IoU for two sets of xyxy image-space boxes."""
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros(
            boxes1.shape[0],
            boxes2.shape[0],
            dtype=torch.float32,
        )

    boxes1 = boxes1.to(dtype=torch.float32)
    boxes2 = boxes2.to(dtype=torch.float32)

    top_left = torch.maximum(
        boxes1[:, None, :2],
        boxes2[None, :, :2],
    )
    bottom_right = torch.minimum(
        boxes1[:, None, 2:],
        boxes2[None, :, 2:],
    )
    intersection_size = (bottom_right - top_left).clamp(min=0.0)
    intersection = intersection_size[..., 0] * intersection_size[..., 1]

    area1_size = (boxes1[:, 2:] - boxes1[:, :2]).clamp(min=0.0)
    area2_size = (boxes2[:, 2:] - boxes2[:, :2]).clamp(min=0.0)
    area1 = area1_size[:, 0] * area1_size[:, 1]
    area2 = area2_size[:, 0] * area2_size[:, 1]

    union = area1[:, None] + area2[None, :] - intersection
    return torch.where(
        union > 0.0,
        intersection / union,
        torch.zeros_like(intersection),
    )


class PedestrianDetectionMetrics:
    """Accumulate single-class 2D detections and calculate IoU-based AP."""

    def __init__(
        self,
        iou_thresholds=None,
        report_iou_threshold=0.5,
        report_score_threshold=0.1,
    ):
        if iou_thresholds is None:
            iou_thresholds = [
                0.50,
                0.55,
                0.60,
                0.65,
                0.70,
                0.75,
                0.80,
                0.85,
                0.90,
                0.95,
            ]

        self.iou_thresholds = [float(value) for value in iou_thresholds]
        self.report_iou_threshold = float(report_iou_threshold)
        self.report_score_threshold = float(report_score_threshold)
        self.reset()

    def reset(self):
        self.detections = []
        self.targets = []

    def update(self, detections, targets):
        if len(detections) != len(targets):
            raise ValueError("detections and targets must have the same batch size.")

        for detection, target in zip(detections, targets):
            self.detections.append(
                {
                    "boxes": detection["boxes"].detach().cpu().float(),
                    "scores": detection["scores"].detach().cpu().float(),
                }
            )
            self.targets.append(
                {
                    "boxes": target["boxes"].detach().cpu().float(),
                }
            )

    def _match_predictions(self, iou_threshold):
        all_scores = []
        all_true_positives = []
        number_of_ground_truths = 0

        for detection, target in zip(self.detections, self.targets):
            predicted_boxes = detection["boxes"]
            predicted_scores = detection["scores"]
            ground_truth_boxes = target["boxes"]
            number_of_ground_truths += ground_truth_boxes.shape[0]

            order = torch.argsort(predicted_scores, descending=True)
            predicted_boxes = predicted_boxes[order]
            predicted_scores = predicted_scores[order]
            matched_ground_truths = torch.zeros(
                ground_truth_boxes.shape[0],
                dtype=torch.bool,
            )

            for predicted_box, score in zip(
                predicted_boxes,
                predicted_scores,
            ):
                is_true_positive = False

                if ground_truth_boxes.shape[0] > 0:
                    overlaps = box_iou(
                        predicted_box.unsqueeze(0),
                        ground_truth_boxes,
                    )[0]
                    overlaps[matched_ground_truths] = -1.0
                    best_overlap, best_index = overlaps.max(dim=0)

                    if float(best_overlap) >= iou_threshold:
                        matched_ground_truths[best_index] = True
                        is_true_positive = True

                all_scores.append(score)
                all_true_positives.append(is_true_positive)

        if all_scores:
            scores = torch.stack(all_scores)
            true_positives = torch.tensor(
                all_true_positives,
                dtype=torch.bool,
            )
            order = torch.argsort(scores, descending=True)
            scores = scores[order]
            true_positives = true_positives[order]
        else:
            scores = torch.empty(0, dtype=torch.float32)
            true_positives = torch.empty(0, dtype=torch.bool)

        return scores, true_positives, number_of_ground_truths

    @staticmethod
    def _calculate_ap(true_positives, number_of_ground_truths):
        if number_of_ground_truths == 0 or true_positives.numel() == 0:
            return 0.0

        true_positive_count = true_positives.to(torch.float32).cumsum(0)
        false_positive_count = (~true_positives).to(torch.float32).cumsum(0)
        recall = true_positive_count / number_of_ground_truths
        precision = true_positive_count / (
            true_positive_count + false_positive_count
        ).clamp(min=1.0)

        interpolated_precisions = []
        for recall_level in torch.linspace(0.0, 1.0, 101):
            valid = recall >= recall_level
            if valid.any():
                interpolated_precisions.append(precision[valid].max())
            else:
                interpolated_precisions.append(torch.tensor(0.0))

        return float(torch.stack(interpolated_precisions).mean())

    def compute(self):
        if not self.targets:
            raise RuntimeError("No samples were added to the metrics accumulator.")

        ap_by_iou = {}
        matched_results = {}

        for threshold in self.iou_thresholds:
            scores, true_positives, number_of_ground_truths = (
                self._match_predictions(threshold)
            )
            ap_by_iou[f"{threshold:.2f}"] = self._calculate_ap(
                true_positives,
                number_of_ground_truths,
            )
            matched_results[round(threshold, 2)] = (
                scores,
                true_positives,
                number_of_ground_truths,
            )

        report_key = round(self.report_iou_threshold, 2)
        if report_key not in matched_results:
            report_result = self._match_predictions(
                self.report_iou_threshold
            )
        else:
            report_result = matched_results[report_key]

        scores, true_positives, number_of_ground_truths = report_result
        report_keep = scores >= self.report_score_threshold
        true_positive_count = int(true_positives[report_keep].sum())
        prediction_count = int(report_keep.sum())
        false_positive_count = prediction_count - true_positive_count
        false_negative_count = max(
            0,
            number_of_ground_truths - true_positive_count,
        )

        precision = (
            true_positive_count / prediction_count
            if prediction_count > 0
            else 0.0
        )
        recall = (
            true_positive_count / number_of_ground_truths
            if number_of_ground_truths > 0
            else 0.0
        )

        return {
            "ap50": ap_by_iou.get("0.50", 0.0),
            "ap75": ap_by_iou.get("0.75", 0.0),
            "map_50_95": sum(ap_by_iou.values()) / len(ap_by_iou),
            "ap_by_iou": ap_by_iou,
            "report_iou_threshold": self.report_iou_threshold,
            "report_score_threshold": self.report_score_threshold,
            "precision": precision,
            "recall": recall,
            "true_positives": true_positive_count,
            "false_positives": false_positive_count,
            "false_negatives": false_negative_count,
            "ground_truths": number_of_ground_truths,
        }
