import torch
from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops import box_iou


class PedestrianDetectionMetrics:
    """Accumulate single-class 2D pedestrian detection metrics."""

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
        self.metric = MeanAveragePrecision(
            box_format="xyxy",
            iou_type="bbox",
            iou_thresholds=self.iou_thresholds,
            extended_summary=True,
        )
        self.detections = []
        self.targets = []

    @staticmethod
    def _prepare_detection(detection):
        boxes = detection["boxes"].detach().cpu().float()
        scores = detection["scores"].detach().cpu().float()
        return {
            "boxes": boxes,
            "scores": scores,
            "labels": torch.zeros(len(boxes), dtype=torch.int64),
        }

    @staticmethod
    def _prepare_target(target):
        boxes = target["boxes"].detach().cpu().float()
        return {
            "boxes": boxes,
            "labels": torch.zeros(len(boxes), dtype=torch.int64),
        }

    def update(self, detections, targets):
        if len(detections) != len(targets):
            raise ValueError("detections and targets must have the same batch size.")

        # turn into the format expected by the metric
        prepared_detections = [self._prepare_detection(detection) for detection in detections]
        prepared_targets = [self._prepare_target(target) for target in targets]

        self.metric.update(prepared_detections, prepared_targets)
        self.detections.extend(prepared_detections)
        self.targets.extend(prepared_targets)

    def _match_predictions(self):
        all_scores = []
        all_true_positives = []
        number_of_ground_truths = 0

        for detection, target in zip(self.detections, self.targets):
            predicted_boxes = detection["boxes"]
            predicted_scores = detection["scores"]
            ground_truth_boxes = target["boxes"]
            number_of_ground_truths += len(ground_truth_boxes)

            order = torch.argsort(predicted_scores, descending=True) # sort the boxes by score
            predicted_boxes = predicted_boxes[order]
            predicted_scores = predicted_scores[order]
            matched_ground_truths = torch.zeros( # one gt can only be matched once
                len(ground_truth_boxes),
                dtype=torch.bool,
            )

            for predicted_box, score in zip(predicted_boxes, predicted_scores):
                is_true_positive = False

                if len(ground_truth_boxes) > 0:
                    overlaps = box_iou(
                        predicted_box.unsqueeze(0),
                        ground_truth_boxes,
                    )[0]
                    overlaps[matched_ground_truths] = -1.0
                    best_overlap, best_index = overlaps.max(dim=0)

                    # if the best overlap is greater than iou threshold, mark it as a true positive
                    if float(best_overlap) >= self.report_iou_threshold:
                        matched_ground_truths[best_index] = True
                        is_true_positive = True

                all_scores.append(score)
                all_true_positives.append(is_true_positive)

        if not all_scores:
            return (
                torch.empty(0, dtype=torch.float32),
                torch.empty(0, dtype=torch.bool),
                number_of_ground_truths,
            )

        scores = torch.stack(all_scores)
        true_positives = torch.tensor(all_true_positives, dtype=torch.bool)
        order = torch.argsort(scores, descending=True)
        return scores[order], true_positives[order], number_of_ground_truths

    def _ap_by_iou(self, metric_results):
        precision = metric_results["precision"]
        ap_by_iou = {}

        for index, threshold in enumerate(self.iou_thresholds):
            values = precision[index, :, :, 0, -1]
            valid_values = values[values >= 0]
            ap_by_iou[f"{threshold:.2f}"] = (
                float(valid_values.mean()) if valid_values.numel() else 0.0
            )

        return ap_by_iou

    @staticmethod
    def _to_float(metric_results, key):
        return float(metric_results[key].detach().cpu())

    def compute(self):
        if not self.targets:
            raise RuntimeError("No samples were added to the metrics accumulator.")

        metric_results = self.metric.compute()
        scores, true_positives, number_of_ground_truths = (
            self._match_predictions()
        )
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
            "ap50": self._to_float(metric_results, "map_50"),
            "ap75": self._to_float(metric_results, "map_75"),
            "map_50_95": self._to_float(metric_results, "map"),
            "map_small": self._to_float(metric_results, "map_small"),
            "map_medium": self._to_float(metric_results, "map_medium"),
            "map_large": self._to_float(metric_results, "map_large"),
            "mar_1": self._to_float(metric_results, "mar_1"),
            "mar_10": self._to_float(metric_results, "mar_10"),
            "mar_100": self._to_float(metric_results, "mar_100"),
            "ap_by_iou": self._ap_by_iou(metric_results),
            "report_iou_threshold": self.report_iou_threshold,
            "report_score_threshold": self.report_score_threshold,
            "precision": precision,
            "recall": recall,
            "true_positives": true_positive_count,
            "false_positives": false_positive_count,
            "false_negatives": false_negative_count,
            "ground_truths": number_of_ground_truths,
        }
