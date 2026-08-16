from pathlib import Path

from PIL import ImageDraw
from torchvision.transforms import functional as F


def save_detection_visualization(
    image,
    target,
    detection,
    output_path,
    score_threshold=0.1,
):
    """Draw ground-truth and predicted pedestrian boxes on one image."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rendered_image = F.to_pil_image(
        image.detach().cpu().clamp(0.0, 1.0)
    )
    draw = ImageDraw.Draw(rendered_image)

    ground_truth_boxes = target["boxes"].detach().cpu()
    for box in ground_truth_boxes:
        x1, y1, x2, y2 = box.tolist()
        draw.rectangle(
            (x1, y1, x2, y2),
            outline="red",
            width=3,
        )
        draw.text(
            (x1, max(0.0, y1 - 12.0)),
            "GT pedestrian",
            fill="red",
        )

    predicted_boxes = detection["boxes"].detach().cpu()
    predicted_scores = detection["scores"].detach().cpu()

    for box, score in zip(predicted_boxes, predicted_scores):
        if float(score) < score_threshold:
            continue

        x1, y1, x2, y2 = box.tolist()
        draw.rectangle(
            (x1, y1, x2, y2),
            outline="lime",
            width=2,
        )
        draw.text(
            (x1, min(float(rendered_image.height - 12), y2 + 2.0)),
            f"Pred {float(score):.2f}",
            fill="lime",
        )

    rendered_image.save(output_path)
    return rendered_image
