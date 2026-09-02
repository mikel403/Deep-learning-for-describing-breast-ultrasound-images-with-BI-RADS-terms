"""Detect lesions and predict their descriptors and BI-RADS categories.

Run from the repository root:
    python -m examples.inference_yolo path/to/ultrasound.png
"""

import argparse
import json
import math
from pathlib import Path

import cv2

from src.YOLO import yoloCrop
from src.openImage import openImage
from src.descriptionModel import results_simple


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Detect ultrasound lesions with YOLO and predict "
            "descriptors and BI-RADS probabilities for each lesion."
        )
    )
    parser.add_argument(
        "image",
        type=Path,
        help="Path to an ultrasound image.",
    )
    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"Image file not found: {args.image}")

    # YOLO receives the color image in OpenCV's BGR format.
    color_image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if color_image is None:
        parser.error(f"Could not read image: {args.image}")

    # Use the same grayscale loading as the descriptor-only example.
    grayscale_image = openImage(args.image)

    if color_image.shape[:2] != grayscale_image.shape:
        raise ValueError("Color and grayscale image dimensions differ.")

    height, width = grayscale_image.shape
    boxes = yoloCrop(color_image)

    labels = (
        "shape",
        "margin",
        "orientation",
        "echogenicity",
        "posterior",
        "suggestivity",
        "benign_malignant",
        "birads",
    )
    lesions = []

    for box in boxes:
        # Round outward and keep coordinates within the image.
        x1 = max(0, min(width, math.floor(box["x"])))
        y1 = max(0, min(height, math.floor(box["y"])))
        x2 = max(
            0, min(width, math.ceil(box["x"] + box["width"]))
        )
        y2 = max(
            0, min(height, math.ceil(box["y"] + box["height"]))
        )

        if x2 <= x1 or y2 <= y1:
            continue

        # NumPy indexing uses rows first: [y1:y2, x1:x2].
        crop = grayscale_image[y1:y2, x1:x2].copy()

        indices, words, probabilities = results_simple(crop)

        lesions.append({
            "bounding_box": {
                "x": x1,
                "y": y1,
                "width": x2 - x1,
                "height": y2 - y1,
            },
            "predictions": dict(zip(labels, words)),
            "indices": indices,
            "probabilities": probabilities,
        })

    output = {
        "image": str(args.image),
        "lesion_count": len(lesions),
        "lesions": lesions,
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()