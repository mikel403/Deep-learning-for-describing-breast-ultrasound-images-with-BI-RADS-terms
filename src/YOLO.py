"""Detect breast lesion bounding boxes using YOLO."""

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent / "models" / "YOLO.pt"
)


@lru_cache(maxsize=1)
def _load_model(model_path):
    """Load and cache the YOLO model."""
    return YOLO(str(model_path))


def convertImageCV(image):
    """Read a binary file-like object into a BGR image array."""
    image_bytes = image.read()

    if not image_bytes:
        raise ValueError("The image file is empty or has already been read.")

    image_array = cv2.imdecode(
        np.frombuffer(image_bytes, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )

    if image_array is None:
        raise ValueError("Could not decode the image.")

    return image_array


def yoloCrop(image, model_path=None):
    """Return detected bounding boxes as x, y, width and height.

    Coordinates are expressed in pixels relative to the original image.
    Returns an empty list when no lesions are detected.
    """
    path = (
        DEFAULT_MODEL_PATH
        if model_path is None
        else Path(model_path).expanduser().resolve()
    )
    model = _load_model(path)
    result = model(image, verbose=False)[0]

    if result.boxes is None:
        return []

    coordinates = result.boxes.xyxy.cpu().tolist()

    return [
        {
            "x": x1,
            "y": y1,
            "width": x2 - x1,
            "height": y2 - y1,
        }
        for x1, y1, x2, y2 in coordinates
    ]
