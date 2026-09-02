"""Load ultrasound images in grayscale."""

import cv2


def openImage(image_path):
    """Load a local image as a grayscale NumPy array."""
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    return image

