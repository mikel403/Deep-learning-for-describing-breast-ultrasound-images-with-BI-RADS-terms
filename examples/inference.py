"""Run descriptor and BI-RADS inference on a lesion image.

Run from the repository root:
    python -m examples.inference path/to/lesion.png
"""

import argparse
import json
from pathlib import Path

from src.descriptionModel import results_simple


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Predict ultrasound descriptors and BI-RADS probabilities "
            "from a cropped lesion image."
        )
    )
    parser.add_argument(
        "image",
        type=Path,
        help="Path to a cropped lesion image.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional path to the descriptor model checkpoint.",
    )
    parser.add_argument(
        "--naive-model-path",
        type=Path,
        default=None,
        help="Optional path to the Naive Bayes model.",
    )
    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"Image file not found: {args.image}")

    indices, words, probabilities = results_simple(
        args.image,
        model_path=args.model_path,
        naive_model_path=args.naive_model_path,
    )

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

    output = {
        "image": str(args.image),
        "predictions": dict(zip(labels, words)),
        "indices": indices,
        "probabilities": probabilities,
    }

    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()