# Deep Learning for Describing Breast Ultrasound Images with BI-RADS Terms

Inference code and pretrained models for breast ultrasound lesion detection, BI-RADS descriptor prediction, and BI-RADS category estimation.

This repository contains the model components used in [BreastUSTraining](https://github.com/mikel403/BreastUSTraining/) and is associated with the publication:

> Carrilero-Mardones, M., Parras-Jurado, M., Nogales, A., Pérez-Martín, J., & Díez, F. J. (2024). Deep Learning for Describing Breast Ultrasound Images with BI-RADS Terms. *Journal of Imaging Informatics in Medicine, 37*, 2940–2954. https://doi.org/10.1007/s10278-024-01155-1

[Article on PubMed](https://pubmed.ncbi.nlm.nih.gov/38926264/)

## Relationship to the published method

The published study uses **multinomial logistic regression** to estimate the BI-RADS category. This repository uses a **Naive Bayes classifier** instead.

In subsequent experiments, Naive Bayes achieved numerically higher accuracy and kappa scores, although the differences were not statistically significant. We selected it for its greater interpretability in this application.

The BI-RADS classification stage therefore differs from the published method. The results reported in the paper should not be attributed to this implementation.

This repository focuses on **inference with pretrained models**. It does not provide a complete workflow for reproducing the training and evaluation reported in the publication.

## Pipeline

The complete inference pipeline consists of:

1. **Lesion detection:** YOLO predicts bounding boxes in an ultrasound image.
2. **Lesion cropping:** each detected region is extracted from the grayscale image.
3. **Descriptor prediction:** a neural network predicts ultrasound descriptors and benign/malignant probabilities.
4. **Rule-based postprocessing:** selected descriptor categories are adjusted using the rules described below.
5. **BI-RADS estimation:** Naive Bayes receives the six selected descriptor indices and returns BI-RADS category probabilities.

The descriptor model can also be used directly on an already cropped lesion image.

## Repository structure

```text
.
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── .gitignore
├── models/
│   ├── YOLO.pt
│   ├── model_birads5.pth
│   └── modelo_naive_bayes.pkl
├── src/
│   ├── __init__.py
│   ├── YOLO.py
│   ├── openImage.py
│   ├── descriptionModel.py
│   └── biradsModel.py
└── examples/
    ├── __init__.py
    ├── inference.py
    └── inference_yolo.py
```

## Installation
The code has been tested with Python 3.9.18 and the dependency versions
specified in `requirements.txt`.

Clone the repository:

```bash
git clone https://github.com/mikel403/Deep-learning-for-describing-breast-ultrasound-images-with-BI-RADS-terms.git
cd Deep-learning-for-describing-breast-ultrasound-images-with-BI-RADS-terms
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The main dependencies are PyTorch, torchvision, Ultralytics, OpenCV, NumPy, and scikit-learn.

Compatibility with the saved checkpoints and pickle model depends on the installed dependency versions. Use the versions specified in `requirements.txt`.

## Pretrained models

Place the following files in `models/`:

| File | Purpose |
| --- | --- |
| `YOLO.pt` | Ultrasound lesion detection |
| `model_birads5.pth` | Descriptor and benign/malignant prediction |
| `modelo_naive_bayes.pkl` | BI-RADS category estimation |

Default model paths are resolved relative to the source files, independently of the current working directory.

Models are loaded when first used and cached for subsequent predictions within the same Python process. The descriptor model runs on CPU by default; YOLO uses Ultralytics' default device selection.

Only load model files from trusted sources. In particular, loading a pickle file can execute code.

## Usage

Run the following commands from the repository root.

### Predict from a cropped lesion image

Use this example when the image already contains the lesion region to be analyzed:

```bash
python -m examples.inference path/to/lesion.png
```

The example predicts descriptors and a BI-RADS category without running YOLO.

### Detect lesions and analyze each crop

Use this example with an ultrasound image requiring lesion detection:

```bash
python -m examples.inference_yolo path/to/ultrasound.png
```

The example:

- Runs YOLO on the image.
- Rounds bounding box coordinates outward and clips them to the image boundaries.
- Extracts each valid crop.
- Predicts descriptors and BI-RADS probabilities for each crop.
- Prints the results as JSON.

Bounding boxes contain `x`, `y`, `width`, and `height`, measured in pixels relative to the original image. The origin is the top-left corner.

If no lesions are detected, the output contains `lesion_count: 0` and an empty `lesions` list.

### Use the descriptor model from Python

```python
from src.descriptionModel import results_simple

indices, words, probabilities = results_simple("path/to/lesion.png")

print("Selected labels:", words)
print("BI-RADS probabilities:", probabilities["birads"])
```

The function also accepts a two-dimensional grayscale NumPy array, such as a crop extracted from an image:

```python
from src.openImage import openImage
from src.descriptionModel import results_simple

image = openImage("path/to/ultrasound.png")

# Example coordinates; replace with a valid lesion bounding box.
x1, y1, x2, y2 = 50, 40, 250, 220
crop = image[y1:y2, x1:x2].copy()

indices, words, probabilities = results_simple(crop)
```

Arrays should use the same 0–255 intensity scale as images loaded by OpenCV. Do not normalize them to 0–1 before calling `results_simple`.

Alternative descriptor and Naive Bayes model paths can be supplied:

```python
indices, words, probabilities = results_simple(
    "path/to/lesion.png",
    model_path="path/to/model_birads5.pth",
    naive_model_path="path/to/modelo_naive_bayes.pkl",
)
```

### Use YOLO from Python

```python
import cv2

from src.YOLO import yoloCrop

image = cv2.imread("path/to/ultrasound.png")
if image is None:
    raise ValueError("Could not read image.")

boxes = yoloCrop(image)
print(boxes)
```

Despite its name, `yoloCrop` returns bounding boxes, not cropped images. See `examples/inference_yolo.py` for cropping and subsequent analysis.

## Predicted categories

Category indices follow the order shown below. This ordering must be preserved because it is tied to the trained models.

| Output | Categories in index order |
| --- | --- |
| Shape | `irregular`, `oval`, `round` |
| Margin | `angulated`, `circumscribed`, `indistinct`, `microlobulated`, `spiculated` |
| Orientation | `no orientation`, `not parallel`, `parallel` |
| Echogenicity | `anechoic`, `heterogeneous`, `hypoechoic`, `isoechoic` |
| Posterior features | `enhancement`, `no features`, `shadowing` |
| Suggestivity | `complicated cyst`, `other`, `simple cyst` |
| Benign/malignant | `benign`, `malignant` |
| BI-RADS | `2`, `3`, `4A`, `4B`, `4C`, `5` |

The Naive Bayes input consists of the selected indices for shape, margin, orientation, echogenicity, posterior features, and suggestivity, in that order. It does not include the neural network's benign/malignant prediction.

The current implementation maps Naive Bayes probability columns to the BI-RADS order above. This mapping must match the classifier's `classes_` order, particularly if replacing or retraining the classifier.

## Preprocessing and model architecture

The descriptor model processes grayscale lesion images.

Preprocessing:

1. Downscales images larger than 450 pixels in either dimension, preserving their aspect ratio.
2. Pads the image with zeros to obtain a 450 × 450 image.
3. Divides pixel values by 255.
4. Adds channel and batch dimensions to obtain a tensor of shape `(1, 1, 450, 450)`.

Smaller images are padded without upscaling.

The neural network uses a convolution mapping grayscale input to three channels, a VGG16 feature extractor, batch normalization, an attention mechanism, and prediction heads for the descriptor groups and benign/malignant classification.

The suggestivity and benign/malignant heads also receive outputs from earlier prediction heads. The architecture and layer names are retained to match the supplied checkpoint.

## Rule-based postprocessing

Two rules are applied to the selected descriptor categories before Naive Bayes inference:

- If the predicted shape is `round`, orientation is set to `no orientation`.
- If echogenicity is `hypoechoic` and suggestivity is `simple cyst`, suggestivity is changed to `other`.

These corrected indices are passed to Naive Bayes.

For display, `other` is represented by an empty string in the returned labels. Its category index remains present in the Naive Bayes input.

Descriptor probability dictionaries retain the original neural network probabilities. Consequently, a rule-adjusted label may differ from the category with the highest reported descriptor probability.

The benign/malignant neural network prediction is not recalculated after these rules.

## Output format

`results_simple` returns:

```python
indices, words, probabilities
```

### `indices`

A list containing seven zero-based category indices:

```text
shape, margin, orientation, echogenicity, posterior, suggestivity, BI-RADS
```

The six descriptor indices include the rule-based corrections.

### `words`

A list containing eight labels:

```text
shape, margin, orientation, echogenicity, posterior, suggestivity,
benign/malignant, BI-RADS
```

### `probabilities`

A dictionary with these keys:

```text
shape, margin, orientation, echogenicity, posterior, suggestivity, birads
```

Each key maps category names to probabilities rounded to four decimal places. Rounding can cause the displayed probabilities to sum to slightly more or less than one.

The current interface returns the selected benign/malignant label but does not expose its probability distribution.

## Relationship to BreastUSTraining

[BreastUSTraining](https://github.com/mikel403/BreastUSTraining/) is the application in which these model components are used.

This repository separates inference functionality from the application. The modules in `src/` do not require Django or depend on Django's `MEDIA_ROOT` setting.

## Intended use and limitations

This repository is intended for research and educational use. It is not intended for clinical diagnosis or patient management.

Predictions depend on image characteristics, lesion localization, preprocessing, and similarity to the data used to develop the models. Detection errors can affect all subsequent predictions.

The returned probabilities should not be interpreted as calibrated clinical risk estimates without appropriate validation.

The provided examples perform inference one lesion at a time. Training scripts, datasets, and a complete reproduction of the published evaluation are outside the scope of this repository.

## Citation

If you use this work, please cite the associated article:

```bibtex
@article{carrilero2024deep,
  title={Deep Learning for Describing Breast Ultrasound Images with BI-RADS Terms},
  author={Carrilero-Mardones, Mikel and Parras-Jurado, Manuela and Nogales, Alberto and P{\'e}rez-Mart{\'\i}n, Jorge and D{\'\i}ez, Francisco Javier},
  journal={Journal of Imaging Informatics in Medicine},
  volume={37},
  number={6},
  pages={2940--2954},
  year={2024},
  publisher={Springer},
  doi={10.1007/s10278-024-01155-1}
}
```

Citation metadata is also provided in `CITATION.cff`.

When describing this repository in a publication, identify the version or commit used and note that its BI-RADS classification stage uses Naive Bayes rather than the multinomial logistic regression described in the paper.

## License

Original code in this repository is provided under the MIT License. See `LICENSE`.

Third-party dependencies and model components remain subject to their respective licenses. The repository's MIT License does not override those terms.