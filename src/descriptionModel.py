"""Predict BI-RADS descriptors and category probabilities."""

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.models._utils import IntermediateLayerGetter

from .openImage import openImage
from .biradsModel import predict_naive


DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "model_birads5.pth"
)

IMAGE_SIZE = 450

# Alphabetical order must match the original training code.
Shape = sorted(["irregular", "oval", "round"])
Margin = sorted([
    "angulated",
    "circumscribed",
    "spiculated",
    "indistinct",
    "microlobulated",
])
Orientation = sorted(["parallel", "no orientation", "not parallel"])
Echogenicity = sorted([
    "anechoic",
    "heterogeneous",
    "hypoechoic",
    "isoechoic",
])
Posterior = sorted(["enhancement", "no features", "shadowing"])
Suggestivity = sorted(["complicated cyst", "simple cyst", "other"])
results = ["benign", "malignant"]
BIRADS = ["2", "3", "4A", "4B", "4C", "5"]

DESCRIPTOR_GROUPS = (
    ("shape", Shape),
    ("margin", Margin),
    ("orientation", Orientation),
    ("echogenicity", Echogenicity),
    ("posterior", Posterior),
    ("suggestivity", Suggestivity),
    ("results", results),
)


def reshape_matmul(matrix_3d, matrix_2d):
    return torch.einsum("ijk,kl->ijl", matrix_3d, matrix_2d)


class simple_encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(1, 3, 3)
        self.gelu = nn.GELU()

        # All parameters are restored from the local checkpoint.
        backbone = torchvision.models.vgg16(weights=None)

        for parameter in backbone.parameters():
            parameter.requires_grad = False

        self.encoder = IntermediateLayerGetter(
            backbone.features,
            return_layers={"23": "pooling"},
        )

    def forward(self, inputs):
        features = self.gelu(self.conv(inputs))
        features = nn.functional.max_pool2d(features, 2)
        return self.encoder(features)["pooling"]


class Attention(nn.Module):
    def __init__(
        self,
        dropout=False,
        L2Attention=False,
        Gatted=False,
        L2dim=20,
        feature_dim=512,
    ):
        super().__init__()

        self.L2Attention = L2Attention
        self.Gatted = Gatted
        self.dropout = dropout

        if L2Attention or Gatted:
            self.w_a_1 = nn.Parameter(torch.rand(feature_dim, L2dim))
            self.b_a_1 = nn.Parameter(torch.zeros(L2dim))
            self.w_a_2 = nn.Parameter(torch.rand(L2dim, 1))
            self.b_a_2 = nn.Parameter(torch.zeros(1))

            if Gatted:
                self.w_a_g = nn.Parameter(torch.rand(feature_dim, L2dim))
                self.b_a_g = nn.Parameter(torch.zeros(L2dim))
        else:
            self.params = nn.ParameterDict({
                "w_a": nn.Parameter(torch.randn(feature_dim, 1)),
                "b_a": nn.Parameter(torch.randn(1)),
            })

        self.dropout_layer = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        features = inputs.flatten(start_dim=2).permute(0, 2, 1)

        if self.L2Attention or self.Gatted:
            hidden = torch.tanh(
                reshape_matmul(features, self.w_a_1) + self.b_a_1
            )

            if self.Gatted:
                gate = torch.sigmoid(
                    reshape_matmul(features, self.w_a_g) + self.b_a_g
                )
                hidden = hidden * gate

            scores = reshape_matmul(hidden, self.w_a_2) + self.b_a_2
        else:
            scores = torch.tanh(
                reshape_matmul(features, self.params["w_a"])
                + self.params["b_a"]
            )

        weights = self.softmax(scores.squeeze(-1))
        context = torch.sum(features * weights.unsqueeze(-1), dim=1)

        if self.dropout:
            context = self.dropout_layer(context)

        return context, weights


class decoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Preserve checkpoint layer names and dimensions.
        self.forma_d = nn.Linear(512, 3)
        self.margen_d = nn.Linear(512, 5)
        self.orientacion_d = nn.Linear(512, 3)
        self.ecogenicidad_d = nn.Linear(512, 4)
        self.posterior_d = nn.Linear(512, 3)
        self.sugestividad_d = nn.Linear(530, 3)
        self.benignidad_d = nn.Linear(533, 2)

    def forward(self, inputs):
        shape = self.forma_d(inputs)
        margin = self.margen_d(inputs)
        orientation = self.orientacion_d(inputs)
        echogenicity = self.ecogenicidad_d(inputs)
        posterior = self.posterior_d(inputs)

        # Subsequent heads receive logits, as in the original model.
        features = torch.cat(
            (inputs, shape, margin, orientation, echogenicity, posterior),
            dim=-1,
        )
        suggestivity = self.sugestividad_d(features)

        final_features = torch.cat((features, suggestivity), dim=-1)
        diagnosis = self.benignidad_d(final_features)

        return tuple(
            torch.softmax(logits, dim=1)
            for logits in (
                shape,
                margin,
                orientation,
                echogenicity,
                posterior,
                suggestivity,
                diagnosis,
            )
        )


class Att_model(nn.Module):
    def __init__(
        self,
        dropout=False,
        L2Attention=False,
        Gatted=False,
        L2dim=20,
        feature_dim=512,
    ):
        super().__init__()

        self.encoder = simple_encoder()
        self.norm = nn.BatchNorm2d(512)
        self.Att = Attention(
            dropout=dropout,
            L2Attention=L2Attention,
            Gatted=Gatted,
            L2dim=L2dim,
            feature_dim=feature_dim,
        )
        self.decoder = decoder()

    def forward(self, inputs):
        features = self.norm(self.encoder(inputs))
        context, _ = self.Att(features)
        return self.decoder(context)


@lru_cache(maxsize=1)
def _load_model(model_path):
    """Load the descriptor model on CPU and cache it."""
    model = Att_model().cpu()
    state_dict = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def bound_img(image):
    """Resize, pad and normalize a grayscale image to (1, 1, 450, 450)."""
    if image.ndim != 2 or image.size == 0:
        raise ValueError("Expected a non-empty grayscale image.")

    image = torch.from_numpy(np.expand_dims(image, axis=0))
    _, height, width = image.shape
    ratio = height / width

    if height > IMAGE_SIZE or width > IMAGE_SIZE:
        if ratio > 1:
            new_size = (IMAGE_SIZE, int(IMAGE_SIZE / ratio))
        else:
            new_size = (int(IMAGE_SIZE * ratio), IMAGE_SIZE)

        if min(new_size) < 1:
            raise ValueError("Image aspect ratio is too extreme to resize.")

        image = torchvision.transforms.Resize(new_size)(image)
        _, height, width = image.shape

    left = (IMAGE_SIZE - width) // 2
    top = (IMAGE_SIZE - height) // 2
    right = IMAGE_SIZE - width - left
    bottom = IMAGE_SIZE - height - top

    image = torchvision.transforms.Pad(
        [left, top, right, bottom]
    )(image)

    return image.unsqueeze(0) / 255.0


def convert_to_words(desc, threshold=0.2):
    """Convert predictions into labels and corrected Naive Bayes inputs.

    Rules modify selected categories, while probability dictionaries
    retain the original neural network probabilities.

    The threshold argument is retained for compatibility but is unused.
    """
    if len(desc) != len(DESCRIPTOR_GROUPS):
        raise ValueError("Expected seven model outputs.")

    indices = []
    probabilities = []

    for output, (_, labels) in zip(desc, DESCRIPTOR_GROUPS):
        values = torch.as_tensor(output).detach().cpu().reshape(-1)

        if values.numel() != len(labels):
            raise ValueError("Expected predictions for a single image.")

        indices.append(int(values.argmax().item()))
        probabilities.append(values.tolist())

    # Round lesions have no orientation.
    if Shape[indices[0]] == "round":
        indices[2] = Orientation.index("no orientation")

    # A hypoechoic lesion predicted as a simple cyst becomes "other".
    if (
        Echogenicity[indices[3]] == "hypoechoic"
        and Suggestivity[indices[5]] == "simple cyst"
    ):
        indices[5] = Suggestivity.index("other")

    words = [
        labels[index]
        for index, (_, labels) in zip(indices, DESCRIPTOR_GROUPS)
    ]

    # Hide "other" in the displayed text, keeping its index for Naive Bayes.
    if words[5] == "other":
        words[5] = ""

    prob_dic = {
        name: {
            label: round(float(value), 4)
            for label, value in zip(labels, values)
        }
        for (name, labels), values in zip(
            DESCRIPTOR_GROUPS[:-1],
            probabilities[:-1],
        )
    }

    return words, indices[:-1], prob_dic

@torch.inference_mode()
def results_simple(imag, model=None, model_path=None, naive_model_path=None):
    """Predict descriptors and BI-RADS probabilities from a local image path.

    Returns:
        indices: Six descriptor indices followed by the BI-RADS index.
        words: Seven predicted labels followed by the BI-RADS category.
        prob_dic: Descriptor and BI-RADS probability dictionaries.
    """
    if model is not None and model_path is not None:
        raise ValueError("Pass either model or model_path, not both.")

    if model is None:
        path = (
            DEFAULT_MODEL_PATH
            if model_path is None
            else Path(model_path).expanduser().resolve()
        )
        model = _load_model(path)

    model.eval()
    image = openImage(imag) if isinstance(imag, (str, Path)) else imag
    image = bound_img(image)
    image = image.to(next(model.parameters()).device)

    predictions = model(image)
    words, features, prob_dic = convert_to_words(predictions)

    birads_probabilities = predict_naive(
        [features],
        model_path=naive_model_path,
    )[0]

    if len(birads_probabilities) != len(BIRADS):
        raise ValueError("Expected six BI-RADS category probabilities.")

    # Assumes the same class order as the original Naive Bayes model.
    prob_dic["birads"] = {
        label: round(float(value), 4)
        for label, value in zip(BIRADS, birads_probabilities)
    }

    birads_index = int(np.argmax(birads_probabilities))
    indices = features + [birads_index]
    words.append(BIRADS[birads_index])

    return indices, words, prob_dic