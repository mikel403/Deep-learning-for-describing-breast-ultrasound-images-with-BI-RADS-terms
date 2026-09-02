"""Estimate BI-RADS category probabilities using a Naive Bayes model."""

import pickle
from functools import lru_cache
from pathlib import Path


DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "modelo_naive_bayes.pkl"
)


@lru_cache(maxsize=1)
def _load_model(model_path):
    """Load and cache the model from a trusted pickle file."""
    with model_path.open("rb") as file:
        return pickle.load(file)


def predict_naive(input_data, model_path=None):
    """Return category probabilities for the supplied feature matrix.

    Features must follow the same order and encoding used during training.
    Probability columns follow the model's classes_ order.
    """
    path = (
        DEFAULT_MODEL_PATH
        if model_path is None
        else Path(model_path).expanduser().resolve()
    )
    model = _load_model(path)
    return model.predict_proba(input_data)