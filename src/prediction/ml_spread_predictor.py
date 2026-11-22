
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT / "models" / "spread_sgd_classifier_v2.pkl"
)

NUMERICAL_FEATURES = [
    "firearea",
    "fwi", "isi", "ffmc", "dmc", "dc", "bui",
    "prec", "tmax", "vpd", "ws", "rh",
    "fwi_prev1", "fwi_prev2",
    "rh_prev1", "rh_prev2",
    "Biomass", "Closure", "prcC",
    "peatprop",
    "hydrodens2k", "hydrodens5k",
    "hydrodens10k", "hydrodens25k",
    "roaddens2k", "roaddens5k",
    "roaddens10k", "roaddens25k",
    "roaddist",
    "nonfuel1k", "nonfuel2k",
    "nonfuel5k", "nonfuel10k",
    "dem", "slope", "twi", "aspect",
    "cumuarea", "pctgrowth", "prevgrow",
]

CATEGORICAL_FEATURES = [
    "ecozone",
    "peattype",
]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


@lru_cache(maxsize=1)
def load_spread_model():
    """Load the trained wildfire spread classification model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Spread model not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def predict_spread_risk(
    features: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Predict whether significant fire growth will occur
    on the next day.

    Returns the predicted class and its estimated probability.
    """

    missing_features = [
        name for name in ALL_FEATURES
        if name not in features
    ]

    if missing_features:
        raise ValueError(
            f"Missing required features: {missing_features}"
        )

    input_data = pd.DataFrame(
        [{name: features[name] for name in ALL_FEATURES}]
    )

    for column in CATEGORICAL_FEATURES:
        value = input_data.at[0, column]

        if pd.notna(value):
            input_data.at[0, column] = str(value)

    model = load_spread_model()

    probabilities = model.predict_proba(input_data)[0]
    positive_class_index = list(model.classes_).index(1)

    spread_probability = float(
        probabilities[positive_class_index]
    )

    prediction = int(model.predict(input_data)[0])

    return {
        "spread_probability": spread_probability,
        "spread_prediction": prediction,
        "prediction_label": (
            "Significant next-day growth"
            if prediction == 1
            else "No significant next-day growth"
        ),
    }
