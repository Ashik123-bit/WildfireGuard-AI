from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "wildfire_area_forecast.pkl"

FEATURES = (
    "firearea",
    "tmax",
    "vpd",
    "ws",
    "rh",
    "prec",
    "lat",
    "lon",
)

TRAINING_REGION = {
    "min_latitude": 43.6,
    "max_latitude": 69.3,
    "min_longitude": -141.0,
    "max_longitude": -55.4,
}


@lru_cache(maxsize=1)
def load_area_forecast_model() -> Any:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Area forecast model not found at {MODEL_PATH}. "
            "Run python -m src.prediction.train_area_forecast_model first."
        )
    return joblib.load(MODEL_PATH)


def is_in_training_region(latitude: float, longitude: float) -> bool:
    return (
        TRAINING_REGION["min_latitude"]
        <= latitude
        <= TRAINING_REGION["max_latitude"]
        and TRAINING_REGION["min_longitude"]
        <= longitude
        <= TRAINING_REGION["max_longitude"]
    )


def predict_next_day_fire_area(
    current_fire_area_hectares: float,
    weather: Mapping[str, float],
    latitude: float,
    longitude: float,
) -> float:
    """Predict next recorded day's daily fire-area growth in hectares."""
    if not math.isfinite(current_fire_area_hectares) or current_fire_area_hectares < 0:
        raise ValueError("Current fire area must be a finite non-negative number.")
    if not is_in_training_region(latitude, longitude):
        raise ValueError("The area model only supports locations within its Canadian training region.")

    features = {
        "firearea": current_fire_area_hectares,
        "tmax": weather["tmax"],
        "vpd": weather["vpd"],
        "ws": weather["ws"],
        "rh": weather["rh"],
        "prec": weather["prec"],
        "lat": latitude,
        "lon": longitude,
    }
    if not all(math.isfinite(value) for value in features.values()):
        raise ValueError("Forecast inputs must all be finite numbers.")

    prediction = float(
        load_area_forecast_model().predict(pd.DataFrame([features], columns=FEATURES))[0]
    )
    if not math.isfinite(prediction):
        raise ValueError("The area forecast model returned a non-finite result.")
    return max(0.0, math.expm1(prediction))
