from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline

from src.prediction.area_forecast import FEATURES, MODEL_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "spread" / "next_day_spread.csv"
TARGET = "next_day_firearea"


def train_area_forecast_model(
    data_path: Path = DATA_PATH, model_path: Path = MODEL_PATH
) -> dict[str, float]:
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Training data not found at {data_path}. Run "
            "python -m src.prediction.prepare_spread_data and "
            "python -m src.prediction.build_spread_targets first."
        )

    frame = pd.read_csv(data_path, usecols=[*FEATURES, "ID", TARGET])
    frame = frame.dropna(subset=["ID", TARGET])
    if frame.empty:
        raise ValueError("Training data has no rows with a next-day area target.")
    if (frame[TARGET] < 0).any():
        raise ValueError("Training data contains negative next-day fire areas.")

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_indices, test_indices = next(
        splitter.split(frame, groups=frame["ID"])
    )
    train = frame.iloc[train_indices]
    test = frame.iloc[test_indices]

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.08,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            random_state=42,
        ),
    )
    model.fit(train.loc[:, FEATURES], np.log1p(train[TARGET]))

    predicted_area = np.maximum(
        0.0, np.expm1(model.predict(test.loc[:, FEATURES]))
    )
    metrics = {
        "rows": float(len(frame)),
        "test_mae_hectares": float(
            mean_absolute_error(test[TARGET], predicted_area)
        ),
        "test_median_absolute_error_hectares": float(
            median_absolute_error(test[TARGET], predicted_area)
        ),
        "test_r2": float(r2_score(test[TARGET], predicted_area)),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return metrics


def main() -> None:
    metrics = train_area_forecast_model()
    print(f"Saved area forecast model to {MODEL_PATH}")
    print(f"Training rows: {int(metrics['rows'])}")
    print(f"Held-out MAE: {metrics['test_mae_hectares']:.1f} ha")
    print(
        "Held-out median absolute error: "
        f"{metrics['test_median_absolute_error_hectares']:.1f} ha"
    )
    print(f"Held-out R²: {metrics['test_r2']:.3f}")


if __name__ == "__main__":
    main()
