
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "spread"
    / "splits"
)

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

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

TARGET = "spread_occurred"


def load_data():
    print("Loading only required columns...")

    feature_columns = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    required_columns = feature_columns + [TARGET]

    train_df = pd.read_csv(
        DATA_DIR / "train.csv",
        usecols=required_columns
    )

    val_df = pd.read_csv(
        DATA_DIR / "validation.csv",
        usecols=required_columns
    )

    test_df = pd.read_csv(
        DATA_DIR / "test.csv",
        usecols=required_columns
    )

    return train_df, val_df, test_df


def build_pipeline():
    numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(
            strategy="constant",
            fill_value="Unknown"
        )),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("numerical", numerical_pipeline, NUMERICAL_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    classifier = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=0.0001,
        max_iter=1000,
        tol=0.001,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    )


    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


def main():
    feature_columns = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

    print("\nSTEP 1: Loading training data...")

    train_df = pd.read_csv(
        DATA_DIR / "train.csv",
        usecols=feature_columns + [TARGET]
    )

    X_train = train_df[feature_columns]
    y_train = train_df[TARGET]

    print("Training records:", len(train_df))

    print("\nSTEP 2: Building SGD pipeline...")

    model = build_pipeline()

    print("\nSTEP 3: Training SGD classifier...")

    model.fit(X_train, y_train)

    del X_train, y_train, train_df

    print("\nTraining completed!")

    print("\nSTEP 4: Loading validation data...")

    val_df = pd.read_csv(
        DATA_DIR / "validation.csv",
        usecols=feature_columns + [TARGET]
    )

    X_val = val_df[feature_columns]
    y_val = val_df[TARGET]

    val_predictions = model.predict(X_val)
    val_probabilities = model.predict_proba(X_val)[:, 1]

    print("\nVALIDATION RESULTS")
    print("------------------")

    print(classification_report(
        y_val,
        val_predictions,
        target_names=[
            "No significant growth",
            "Significant growth",
        ],
        zero_division=0,
    ))

    print("Validation accuracy:",
          round(accuracy_score(y_val, val_predictions), 4))

    print("Validation balanced accuracy:",
          round(balanced_accuracy_score(y_val, val_predictions), 4))

    print("Validation PR-AUC:",
          round(average_precision_score(y_val, val_probabilities), 4))

    print("Validation ROC-AUC:",
          round(roc_auc_score(y_val, val_probabilities), 4))

    del X_val, y_val, val_df
    del val_predictions, val_probabilities

    print("\nSTEP 5: Loading test data...")

    test_df = pd.read_csv(
        DATA_DIR / "test.csv",
        usecols=feature_columns + [TARGET]
    )

    X_test = test_df[feature_columns]
    y_test = test_df[TARGET]

    test_predictions = model.predict(X_test)
    test_probabilities = model.predict_proba(X_test)[:, 1]

    print("\nTEST RESULTS")
    print("------------")

    print(classification_report(
        y_test,
        test_predictions,
        target_names=[
            "No significant growth",
            "Significant growth",
        ],
        zero_division=0,
    ))

    print("Test accuracy:",
          round(accuracy_score(y_test, test_predictions), 4))

    print("Test balanced accuracy:",
          round(balanced_accuracy_score(y_test, test_predictions), 4))

    print("Test PR-AUC:",
          round(average_precision_score(y_test, test_probabilities), 4))

    print("Test ROC-AUC:",
          round(roc_auc_score(y_test, test_probabilities), 4))

    print("\nTest confusion matrix:")
    print(confusion_matrix(y_test, test_predictions))

    model_path = MODEL_DIR / "spread_sgd_classifier.pkl"

    joblib.dump(model, model_path)

    print("\nModel saved to:")
    print(model_path)


if __name__ == "__main__":
    main()