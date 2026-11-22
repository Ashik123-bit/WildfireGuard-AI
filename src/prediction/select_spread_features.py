from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "spread"
    / "next_day_spread.csv"
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

EXCLUDED_FEATURES = [
    "ID",
    "DOB",
    "year",
    "fireday",
    "sprdistm",
    "next_day_sprdistm",
    "spread_occurred",
]

UNVERIFIED_FEATURES = [
    "d_fwi", "d_isi", "d_ffmc",
    "d_dmc", "d_dc", "d_bui",
    "d_prec", "d_vpd", "d_tmax",
    "d_ws", "d_rh",
]


def main():
    print("Loading next-day spread dataset...")

    df = pd.read_csv(INPUT_FILE)

    selected_features = (
        NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    )

    missing_columns = [
        col for col in selected_features
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing feature columns: {missing_columns}"
        )

    print("\nFEATURE SELECTION REPORT")
    print("------------------------")
    print("Total records:", len(df))
    print("Unique fires:", df["ID"].nunique())

    print("\nNumerical features:", len(NUMERICAL_FEATURES))
    print("Categorical features:", len(CATEGORICAL_FEATURES))
    print("Unverified features:", len(UNVERIFIED_FEATURES))

    print("\nExcluded columns:")
    print(", ".join(EXCLUDED_FEATURES))

    print("\nUnverified features (not selected yet):")
    print(", ".join(UNVERIFIED_FEATURES))

    print("\nSelected feature missing values:")
    print(
        df[selected_features]
        .isnull()
        .sum()
        .sort_values(ascending=False)
        .to_string()
    )

    print("\nTarget distribution:")
    print(df["spread_occurred"].value_counts().to_string())

    print("\nFeature inspection completed.")


if __name__ == "__main__":
    main()