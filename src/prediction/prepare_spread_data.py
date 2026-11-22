
from pathlib import Path

import pandas as pd


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "spread"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "spread"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_spread_data():
    """Load and combine all annual Canadian fire-growth CSV files."""

    csv_files = sorted(DATA_DIR.glob("Firegrowth_groups_v1_1_*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No annual CSV files found in {DATA_DIR}"
        )

    dataframes = []

    for file in csv_files:
        print(f"Loading: {file.name}")

        df = pd.read_csv(file)
        dataframes.append(df)

    data = pd.concat(dataframes, ignore_index=True)

    print("\nDataset loaded successfully!")
    print("Files loaded:", len(csv_files))
    print("Total records:", len(data))
    print("Total columns:", len(data.columns))

    return data


def validate_spread_data(data):
    """Perform basic data-quality checks."""

    print("\n--- DATA QUALITY REPORT ---")

    print("\nMissing values:")
    print(data.isnull().sum().sort_values(ascending=False).head(10))

    print("\nDuplicate ID/fireday records:")
    duplicates = data.duplicated(["ID", "fireday"]).sum()
    print(duplicates)

    print("\nUnique fires:", data["ID"].nunique())

    print("\nSpread distance statistics:")
    print(data["sprdistm"].describe())

    print("\nFire growth statistics:")
    print(data["firearea"].describe())

    if duplicates > 0:
        raise ValueError("Duplicate ID/fireday records detected!")

    if (data["sprdistm"] < 0).any():
        raise ValueError("Negative spread distances detected!")

    if (data["firearea"] < 0).any():
        raise ValueError("Negative fire growth values detected!")


def main():
    data = load_spread_data()

    validate_spread_data(data)

    # Save combined dataset
    output_file = OUTPUT_DIR / "combined_fire_growth.csv"

    data.to_csv(output_file, index=False)

    print("\nCombined dataset saved to:")
    print(output_file)


if __name__ == "__main__":
    main()
