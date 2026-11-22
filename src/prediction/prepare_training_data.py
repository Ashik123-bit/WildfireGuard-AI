
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

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "spread"
    / "filtered_training_data.csv"
)

GROWTH_THRESHOLD = 9.72


def main():
    print("Loading next-day spread dataset...")

    df = pd.read_csv(INPUT_FILE)

    print("Original records:", len(df))

    # Keep records with significant current-day fire growth
    filtered_df = df[
        df["firearea"] >= GROWTH_THRESHOLD
    ].copy()

    print("\nFILTERED DATASET REPORT")
    print("-----------------------")

    print("Filtered records:", len(filtered_df))
    print("Unique fires:", filtered_df["ID"].nunique())

    print("\nTarget distribution:")
    print(
        filtered_df["spread_occurred"]
        .value_counts()
        .rename(
            index={
                0: "No significant next-day growth",
                1: "Significant next-day growth",
            }
        )
        .to_string()
    )

    # Save the separate experimental dataset
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    filtered_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nFiltered dataset saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()

