
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "spread"
    / "combined_fire_growth.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "spread"
    / "next_day_spread.csv"
)


def main():
    print("Loading wildfire spread dataset...")

    df = pd.read_csv(INPUT_FILE)

    df = df.sort_values(
        ["ID", "fireday"]
    ).copy()

    # Next recorded day's spread distance
    df["next_day_sprdistm"] = (
        df.groupby("ID")["sprdistm"].shift(-1)
    )

    # Next recorded day's fire growth
    df["next_day_firearea"] = (
        df.groupby("ID")["firearea"].shift(-1)
    )

    # Remove final records without a next-day target
    df = df.dropna(
        subset=["next_day_sprdistm", "next_day_firearea"]
    ).copy()

    # Classification target:
    # 0 = growth below 9.72 hectares
    # 1 = growth of at least 9.72 hectares
    df["spread_occurred"] = (
        df["next_day_firearea"] >= 9.72
    ).astype(int)

    print("\nTARGET DATASET REPORT")
    print("---------------------")

    print("Total records:", len(df))
    print("Unique fires:", df["ID"].nunique())

    print("\nSpread classification:")
    print(
        df["spread_occurred"]
        .value_counts()
        .sort_index()
        .rename(index={0: "No significant growth", 1: "Significant growth"})
    )

    print("\nNext-day spread distance:")
    print(df["next_day_sprdistm"].describe())

    print("\nNext-day fire growth:")
    print(df["next_day_firearea"].describe())

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(OUTPUT_FILE, index=False)

    print("\nSaved dataset to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
