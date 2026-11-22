
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "spread"
    / "filtered_training_data.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "spread"
    / "splits"
)


def main():
    print("Loading filtered wildfire dataset...")

    df = pd.read_csv(INPUT_FILE)

    print("Total records:", len(df))
    print("Unique fires:", df["ID"].nunique())

    # First split: 70% training, 30% temporary
    first_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=42
    )

    train_idx, temporary_idx = next(
        first_split.split(
            df,
            groups=df["ID"]
        )
    )

    train_df = df.iloc[train_idx].copy()
    temporary_df = df.iloc[temporary_idx].copy()

    # Second split: 50% of temporary for validation,
    # 50% for testing (15% each of the original dataset)
    second_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=42
    )

    val_idx, test_idx = next(
        second_split.split(
            temporary_df,
            groups=temporary_df["ID"]
        )
    )

    val_df = temporary_df.iloc[val_idx].copy()
    test_df = temporary_df.iloc[test_idx].copy()

    datasets = {
        "train": train_df,
        "validation": val_df,
        "test": test_df,
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nDATA SPLIT REPORT")
    print("-----------------")

    for name, split_df in datasets.items():
        output_file = OUTPUT_DIR / f"{name}.csv"

        split_df.to_csv(
            output_file,
            index=False
        )

        print(f"\n{name.upper()}")
        print("Records:", len(split_df))
        print("Unique fires:", split_df["ID"].nunique())

        print("Target distribution:")
        print(
            split_df["spread_occurred"]
            .value_counts()
            .sort_index()
            .to_dict()
        )

        print("Saved:", output_file)

    # Verify no fire appears in multiple datasets
    train_ids = set(train_df["ID"])
    val_ids = set(val_df["ID"])
    test_ids = set(test_df["ID"])

    print("\nGROUP LEAKAGE CHECK")
    print("-------------------")
    print("Train/validation overlap:", len(train_ids & val_ids))
    print("Train/test overlap:", len(train_ids & test_ids))
    print("Validation/test overlap:", len(val_ids & test_ids))


if __name__ == "__main__":
    main()
