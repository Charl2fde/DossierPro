"""Train/validation/test split for the harmonized, preprocessed reviews dataset.

Stratifies on `sentiment` (the modeling target: negative/neutral/positive,
derived from `rating` in `preprocess.py`) to keep the same class imbalance
proportionally represented in all three splits, so validation/test metrics
stay representative of the full distribution.
"""

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

PROCESSED_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
INPUT_PATH = PROCESSED_DATA_DIR / "reviews_nlp.csv"
RANDOM_STATE = 42
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15


def split_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the dataset into stratified train/val/test sets.

    Returns:
        The (train, val, test) DataFrames.
    """
    df = pd.read_csv(INPUT_PATH, low_memory=False)

    train_df, rest_df = train_test_split(
        df, train_size=TRAIN_SIZE, stratify=df["sentiment"], random_state=RANDOM_STATE
    )
    val_df, test_df = train_test_split(
        rest_df,
        train_size=VAL_SIZE / (VAL_SIZE + TEST_SIZE),
        stratify=rest_df["sentiment"],
        random_state=RANDOM_STATE,
    )
    return train_df, val_df, test_df


def main() -> None:
    """Build the splits and save them to data/processed/."""
    train_df, val_df, test_df = split_dataset()

    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        path = PROCESSED_DATA_DIR / f"{name}.csv"
        split_df.to_csv(path, index=False, encoding="utf-8")
        logger.info("%s: %d rows -> %s", name, len(split_df), path)
        logger.info(
            "  sentiment distribution: %s",
            split_df["sentiment"]
            .value_counts(normalize=True)
            .sort_index()
            .round(3)
            .to_dict(),
        )


if __name__ == "__main__":
    main()
