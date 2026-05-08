"""Data loading, cleaning, and label preparation for the review pipeline."""

from __future__ import annotations

import pandas as pd

SENTIMENT_MAP: dict[int, str] = {
    1: "negative",
    2: "negative",
    3: "neutral",
    4: "positive",
    5: "positive",
}

LABEL_TO_INT: dict[str, int] = {"negative": 0, "neutral": 1, "positive": 2}
INT_TO_LABEL: dict[int, str] = {v: k for k, v in LABEL_TO_INT.items()}

META_CATEGORIES: list[str] = [
    "Electronics",
    "Tablets & E-readers",
    "Health & Beauty",
    "Home & Kitchen",
    "Office Supplies",
    "Pet Supplies",
]

# Columns that are mostly null in the Datafiniti dataset and not used downstream.
DROP_COLUMNS_DEFAULT: list[str] = [
    "reviews.didpurchase",
    "reviews.id",
    "reviews.numhelpful",
    "reviews.dorecommend",
]


def load_reviews(path: str) -> pd.DataFrame:
    """Load the raw Datafiniti reviews CSV."""
    return pd.read_csv(path)


def clean_columns(df: pd.DataFrame, drop: list[str] | None = None) -> pd.DataFrame:
    """Lowercase column names and drop columns that are heavily null / unused.

    Returns a new DataFrame; the input is not modified.
    """
    drop = drop if drop is not None else DROP_COLUMNS_DEFAULT
    out = df.copy()
    out.columns = out.columns.str.lower()
    return out.drop(columns=[c for c in drop if c in out.columns])


def assign_sentiment_labels(df: pd.DataFrame, rating_col: str = "reviews.rating") -> pd.DataFrame:
    """Map star ratings (1-5) to sentiment labels and add a ``rating_sentiment`` column.

    1-2 -> negative, 3 -> neutral, 4-5 -> positive.
    """
    out = df.copy()
    out["rating_sentiment"] = out[rating_col].map(SENTIMENT_MAP)
    return out


def balance_classes(
    df: pd.DataFrame,
    label_col: str = "rating_sentiment",
    n_per_class: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Downsample so each class in ``label_col`` has equal representation.

    If ``n_per_class`` is None, uses the size of the smallest class. The neutral
    class is typically smallest in this dataset, so it is preserved in full.
    """
    counts = df[label_col].value_counts()
    if n_per_class is None:
        n_per_class = int(counts.min())

    pieces = []
    for cls in counts.index:
        cls_df = df[df[label_col] == cls]
        if len(cls_df) > n_per_class:
            cls_df = cls_df.sample(n=n_per_class, random_state=seed)
        pieces.append(cls_df)
    return pd.concat(pieces).reset_index(drop=True)


def encode_labels(df: pd.DataFrame, label_col: str = "rating_sentiment") -> pd.DataFrame:
    """Add an integer ``actual_sentiment`` column suitable for HF Trainer."""
    out = df.copy()
    out["actual_sentiment"] = out[label_col].map(LABEL_TO_INT)
    return out
