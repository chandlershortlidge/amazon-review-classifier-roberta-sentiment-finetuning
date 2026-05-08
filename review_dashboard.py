"""Pure helpers used by the Streamlit dashboard. Kept import-free of Streamlit
so they can be unit-tested without a running app."""

import pandas as pd


def category_key(category: str) -> str:
    """Normalise a display category name to a dict key,
    e.g. 'Health & Beauty' -> 'health_beauty'."""
    return category.lower().replace(" & ", "_").replace("&", "_").replace(" ", "_")


def get_top_n(df: pd.DataFrame, category: str, n: int = 3) -> pd.DataFrame:
    cat_df = df[df["meta_category"] == category]
    return (
        cat_df.groupby("name")
        .agg(avg_rating=("reviews.rating", "mean"), num_reviews=("reviews.rating", "count"))
        .sort_values("num_reviews", ascending=False)
        .head(n)
        .reset_index()
        .rename(columns={"name": "Product", "avg_rating": "Avg Rating", "num_reviews": "# Reviews"})
    )


def get_worst(df: pd.DataFrame, category: str) -> pd.DataFrame:
    cat_df = df[df["meta_category"] == category]
    return (
        cat_df.groupby("name")
        .agg(avg_rating=("reviews.rating", "mean"), num_reviews=("reviews.rating", "count"))
        .sort_values("avg_rating", ascending=True)
        .head(1)
        .reset_index()
        .rename(columns={"name": "Product", "avg_rating": "Avg Rating", "num_reviews": "# Reviews"})
    )
