from pathlib import Path

import pandas as pd
import pytest

from src.data import (
    LABEL_TO_INT,
    SENTIMENT_MAP,
    assign_sentiment_labels,
    balance_classes,
    clean_columns,
    encode_labels,
    load_reviews,
)

FIXTURE = Path(__file__).parent / "fixtures" / "sample_reviews.csv"


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return load_reviews(str(FIXTURE))


@pytest.fixture
def labelled_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    cleaned = clean_columns(raw_df)
    return assign_sentiment_labels(cleaned)


class TestLoadReviews:
    def test_returns_dataframe(self, raw_df):
        assert isinstance(raw_df, pd.DataFrame)
        assert len(raw_df) == 12


class TestCleanColumns:
    def test_lowercases_columns(self, raw_df):
        out = clean_columns(raw_df)
        assert all(c == c.lower() for c in out.columns)

    def test_drops_default_unused_columns(self, raw_df):
        out = clean_columns(raw_df)
        for col in ["reviews.didpurchase", "reviews.id", "reviews.numhelpful", "reviews.dorecommend"]:
            assert col not in out.columns

    def test_does_not_mutate_input(self, raw_df):
        before = list(raw_df.columns)
        _ = clean_columns(raw_df)
        assert list(raw_df.columns) == before

    def test_custom_drop_list(self, raw_df):
        out = clean_columns(raw_df, drop=["categories"])
        assert "categories" not in out.columns
        # Default-drop columns NOT in custom list should remain
        assert "reviews.id" in out.columns


class TestAssignSentimentLabels:
    def test_maps_each_rating(self, labelled_df):
        for rating, expected in SENTIMENT_MAP.items():
            rows = labelled_df[labelled_df["reviews.rating"] == rating]
            if not rows.empty:
                assert (rows["rating_sentiment"] == expected).all()

    def test_no_unmapped_ratings(self, labelled_df):
        # Fixture only contains ratings 1-5 so nothing should be NaN.
        assert labelled_df["rating_sentiment"].notna().all()

    def test_label_set(self, labelled_df):
        assert set(labelled_df["rating_sentiment"].unique()) <= {"negative", "neutral", "positive"}


class TestBalanceClasses:
    def test_equal_class_sizes_when_n_specified(self, labelled_df):
        out = balance_classes(labelled_df, n_per_class=2, seed=0)
        counts = out["rating_sentiment"].value_counts()
        assert (counts == 2).all()
        assert len(out) == len(counts) * 2

    def test_default_n_uses_smallest_class(self, labelled_df):
        # Fixture: 6 positive, 4 negative, 2 neutral -> default should be 2 each.
        out = balance_classes(labelled_df, seed=0)
        counts = out["rating_sentiment"].value_counts()
        assert (counts == counts.min()).all()
        assert counts.min() == labelled_df["rating_sentiment"].value_counts().min()

    def test_seed_reproducibility(self, labelled_df):
        a = balance_classes(labelled_df, n_per_class=2, seed=123)
        b = balance_classes(labelled_df, n_per_class=2, seed=123)
        pd.testing.assert_frame_equal(a, b)

    def test_smaller_class_kept_intact(self, labelled_df):
        # When n_per_class >= class size, that class is kept whole (not upsampled).
        neutral_count = (labelled_df["rating_sentiment"] == "neutral").sum()
        out = balance_classes(labelled_df, n_per_class=10, seed=0)
        assert (out["rating_sentiment"] == "neutral").sum() == neutral_count


class TestEncodeLabels:
    def test_adds_integer_column(self, labelled_df):
        out = encode_labels(labelled_df)
        assert "actual_sentiment" in out.columns
        assert set(out["actual_sentiment"].unique()) <= set(LABEL_TO_INT.values())

    def test_mapping_matches(self, labelled_df):
        out = encode_labels(labelled_df)
        for sentiment, expected_int in LABEL_TO_INT.items():
            rows = out[out["rating_sentiment"] == sentiment]
            assert (rows["actual_sentiment"] == expected_int).all()
