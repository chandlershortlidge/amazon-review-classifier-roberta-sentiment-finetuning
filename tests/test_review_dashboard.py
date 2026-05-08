import pandas as pd
import pytest

from review_dashboard import category_key, get_top_n, get_worst


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "meta_category": [
                "Electronics", "Electronics", "Electronics",
                "Electronics", "Electronics",
                "Electronics", "Electronics",
                "Pet Supplies", "Pet Supplies",
            ],
            "name": [
                "Echo Dot", "Echo Dot", "Echo Dot",
                "Fire TV Stick", "Fire TV Stick",
                "Bad Cable", "Bad Cable",
                "Dog Crate", "Cat Litter Box",
            ],
            "reviews.rating": [5, 5, 4, 5, 4, 1, 2, 5, 3],
        }
    )


class TestCategoryKey:
    def test_ampersand_with_spaces(self):
        assert category_key("Health & Beauty") == "health_beauty"

    def test_ampersand_no_spaces(self):
        assert category_key("Foo&Bar") == "foo_bar"

    def test_plain_spaces(self):
        assert category_key("Pet Supplies") == "pet_supplies"

    def test_single_word(self):
        assert category_key("Electronics") == "electronics"

    def test_lowercases(self):
        assert category_key("ELECTRONICS") == "electronics"


class TestGetTopN:
    def test_orders_by_review_count_desc(self, sample_df):
        top = get_top_n(sample_df, "Electronics", n=3)
        # First row must be the most-reviewed; the two 2-review products tie
        # and pandas' tiebreaker is implementation-defined, so don't assert order.
        assert top["Product"].iloc[0] == "Echo Dot"
        assert list(top["# Reviews"]) == [3, 2, 2]
        assert set(top["Product"]) == {"Echo Dot", "Fire TV Stick", "Bad Cable"}

    def test_filters_by_category(self, sample_df):
        top = get_top_n(sample_df, "Pet Supplies", n=5)
        assert set(top["Product"]) == {"Dog Crate", "Cat Litter Box"}
        assert len(top) == 2

    def test_respects_n(self, sample_df):
        top = get_top_n(sample_df, "Electronics", n=1)
        assert len(top) == 1
        assert top["Product"].iloc[0] == "Echo Dot"

    def test_avg_rating_correct(self, sample_df):
        top = get_top_n(sample_df, "Electronics", n=3)
        echo = top[top["Product"] == "Echo Dot"].iloc[0]
        assert echo["Avg Rating"] == pytest.approx((5 + 5 + 4) / 3)

    def test_renamed_columns(self, sample_df):
        top = get_top_n(sample_df, "Electronics")
        assert list(top.columns) == ["Product", "Avg Rating", "# Reviews"]


class TestGetWorst:
    def test_returns_lowest_avg_rating(self, sample_df):
        worst = get_worst(sample_df, "Electronics")
        assert len(worst) == 1
        assert worst["Product"].iloc[0] == "Bad Cable"
        assert worst["Avg Rating"].iloc[0] == pytest.approx(1.5)

    def test_filters_by_category(self, sample_df):
        worst = get_worst(sample_df, "Pet Supplies")
        assert worst["Product"].iloc[0] == "Cat Litter Box"
        assert worst["Avg Rating"].iloc[0] == pytest.approx(3.0)

    def test_renamed_columns(self, sample_df):
        worst = get_worst(sample_df, "Electronics")
        assert list(worst.columns) == ["Product", "Avg Rating", "# Reviews"]
