"""Inference helpers: baseline sentiment, LLM category classification, blog generation.

These functions wrap external models (Hugging Face pipelines, Ollama). Imports of
heavy dependencies (``transformers``, ``ollama``) happen inside the functions so
``import src.predict`` stays cheap and the module can be inspected without a model
runtime available.
"""

from __future__ import annotations

import pandas as pd

from .data import META_CATEGORIES

CARDIFF_LABEL_MAP: dict[str, str] = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
}


def baseline_sentiment(
    reviews: list[str],
    model_id: str = "cardiffnlp/twitter-roberta-base-sentiment",
    max_length: int = 512,
) -> list[str]:
    """Run zero-shot sentiment classification with the CardiffNLP RoBERTa baseline."""
    from transformers import pipeline

    classifier = pipeline(task="sentiment-analysis", model=model_id)
    raw = classifier(reviews, truncation=True, max_length=max_length)
    return [CARDIFF_LABEL_MAP[r["label"]] for r in raw]


CATEGORY_PROMPT = """Classify this product category: '{cat}' into ONE of:
{options}.
Reply with ONLY the category name. DO NOT invent your own category name.
Use ONLY the names provided.
If the product category contains 'Health & Beauty' or 'Health and Beauty',
the answer is: Health & Beauty.
Litter boxes are classified as Pet Supplies.
"""


def classify_categories(
    raw_categories: list[str],
    options: list[str] | None = None,
    model: str = "qwen2.5",
) -> dict[str, str]:
    """Use a local Ollama model to map each raw category string to a meta-category.

    Returns a ``{raw_category: meta_category}`` dict.
    """
    import ollama

    options = options or META_CATEGORIES
    options_str = ", ".join(f"'{o}'" for o in options)
    mapping: dict[str, str] = {}
    for cat in raw_categories:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "user", "content": CATEGORY_PROMPT.format(cat=cat, options=options_str)}
            ],
        )
        mapping[cat] = response["message"]["content"].strip()
    return mapping


def generate_category_blog(
    df: pd.DataFrame,
    category: str,
    model: str = "qwen2.5",
    n_top: int = 3,
    max_reviews_per_product: int = 30,
) -> str:
    """Generate a blog-style summary for one meta-category via Ollama."""
    import ollama

    cat_df = df[df["meta_category"] == category]

    top_n = (
        cat_df.groupby(["meta_category", "name"])
        .agg(avg_rating=("reviews.rating", "mean"), num_reviews=("reviews.rating", "count"))
        .sort_values(by="num_reviews", ascending=False)
        .head(n_top)
    )
    top_names = top_n.index.get_level_values("name").tolist()
    top_reviews = [
        df[df["name"] == name]["reviews.text"].tolist()[:max_reviews_per_product]
        for name in top_names
    ]

    worst_product = (
        cat_df.groupby(["meta_category", "name"])
        .agg(avg_rating=("reviews.rating", "mean"), num_reviews=("reviews.rating", "count"))
        .sort_values(by="avg_rating", ascending=True)
        .head(1)
    )
    worst_name = worst_product.index.get_level_values("name")[0]
    worst_reviews = df[df["name"] == worst_name]["reviews.text"].tolist()[:max_reviews_per_product]

    top_reviews_str = "\n\n".join(
        f"Product {i + 1} - {top_names[i]}:\n{top_reviews[i]}" for i in range(len(top_names))
    )

    prompt = f"""
Write a short article (like a blog post) about the product category: {category}.
The output should include:

- Top {n_top} products {top_n} and key differences between them.
- Reviews and top complaints for each: {top_reviews_str}

-----
You should also include the worst product {worst_product} in the category and why
it should be avoided.
Worst product reviews: {worst_reviews}
Do not forget to include this.
----

Make sure the style is like a blog.
"""
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]
