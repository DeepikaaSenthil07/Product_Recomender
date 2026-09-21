"""
test_recommender.py
Basic unit tests for the ProductRecommender class.

Run with: pytest tests/
"""

import sys
import os
import pytest
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.recommender import ProductRecommender


@pytest.fixture
def sample_products():
    return pd.DataFrame({
        "id": [1, 2, 3],
        "name": ["Wireless Earbuds", "Wired Earphones", "Gaming Laptop"],
        "category": ["Audio", "Audio", "Laptop"],
        "description": [
            "bluetooth wireless earbuds with noise cancellation",
            "wired earphones with tangle free cable",
            "high performance gaming laptop with graphics card",
        ],
    })


def test_recommender_initializes(sample_products):
    rec = ProductRecommender(sample_products)
    assert rec._tfidf_matrix is not None
    assert rec._similarity_matrix.shape == (3, 3)


def test_recommend_excludes_self(sample_products):
    rec = ProductRecommender(sample_products)
    results = rec.recommend(product_id=1, top_n=2)
    assert 1 not in results["id"].values


def test_recommend_similar_category_ranks_higher(sample_products):
    rec = ProductRecommender(sample_products)
    results = rec.recommend(product_id=1, top_n=2)
    # Product 2 (Audio) should rank above product 3 (Laptop) for product 1 (Audio)
    top_result_id = results.iloc[0]["id"]
    assert top_result_id == 2


def test_recommend_invalid_id_raises(sample_products):
    rec = ProductRecommender(sample_products)
    with pytest.raises(ValueError):
        rec.recommend(product_id=999)


def test_recommend_by_text(sample_products):
    rec = ProductRecommender(sample_products)
    results = rec.recommend_by_text("bluetooth audio", top_n=1)
    assert len(results) == 1
    assert results.iloc[0]["id"] == 1


def test_similarity_scores_are_sorted_descending(sample_products):
    rec = ProductRecommender(sample_products)
    results = rec.recommend(product_id=1, top_n=2)
    scores = results["similarity_score"].tolist()
    assert scores == sorted(scores, reverse=True)
