"""
recommender.py
Content-based product recommender using TF-IDF vectorization and
cosine similarity over product descriptions and categories.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ProductRecommender:
    """
    A content-based recommendation engine.

    Combines each product's category and description into a single text
    field, vectorizes it with TF-IDF, and recommends similar products
    using cosine similarity between vectors.
    """

    def __init__(self, products: pd.DataFrame):
        self.products = products.reset_index(drop=True)
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._tfidf_matrix = None
        self._similarity_matrix = None
        self._id_to_index = {}
        self._fit()

    def _build_corpus(self) -> pd.Series:
        """Combine category + description into one text field per product."""
        return (
            self.products["category"].astype(str)
            + " "
            + self.products["description"].astype(str)
        )

    def _fit(self) -> None:
        """Fit the TF-IDF vectorizer and precompute the similarity matrix."""
        corpus = self._build_corpus()
        self._tfidf_matrix = self._vectorizer.fit_transform(corpus)
        self._similarity_matrix = cosine_similarity(self._tfidf_matrix)
        self._id_to_index = {
            pid: idx for idx, pid in enumerate(self.products["id"])
        }

    def recommend(self, product_id: int, top_n: int = 3) -> pd.DataFrame:
        """
        Recommend products similar to a given product ID.

        Args:
            product_id: The ID of the product to base recommendations on.
            top_n: Number of recommendations to return.

        Returns:
            DataFrame of the top_n most similar products, with a
            'similarity_score' column, excluding the input product itself.

        Raises:
            ValueError: If product_id does not exist in the catalog.
        """
        if product_id not in self._id_to_index:
            raise ValueError(f"Product ID {product_id} not found in catalog.")

        idx = self._id_to_index[product_id]
        scores = list(enumerate(self._similarity_matrix[idx]))

        # Exclude the product itself, sort by score descending
        scores = [s for s in scores if s[0] != idx]
        scores.sort(key=lambda x: x[1], reverse=True)
        top_matches = scores[:top_n]

        result = self.products.iloc[[i for i, _ in top_matches]].copy()
        result["similarity_score"] = [round(score, 3) for _, score in top_matches]
        return result.reset_index(drop=True)

    def recommend_by_text(self, query: str, top_n: int = 3) -> pd.DataFrame:
        """
        Recommend products based on a free-text query, e.g. a search box.

        Args:
            query: Free-text description of what the user is looking for.
            top_n: Number of recommendations to return.

        Returns:
            DataFrame of the top_n most similar products with scores.
        """
        query_vector = self._vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self._tfidf_matrix).flatten()

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        top_matches = ranked[:top_n]

        result = self.products.iloc[[i for i, _ in top_matches]].copy()
        result["similarity_score"] = [round(score, 3) for _, score in top_matches]
        return result.reset_index(drop=True)
