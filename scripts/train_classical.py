"""
Classical ML model: TF-IDF + Logistic Regression retrieval scoring.

Encodes patents and products with TF-IDF vectors, then trains a logistic
regression classifier on (patent, product) pairs to predict infringement
likelihood. Re-ranks candidates retrieved by the naive baseline.

Model artifacts saved to models/.
"""

from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


class ClassicalRetriever:
    """TF-IDF + Logistic Regression retrieval scoring model."""

    def __init__(self, max_features: int = 50000):
        self.patent_vectorizer = TfidfVectorizer(max_features=max_features)
        self.product_vectorizer = TfidfVectorizer(max_features=max_features)
        self.classifier = LogisticRegression(max_iter=1000, class_weight="balanced")

    def fit(self, patent_texts: list[str], product_texts: list[str], labels: list[int]):
        """
        Train the retrieval scoring model.

        Args:
            patent_texts: Patent claim texts.
            product_texts: Corresponding product description texts.
            labels: Binary labels (1 = infringement, 0 = no infringement).
        """
        import numpy as np
        from scipy.sparse import hstack

        patent_features = self.patent_vectorizer.fit_transform(patent_texts)
        product_features = self.product_vectorizer.fit_transform(product_texts)
        combined = hstack([patent_features, product_features])

        self.classifier.fit(combined, labels)

    def predict_score(self, patent_text: str, product_text: str) -> float:
        """Return infringement probability for a (patent, product) pair."""
        from scipy.sparse import hstack

        patent_feat = self.patent_vectorizer.transform([patent_text])
        product_feat = self.product_vectorizer.transform([product_text])
        combined = hstack([patent_feat, product_feat])

        return float(self.classifier.predict_proba(combined)[0, 1])

    def rank_products(
        self, patent_text: str, product_corpus: list[dict], top_k: int = 50
    ) -> list[tuple[str, float]]:
        """
        Rank products by infringement probability.

        Args:
            patent_text: Patent claims text.
            product_corpus: List of dicts with 'id' and 'text' keys.
            top_k: Number of results to return.

        Returns:
            List of (product_id, score) sorted descending.
        """
        scores = [
            (doc["id"], self.predict_score(patent_text, doc["text"]))
            for doc in product_corpus
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


def main():
    """Train and save the classical ML model."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Training classical model (TF-IDF + Logistic Regression)...")

    # TODO: Load processed training pairs from data/processed/
    # TODO: Train TF-IDF + LogReg model
    # TODO: Evaluate on PAE-Bench (Recall@10, Recall@50, MRR, nDCG@10)
    # TODO: Save model artifacts (vectorizers + classifier)

    print("Classical model training not yet implemented. See TODOs.")


if __name__ == "__main__":
    main()
