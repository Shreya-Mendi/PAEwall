"""
Naive baseline: BM25 keyword retrieval.

Given a patent's claims text, ranks candidate products by BM25 score.
This is the simplest reasonable baseline for the retrieval task.

Model artifacts saved to models/.
"""

from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


class NaiveBaseline:
    """BM25-based patent-to-product retrieval baseline."""

    def __init__(self):
        self.bm25 = None
        self.corpus_ids = None

    def fit(self, product_corpus: list[dict]):
        """
        Build BM25 index over product descriptions.

        Args:
            product_corpus: List of dicts with 'id' and 'text' keys.
        """
        tokenized = [doc["text"].lower().split() for doc in product_corpus]
        self.bm25 = BM25Okapi(tokenized)
        self.corpus_ids = [doc["id"] for doc in product_corpus]

    def predict(self, patent_claims: str, top_k: int = 50) -> list[tuple[str, float]]:
        """
        Rank products by BM25 score against patent claims.

        Returns:
            List of (product_id, score) tuples, sorted descending.
        """
        if self.bm25 is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        query_tokens = patent_claims.lower().split()
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [(self.corpus_ids[i], float(scores[i])) for i in top_indices]


def main():
    """Train and save the naive BM25 baseline."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Training naive baseline (BM25)...")

    # TODO: Load processed product corpus from data/processed/
    # TODO: Fit BM25 index
    # TODO: Evaluate on PAE-Bench (Recall@10, Recall@50, MRR, nDCG@10)
    # TODO: Save model artifact

    print("Naive baseline training not yet implemented. See TODOs.")


if __name__ == "__main__":
    main()
