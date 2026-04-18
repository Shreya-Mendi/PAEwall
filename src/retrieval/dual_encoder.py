"""
Dual-encoder retrieval model for patent-to-product matching.

Wraps the trained DualEncoder model (from scripts/train_deep_learning.py)
with a FAISS index for fast inference-time retrieval.
"""

from pathlib import Path

import numpy as np


class RetrievalEngine:
    """Production retrieval engine using the trained dual-encoder + FAISS index."""

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.model = None
        self.index = None
        self.product_ids = None

    def load(self):
        """Load trained model and FAISS index."""
        # TODO: Load DualEncoder checkpoint
        # TODO: Load pre-built FAISS index over product corpus
        # TODO: Load product ID mapping
        raise NotImplementedError

    def encode_patent(self, claim_text: str, figure_path: str | None = None) -> np.ndarray:
        """Encode a patent into the shared embedding space."""
        # TODO: Tokenize claims, load figure if provided, run patent tower
        raise NotImplementedError

    def retrieve(self, claim_text: str, figure_path: str | None = None, top_k: int = 50) -> list[dict]:
        """
        Retrieve top-K candidate infringing products for a patent.

        Returns:
            List of dicts with 'product_id', 'company', 'score', 'description'.
        """
        # TODO: Encode patent, query FAISS index, return ranked results
        raise NotImplementedError

    def build_index(self, product_corpus: list[dict]):
        """Build FAISS index over product corpus embeddings."""
        # TODO: Encode all products with product tower
        # TODO: Build FAISS IndexFlatIP or IndexIVFFlat
        raise NotImplementedError
