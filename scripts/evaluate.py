"""
Unified evaluation script for all three models on PAE-Bench.

Computes: Recall@10, Recall@50, MRR, nDCG@10 for retrieval.
Reports per-vertical sub-splits and overall cross-vertical numbers.
Generates comparison tables and visualizations for the report.
"""

from pathlib import Path

import numpy as np

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "data" / "outputs"


def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Compute Recall@K."""
    retrieved = set(ranked_ids[:k])
    return len(retrieved & relevant_ids) / max(len(relevant_ids), 1)


def mrr(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    """Compute Mean Reciprocal Rank."""
    for i, pid in enumerate(ranked_ids):
        if pid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Compute nDCG@K with binary relevance."""
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, pid in enumerate(ranked_ids[:k])
        if pid in relevant_ids
    )
    ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_ids), k)))
    return dcg / max(ideal_dcg, 1e-10)


def evaluate_retrieval(model, benchmark: list[dict], top_k: int = 50) -> dict:
    """
    Evaluate a retrieval model on PAE-Bench.

    Args:
        model: Any model with a .predict() or .rank_products() method
               returning list of (product_id, score).
        benchmark: List of dicts with 'patent_text', 'relevant_ids', 'vertical'.

    Returns:
        Dict of metric name -> value, overall and per-vertical.
    """
    # TODO: Implement evaluation loop
    # TODO: Compute metrics per-query, then average
    # TODO: Break down by vertical sub-splits
    raise NotImplementedError


def main():
    """Run evaluation across all models and generate comparison report."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Running evaluation on PAE-Bench...")

    # TODO: Load PAE-Bench from data/processed/
    # TODO: Load all three trained models
    # TODO: Evaluate each model
    # TODO: Generate comparison table (data/outputs/model_comparison.csv)
    # TODO: Generate visualizations (data/outputs/figures/)

    print("Evaluation not yet implemented. See TODOs.")


if __name__ == "__main__":
    main()
