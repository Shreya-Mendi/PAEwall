"""
Naive baseline: BM25 keyword retrieval.

Given a patent's claims text, ranks candidate products by BM25 score against
their business descriptions. No training required — BM25 is purely index-based.

Model artifacts saved to models/naive/.

Usage:
    python scripts/train_naive.py
    python scripts/train_naive.py --eval-only
"""

import argparse
import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from rank_bm25 import BM25Okapi

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import MODELS_DIR, OUTPUTS_DIR, PROCESSED_DIR

NAIVE_MODEL_DIR = MODELS_DIR / "naive"


def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


class BM25Retriever:
    """BM25-based patent-to-product retrieval baseline."""

    def __init__(self):
        self.bm25: BM25Okapi | None = None
        self.product_ids: list[str] = []
        self.product_names: list[str] = []

    def fit(self, products: pd.DataFrame):
        """
        Build BM25 index over product descriptions.

        Args:
            products: DataFrame with columns 'company_name' and 'product_description'.
        """
        corpus = products["product_description"].fillna("").tolist()
        tokenized = [tokenize(doc) for doc in corpus]
        self.bm25 = BM25Okapi(tokenized)
        self.product_ids = products.index.tolist()
        self.product_names = products["company_name"].tolist()
        logger.info(f"BM25 index built over {len(corpus)} products")

    def predict(self, patent_claims: str, top_k: int = 50) -> list[dict]:
        """
        Rank products by BM25 score against patent claims.

        Returns:
            List of dicts with rank, company_name, score — sorted descending.
        """
        if self.bm25 is None:
            raise RuntimeError("Call fit() before predict()")

        query_tokens = tokenize(patent_claims)
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            {
                "rank": rank + 1,
                "product_id": self.product_ids[i],
                "company_name": self.product_names[i],
                "score": float(scores[i]),
            }
            for rank, i in enumerate(top_indices)
        ]

    def save(self, path: Path):
        """Pickle the fitted index."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Saved BM25 model to {path}")

    @classmethod
    def load(cls, path: Path) -> "BM25Retriever":
        """Load a saved BM25 index."""
        with open(path, "rb") as f:
            return pickle.load(f)


def compute_metrics(
    retriever: BM25Retriever,
    benchmark: pd.DataFrame,
    product_corpus: pd.DataFrame,
    top_k: int = 50,
) -> dict:
    """
    Evaluate BM25 on PAE-Bench.

    Args:
        retriever: Fitted BM25Retriever.
        benchmark: PAE-Bench DataFrame with patent_id, company_name, patent_claims.
        product_corpus: All unique companies with product_description.
        top_k: Max candidates to retrieve.

    Returns:
        Dict with Recall@10, Recall@50, MRR, nDCG@10.
    """
    recall_10_scores = []
    recall_50_scores = []
    mrr_scores = []
    ndcg_10_scores = []

    for patent_id, group in benchmark.groupby("patent_id"):
        claims = group["patent_claims"].iloc[0]
        relevant_companies = set(group["company_name"].str.lower().tolist())

        results = retriever.predict(claims, top_k=top_k)
        retrieved_companies = [r["company_name"].lower() for r in results]

        # Recall@K
        retrieved_10 = set(retrieved_companies[:10])
        retrieved_50 = set(retrieved_companies[:50])
        recall_10_scores.append(len(retrieved_10 & relevant_companies) / max(len(relevant_companies), 1))
        recall_50_scores.append(len(retrieved_50 & relevant_companies) / max(len(relevant_companies), 1))

        # MRR
        mrr = 0.0
        for rank, company in enumerate(retrieved_companies, start=1):
            if company in relevant_companies:
                mrr = 1.0 / rank
                break
        mrr_scores.append(mrr)

        # nDCG@10
        dcg = sum(
            1.0 / np.log2(rank + 2)
            for rank, company in enumerate(retrieved_companies[:10])
            if company in relevant_companies
        )
        ideal_hits = min(len(relevant_companies), 10)
        ideal_dcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
        ndcg_10_scores.append(dcg / max(ideal_dcg, 1e-10))

    return {
        "Recall@10": float(np.mean(recall_10_scores)),
        "Recall@50": float(np.mean(recall_50_scores)),
        "MRR": float(np.mean(mrr_scores)),
        "nDCG@10": float(np.mean(ndcg_10_scores)),
        "n_queries": len(recall_10_scores),
    }


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate BM25 naive baseline")
    parser.add_argument("--eval-only", action="store_true", help="Skip fitting, load saved model")
    args = parser.parse_args()

    NAIVE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = NAIVE_MODEL_DIR / "bm25.pkl"
    bench_path = PROCESSED_DIR / "pae_bench.parquet"

    if not bench_path.exists():
        logger.error(f"PAE-Bench not found at {bench_path}. Run make_dataset.py --assemble first.")
        return

    benchmark = pd.read_parquet(bench_path)
    logger.info(f"Loaded PAE-Bench: {len(benchmark)} rows, {benchmark['patent_id'].nunique()} patents")

    # Build product corpus — unique companies with descriptions
    product_corpus = (
        benchmark[["company_name", "product_description"]]
        .drop_duplicates(subset="company_name")
        .reset_index(drop=True)
    )
    logger.info(f"Product corpus: {len(product_corpus)} unique companies")

    if args.eval_only:
        if not model_path.exists():
            logger.error(f"No saved model at {model_path}. Run without --eval-only first.")
            return
        retriever = BM25Retriever.load(model_path)
        logger.info("Loaded saved BM25 model")
    else:
        retriever = BM25Retriever()
        retriever.fit(product_corpus)
        retriever.save(model_path)

    # Evaluate overall
    logger.info("Evaluating on full PAE-Bench...")
    metrics = compute_metrics(retriever, benchmark, product_corpus)

    logger.info("=" * 50)
    logger.info("BM25 Naive Baseline — Overall Results")
    logger.info("=" * 50)
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Evaluate per vertical
    per_vertical = {}
    for vertical in benchmark["vertical"].unique():
        subset = benchmark[benchmark["vertical"] == vertical]
        if len(subset) < 5:
            continue
        v_metrics = compute_metrics(retriever, subset, product_corpus)
        per_vertical[vertical] = v_metrics
        logger.info(f"  [{vertical}] Recall@10: {v_metrics['Recall@10']:.4f}  MRR: {v_metrics['MRR']:.4f}")

    # Save results
    results = {"overall": metrics, "per_vertical": per_vertical}
    out_path = OUTPUTS_DIR / "naive_bm25_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
