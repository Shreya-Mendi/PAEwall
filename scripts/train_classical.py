"""
Classical ML model: TF-IDF + Logistic Regression retrieval scoring.

Encodes (patent_claims, product_description) pairs with TF-IDF vectors,
trains a logistic regression binary classifier to predict infringement,
and ranks candidate products by predicted probability.

Model artifacts saved to models/classical/.

Usage:
    python scripts/train_classical.py
    python scripts/train_classical.py --eval-only
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import MODELS_DIR, OUTPUTS_DIR, PROCESSED_DIR

CLASSICAL_MODEL_DIR = MODELS_DIR / "classical"


class ClassicalRetriever:
    """
    TF-IDF + Logistic Regression patent-to-product retrieval model.

    Trains on (patent_claims, product_description, label) triples.
    At inference time, scores all candidate products for a given patent
    and returns a ranked list.
    """

    def __init__(self, max_features: int = 30000, ngram_range: tuple = (1, 2)):
        self.patent_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"\b[a-zA-Z][a-zA-Z0-9]{1,}\b",
        )
        self.product_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"\b[a-zA-Z][a-zA-Z0-9]{1,}\b",
        )
        self.scaler = MaxAbsScaler()
        self.classifier = LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight="balanced",
            solver="saga",
            n_jobs=-1,
        )
        self._fitted = False

    def _build_features(self, patent_texts: list[str], product_texts: list[str], fit: bool = False):
        """Vectorize patent and product texts and concatenate features."""
        if fit:
            patent_feats = self.patent_vectorizer.fit_transform(patent_texts)
            product_feats = self.product_vectorizer.fit_transform(product_texts)
        else:
            patent_feats = self.patent_vectorizer.transform(patent_texts)
            product_feats = self.product_vectorizer.transform(product_texts)

        combined = hstack([patent_feats, product_feats])

        if fit:
            combined = self.scaler.fit_transform(combined)
        else:
            combined = self.scaler.transform(combined)

        return combined

    def fit(self, df: pd.DataFrame):
        """
        Train on PAE-Bench training split.

        Args:
            df: DataFrame with columns patent_claims, product_description, label.
                label should be 1 for infringement, 0 for non-infringement.
        """
        patent_texts = df["patent_claims"].fillna("").tolist()
        product_texts = df["product_description"].fillna("").tolist()
        labels = df["label_binary"].tolist()

        logger.info(f"Fitting TF-IDF vectorizers on {len(df)} training pairs...")
        X = self._build_features(patent_texts, product_texts, fit=True)

        logger.info(f"Training LogisticRegression on {X.shape}...")
        self.classifier.fit(X, labels)
        self._fitted = True

        # Training AUC
        proba = self.classifier.predict_proba(X)[:, 1]
        auc = roc_auc_score(labels, proba)
        logger.info(f"Training AUC: {auc:.4f}")

    def predict_score(self, patent_claims: str, product_description: str) -> float:
        """Return infringement probability for a single (patent, product) pair."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        X = self._build_features([patent_claims], [product_description])
        return float(self.classifier.predict_proba(X)[0, 1])

    def rank_products(
        self, patent_claims: str, product_corpus: pd.DataFrame, top_k: int = 50
    ) -> list[dict]:
        """
        Rank all products for a given patent by infringement probability.

        Args:
            patent_claims: Full claims text of the query patent.
            product_corpus: DataFrame with company_name and product_description.
            top_k: Number of top results to return.

        Returns:
            List of dicts with rank, company_name, score — sorted descending.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first.")

        n = len(product_corpus)
        patent_texts = [patent_claims] * n
        product_texts = product_corpus["product_description"].fillna("").tolist()

        X = self._build_features(patent_texts, product_texts)
        scores = self.classifier.predict_proba(X)[:, 1]

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            {
                "rank": rank + 1,
                "product_id": int(product_corpus.index[i]),
                "company_name": product_corpus["company_name"].iloc[i],
                "score": float(scores[i]),
            }
            for rank, i in enumerate(top_indices)
        ]

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Saved classical model to {path}")

    @classmethod
    def load(cls, path: Path) -> "ClassicalRetriever":
        with open(path, "rb") as f:
            return pickle.load(f)


def build_training_pairs(benchmark: pd.DataFrame, neg_ratio: int = 3) -> pd.DataFrame:
    """
    Build (patent, product, label) training pairs.

    Positive pairs: litigated (patent, defendant) pairs from PAE-Bench.
    Negative pairs: random (patent, unrelated_company) pairs sampled at neg_ratio:1.

    Args:
        benchmark: PAE-Bench DataFrame.
        neg_ratio: How many negatives per positive.
    """
    positives = benchmark[["patent_id", "patent_claims", "company_name", "product_description"]].copy()
    positives["label_binary"] = 1

    # Sample negatives: for each patent, pair with random companies that
    # were NOT sued by that patent's plaintiff
    all_companies = benchmark[["company_name", "product_description"]].drop_duplicates("company_name")
    rng = np.random.default_rng(seed=42)

    negatives = []
    for _, pos_row in positives.iterrows():
        patent_id = pos_row["patent_id"]
        sued_companies = set(
            benchmark.loc[benchmark["patent_id"] == patent_id, "company_name"]
        )
        candidates = all_companies[~all_companies["company_name"].isin(sued_companies)]
        if len(candidates) == 0:
            continue
        sampled = candidates.sample(
            n=min(neg_ratio, len(candidates)),
            random_state=int(rng.integers(0, 10000)),
            replace=False,
        )
        for _, neg_company in sampled.iterrows():
            negatives.append({
                "patent_id": patent_id,
                "patent_claims": pos_row["patent_claims"],
                "company_name": neg_company["company_name"],
                "product_description": neg_company["product_description"],
                "label_binary": 0,
            })

    neg_df = pd.DataFrame(negatives)
    combined = pd.concat([positives, neg_df], ignore_index=True).sample(frac=1, random_state=42)
    logger.info(
        f"Training pairs: {len(positives)} positives + {len(neg_df)} negatives = {len(combined)} total"
    )
    return combined


def compute_metrics(
    retriever: ClassicalRetriever,
    benchmark: pd.DataFrame,
    product_corpus: pd.DataFrame,
    top_k: int = 50,
) -> dict:
    """Evaluate on PAE-Bench — same metrics as BM25 baseline for direct comparison."""
    recall_10, recall_50, mrr_scores, ndcg_10 = [], [], [], []

    for patent_id, group in benchmark.groupby("patent_id"):
        claims = group["patent_claims"].iloc[0]
        relevant = set(group["company_name"].str.lower())

        results = retriever.rank_products(claims, product_corpus, top_k=top_k)
        retrieved = [r["company_name"].lower() for r in results]

        retrieved_10 = set(retrieved[:10])
        retrieved_50 = set(retrieved[:50])
        recall_10.append(len(retrieved_10 & relevant) / max(len(relevant), 1))
        recall_50.append(len(retrieved_50 & relevant) / max(len(relevant), 1))

        mrr = 0.0
        for rank, company in enumerate(retrieved, 1):
            if company in relevant:
                mrr = 1.0 / rank
                break
        mrr_scores.append(mrr)

        dcg = sum(
            1.0 / np.log2(rank + 2)
            for rank, company in enumerate(retrieved[:10])
            if company in relevant
        )
        ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), 10)))
        ndcg_10.append(dcg / max(ideal, 1e-10))

    return {
        "Recall@10": float(np.mean(recall_10)),
        "Recall@50": float(np.mean(recall_50)),
        "MRR": float(np.mean(mrr_scores)),
        "nDCG@10": float(np.mean(ndcg_10)),
        "n_queries": len(recall_10),
    }


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate TF-IDF + LogReg classical model")
    parser.add_argument("--eval-only", action="store_true", help="Skip training, load saved model")
    parser.add_argument("--neg-ratio", type=int, default=3, help="Negatives per positive pair")
    args = parser.parse_args()

    CLASSICAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = CLASSICAL_MODEL_DIR / "tfidf_logreg.pkl"
    bench_path = PROCESSED_DIR / "pae_bench.parquet"

    if not bench_path.exists():
        logger.error(f"PAE-Bench not found at {bench_path}. Run make_dataset.py --assemble first.")
        return

    benchmark = pd.read_parquet(bench_path)
    logger.info(f"Loaded PAE-Bench: {len(benchmark)} rows, {benchmark['patent_id'].nunique()} patents")

    product_corpus = (
        benchmark[["company_name", "product_description"]]
        .drop_duplicates("company_name")
        .reset_index(drop=True)
    )
    logger.info(f"Product corpus: {len(product_corpus)} unique companies")

    # Train/test split by patent_id (no leakage)
    patent_ids = benchmark["patent_id"].unique()
    train_ids, test_ids = train_test_split(patent_ids, test_size=0.2, random_state=42)
    train_bench = benchmark[benchmark["patent_id"].isin(train_ids)]
    test_bench = benchmark[benchmark["patent_id"].isin(test_ids)]

    if args.eval_only:
        if not model_path.exists():
            logger.error(f"No saved model at {model_path}. Run without --eval-only first.")
            return
        retriever = ClassicalRetriever.load(model_path)
        logger.info("Loaded saved classical model")
    else:
        training_pairs = build_training_pairs(train_bench, neg_ratio=args.neg_ratio)
        retriever = ClassicalRetriever()
        retriever.fit(training_pairs)
        retriever.save(model_path)

    # Evaluate on test set overall
    logger.info("Evaluating on test split of PAE-Bench...")
    metrics = compute_metrics(retriever, test_bench, product_corpus)

    logger.info("=" * 50)
    logger.info("TF-IDF + LogReg — Overall Results (test split)")
    logger.info("=" * 50)
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Per-vertical breakdown
    per_vertical = {}
    for vertical in test_bench["vertical"].unique():
        subset = test_bench[test_bench["vertical"] == vertical]
        if len(subset) < 5:
            continue
        v_metrics = compute_metrics(retriever, subset, product_corpus)
        per_vertical[vertical] = v_metrics
        logger.info(f"  [{vertical}] Recall@10: {v_metrics['Recall@10']:.4f}  MRR: {v_metrics['MRR']:.4f}")

    results = {"overall": metrics, "per_vertical": per_vertical}
    out_path = OUTPUTS_DIR / "classical_tfidf_logreg_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
