"""
Error Analysis — 5 specific retrieval failures with root cause diagnosis.

For each failure, reports:
- Patent ID + claims excerpt
- True company (should have been retrieved)
- What the model returned instead (top-3)
- Root cause category
- Proposed mitigation

Usage:
    python scripts/error_analysis.py
"""

import importlib
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import MODELS_DIR, OUTPUTS_DIR, PROCESSED_DIR

_CLASS_MAP = {"ClassicalRetriever": "train_classical", "BM25Retriever": "train_naive"}

class _ScriptUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__" and name in _CLASS_MAP:
            return getattr(importlib.import_module(_CLASS_MAP[name]), name)
        return super().find_class(module, name)

def _load_pkl(path):
    with open(path, "rb") as f:
        return _ScriptUnpickler(f).load()


ROOT_CAUSE_RULES = {
    "vocabulary_mismatch": lambda claims, product: (
        len(set(_tok(claims[:500])) & set(_tok(product[:500]))) < 3
    ),
    "truncated_claims": lambda claims, product: len(claims.split()) > 400,
    "sparse_product_description": lambda claims, product: len(product.split()) < 30,
    "generic_claim_language": lambda claims, product: (
        sum(claims.lower().count(w) for w in ["computer", "system", "method", "process",
                                               "device", "apparatus"]) > 8
    ),
}

MITIGATIONS = {
    "vocabulary_mismatch":        "Expand product corpus with domain synonyms; train on more vertical-diverse pairs.",
    "truncated_claims":           "Increase MAX_SEQ_LEN or summarize claims before encoding.",
    "sparse_product_description": "Supplement EDGAR descriptions with company website scraping.",
    "generic_claim_language":     "Use claim element decomposition (preamble only) rather than full claims text.",
    "other":                      "Increase training data and hard negative diversity for this vertical.",
}


def _tok(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def diagnose(claims: str, true_product: str) -> tuple[str, str]:
    """Return (root_cause_label, mitigation)."""
    for cause, rule in ROOT_CAUSE_RULES.items():
        try:
            if rule(claims, true_product):
                return cause, MITIGATIONS[cause]
        except Exception:
            pass
    return "other", MITIGATIONS["other"]


def run_bm25_errors(benchmark: pd.DataFrame, product_corpus: pd.DataFrame, n: int = 5) -> list[dict]:
    """Find the n worst BM25 failures — cases where the true company is not in top-10."""
    model_path = MODELS_DIR / "naive" / "bm25.pkl"
    retriever = _load_pkl(model_path)

    failures = []
    for patent_id, group in benchmark.groupby("patent_id"):
        claims   = group["patent_claims"].iloc[0]
        relevant = set(group["company_name"].str.lower())
        results  = retriever.predict(claims, top_k=50)
        retrieved = [r["company_name"].lower() for r in results]
        retrieved_top10 = set(retrieved[:10])

        missed = relevant - retrieved_top10
        if not missed:
            continue

        true_company = list(missed)[0]
        true_rank = next((i + 1 for i, name in enumerate(retrieved) if name == true_company), None)
        true_product = group.loc[group["company_name"].str.lower() == true_company,
                                 "product_description"].values
        true_product_text = true_product[0] if len(true_product) > 0 else ""

        root_cause, mitigation = diagnose(claims, true_product_text)

        failures.append({
            "patent_id":        patent_id,
            "vertical":         group["vertical"].iloc[0],
            "claims_excerpt":   claims[:300].replace("\n", " ").strip() + "…",
            "true_company":     true_company,
            "true_rank":        true_rank,
            "top3_returned":    retrieved[:3],
            "root_cause":       root_cause,
            "mitigation":       mitigation,
            "true_product_words": len(true_product_text.split()),
            "claims_words":     len(claims.split()),
        })

    # Sort by how badly the model missed (highest rank first = worst miss)
    failures.sort(key=lambda x: x["true_rank"] if x["true_rank"] else 9999, reverse=True)
    return failures[:n]


def run_classical_errors(benchmark: pd.DataFrame, product_corpus: pd.DataFrame, n: int = 5) -> list[dict]:
    """Find the n worst classical model failures."""
    model_path = MODELS_DIR / "classical" / "tfidf_logreg.pkl"
    retriever = _load_pkl(model_path)

    failures = []
    for patent_id, group in benchmark.groupby("patent_id"):
        claims   = group["patent_claims"].iloc[0]
        relevant = set(group["company_name"].str.lower())
        results  = retriever.rank_products(claims, product_corpus, top_k=50)
        retrieved = [r["company_name"].lower() for r in results]
        retrieved_top10 = set(retrieved[:10])

        missed = relevant - retrieved_top10
        if not missed:
            continue

        true_company = list(missed)[0]
        true_rank    = next((i + 1 for i, name in enumerate(retrieved) if name == true_company), None)
        true_product = group.loc[group["company_name"].str.lower() == true_company,
                                 "product_description"].values
        true_product_text = true_product[0] if len(true_product) > 0 else ""
        root_cause, mitigation = diagnose(claims, true_product_text)

        failures.append({
            "patent_id":      patent_id,
            "vertical":       group["vertical"].iloc[0],
            "claims_excerpt": claims[:300].replace("\n", " ").strip() + "…",
            "true_company":   true_company,
            "true_rank":      true_rank,
            "top3_returned":  retrieved[:3],
            "root_cause":     root_cause,
            "mitigation":     mitigation,
            "true_product_words": len(true_product_text.split()),
            "claims_words":   len(claims.split()),
        })

    failures.sort(key=lambda x: x["true_rank"] if x["true_rank"] else 9999, reverse=True)
    return failures[:n]


def main():
    bench_path = PROCESSED_DIR / "pae_bench.parquet"
    if not bench_path.exists():
        logger.error("PAE-Bench not found. Run make_dataset.py --assemble first.")
        return

    benchmark = pd.read_parquet(bench_path)
    product_corpus = (
        benchmark[["company_name", "product_description"]]
        .drop_duplicates("company_name")
        .reset_index(drop=True)
    )

    patent_ids = benchmark["patent_id"].unique()
    _, test_ids = train_test_split(patent_ids, test_size=0.2, random_state=42)
    test_bench  = benchmark[benchmark["patent_id"].isin(test_ids)]

    logger.info("Running BM25 error analysis...")
    bm25_errors = run_bm25_errors(test_bench, product_corpus, n=5)

    logger.info("Running classical model error analysis...")
    classical_errors = run_classical_errors(test_bench, product_corpus, n=5)

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("ERROR ANALYSIS — BM25 Top-5 Failures")
    logger.info("=" * 70)
    for i, e in enumerate(bm25_errors, 1):
        logger.info(f"\n[{i}] Patent: {e['patent_id']}  Vertical: {e['vertical']}")
        logger.info(f"    Claims ({e['claims_words']} words): {e['claims_excerpt'][:120]}…")
        logger.info(f"    True company: {e['true_company']}  (ranked #{e['true_rank']})")
        logger.info(f"    Product description: {e['true_product_words']} words")
        logger.info(f"    Top-3 returned: {e['top3_returned']}")
        logger.info(f"    Root cause: {e['root_cause']}")
        logger.info(f"    Mitigation: {e['mitigation']}")

    logger.info("\n" + "=" * 70)
    logger.info("ERROR ANALYSIS — Classical Top-5 Failures")
    logger.info("=" * 70)
    for i, e in enumerate(classical_errors, 1):
        logger.info(f"\n[{i}] Patent: {e['patent_id']}  Vertical: {e['vertical']}")
        logger.info(f"    Claims ({e['claims_words']} words): {e['claims_excerpt'][:120]}…")
        logger.info(f"    True company: {e['true_company']}  (ranked #{e['true_rank']})")
        logger.info(f"    Product description: {e['true_product_words']} words")
        logger.info(f"    Top-3 returned: {e['top3_returned']}")
        logger.info(f"    Root cause: {e['root_cause']}")
        logger.info(f"    Mitigation: {e['mitigation']}")

    out = {
        "bm25_failures":      bm25_errors,
        "classical_failures": classical_errors,
    }
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / "error_analysis.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
