"""
Experiment: Impact of Fine-Tuning on Cross-Vertical Patent Retrieval

Ablation study comparing four retrieval conditions to answer RQ1:
  (1) BM25 keyword baseline
  (2) TF-IDF + Logistic Regression classical ML
  (3) PatentSBERTa zero-shot  (pre-trained, NO fine-tuning)
  (4) Dual-encoder fine-tuned (PatentSBERTa + all-mpnet, InfoNCE on PAE-Bench)

Hypothesis: Fine-tuning on litigation-derived positive pairs outperforms
zero-shot PatentSBERTa on cross-vertical retrieval.

Usage:
    python scripts/experiment_ablation.py
    python scripts/experiment_ablation.py --skip-zeroshot   # if no GPU
"""

import argparse
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


# ---------------------------------------------------------------------------
# Shared metric helpers
# ---------------------------------------------------------------------------

def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return len(set(retrieved[:k]) & relevant) / max(len(relevant), 1)

def mrr(retrieved: list[str], relevant: set[str]) -> float:
    for rank, name in enumerate(retrieved, 1):
        if name in relevant:
            return 1.0 / rank
    return 0.0

def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int = 10) -> float:
    dcg  = sum(1.0 / np.log2(r + 2) for r, c in enumerate(retrieved[:k]) if c in relevant)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / max(idcg, 1e-10)

def aggregate_metrics(results_per_query: list[dict]) -> dict:
    return {
        "Recall@10":  float(np.mean([r["Recall@10"]  for r in results_per_query])),
        "Recall@50":  float(np.mean([r["Recall@50"]  for r in results_per_query])),
        "MRR":        float(np.mean([r["MRR"]        for r in results_per_query])),
        "nDCG@10":    float(np.mean([r["nDCG@10"]    for r in results_per_query])),
        "n_queries":  len(results_per_query),
    }


# ---------------------------------------------------------------------------
# Condition 1: BM25 (load saved model)
# ---------------------------------------------------------------------------

def eval_bm25(benchmark: pd.DataFrame, product_corpus: pd.DataFrame) -> dict:
    model_path = MODELS_DIR / "naive" / "bm25.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"BM25 model not found at {model_path}. Run train_naive.py first.")
    with open(model_path, "rb") as f:
        retriever = pickle.load(f)

    per_query = []
    for patent_id, group in benchmark.groupby("patent_id"):
        claims   = group["patent_claims"].iloc[0]
        relevant = set(group["company_name"].str.lower())
        results  = retriever.predict(claims, top_k=50)
        retrieved = [r["company_name"].lower() for r in results]
        per_query.append({
            "patent_id":  patent_id,
            "Recall@10":  recall_at_k(retrieved, relevant, 10),
            "Recall@50":  recall_at_k(retrieved, relevant, 50),
            "MRR":        mrr(retrieved, relevant),
            "nDCG@10":    ndcg_at_k(retrieved, relevant, 10),
        })
    return aggregate_metrics(per_query)


# ---------------------------------------------------------------------------
# Condition 2: TF-IDF + LogReg (load saved model)
# ---------------------------------------------------------------------------

def eval_classical(benchmark: pd.DataFrame, product_corpus: pd.DataFrame) -> dict:
    model_path = MODELS_DIR / "classical" / "tfidf_logreg.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Classical model not found. Run train_classical.py first.")
    with open(model_path, "rb") as f:
        retriever = pickle.load(f)

    per_query = []
    for patent_id, group in benchmark.groupby("patent_id"):
        claims   = group["patent_claims"].iloc[0]
        relevant = set(group["company_name"].str.lower())
        results  = retriever.rank_products(claims, product_corpus, top_k=50)
        retrieved = [r["company_name"].lower() for r in results]
        per_query.append({
            "patent_id":  patent_id,
            "Recall@10":  recall_at_k(retrieved, relevant, 10),
            "Recall@50":  recall_at_k(retrieved, relevant, 50),
            "MRR":        mrr(retrieved, relevant),
            "nDCG@10":    ndcg_at_k(retrieved, relevant, 10),
        })
    return aggregate_metrics(per_query)


# ---------------------------------------------------------------------------
# Condition 3: PatentSBERTa zero-shot
# ---------------------------------------------------------------------------

def eval_zeroshot(benchmark: pd.DataFrame, product_corpus: pd.DataFrame) -> dict:
    """
    Zero-shot retrieval using PatentSBERTa (patent side) and
    all-mpnet-base-v2 (product side) with NO fine-tuning.

    This is the direct comparison point for RQ1: does fine-tuning help?
    """
    import torch
    from transformers import AutoModel, AutoTokenizer
    import torch.nn.functional as F

    PATENT_MODEL  = "AI-Growth-Lab/PatentSBERTa"
    PRODUCT_MODEL = "sentence-transformers/all-mpnet-base-v2"
    MAX_LEN = 256
    BATCH   = 32
    device  = torch.device("cuda" if torch.cuda.is_available()
                           else "mps" if torch.backends.mps.is_available()
                           else "cpu")
    logger.info(f"Zero-shot eval on {device}")

    patent_tok  = AutoTokenizer.from_pretrained(PATENT_MODEL)
    product_tok = AutoTokenizer.from_pretrained(PRODUCT_MODEL)
    patent_enc  = AutoModel.from_pretrained(PATENT_MODEL).to(device).eval()
    product_enc = AutoModel.from_pretrained(PRODUCT_MODEL).to(device).eval()

    def mean_pool(out, mask):
        m = mask.unsqueeze(-1).float()
        return (out.last_hidden_state * m).sum(1) / m.sum(1).clamp(min=1e-9)

    def encode(texts: list[str], tokenizer, model) -> np.ndarray:
        all_embs = []
        for i in range(0, len(texts), BATCH):
            batch = texts[i : i + BATCH]
            enc = tokenizer(batch, max_length=MAX_LEN, truncation=True,
                            padding=True, return_tensors="pt").to(device)
            with torch.no_grad():
                emb = mean_pool(model(**enc), enc["attention_mask"])
                emb = F.normalize(emb, dim=-1)
            all_embs.append(emb.cpu().float().numpy())
        return np.vstack(all_embs)

    logger.info("Encoding product corpus (zero-shot)...")
    product_texts = product_corpus["product_description"].fillna("").tolist()
    product_names = product_corpus["company_name"].tolist()
    product_embs  = encode(product_texts, product_tok, product_enc)  # (N, D)

    per_query = []
    unique_patents = benchmark.groupby("patent_id")
    logger.info(f"Evaluating {len(unique_patents)} patents (zero-shot)...")

    for patent_id, group in unique_patents:
        claims   = group["patent_claims"].iloc[0]
        relevant = set(group["company_name"].str.lower())

        patent_emb = encode([claims], patent_tok, patent_enc)  # (1, D)
        sims = (patent_emb @ product_embs.T).squeeze(0)        # (N,)
        ranked_idx = np.argsort(sims)[::-1]
        retrieved  = [product_names[i].lower() for i in ranked_idx]

        per_query.append({
            "patent_id": patent_id,
            "Recall@10": recall_at_k(retrieved, relevant, 10),
            "Recall@50": recall_at_k(retrieved, relevant, 50),
            "MRR":       mrr(retrieved, relevant),
            "nDCG@10":   ndcg_at_k(retrieved, relevant, 10),
        })

    return aggregate_metrics(per_query)


# ---------------------------------------------------------------------------
# Condition 4: Fine-tuned dual encoder (load from checkpoint if available)
# ---------------------------------------------------------------------------

def eval_finetuned(benchmark: pd.DataFrame, product_corpus: pd.DataFrame) -> dict | None:
    """Load the fine-tuned dual encoder and evaluate. Returns None if not trained."""
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    model_path = MODELS_DIR / "dual_encoder" / "best_model.pt"
    if not model_path.exists():
        logger.warning("Fine-tuned checkpoint not found — skipping condition 4.")
        return None

    PATENT_MODEL  = "AI-Growth-Lab/PatentSBERTa"
    PRODUCT_MODEL = "sentence-transformers/all-mpnet-base-v2"
    PROJ_DIM = 256
    MAX_LEN  = 256
    BATCH    = 32
    device   = torch.device("cuda" if torch.cuda.is_available()
                            else "mps" if torch.backends.mps.is_available()
                            else "cpu")

    import torch.nn as nn

    class _Tower(nn.Module):
        def __init__(self, model_name):
            super().__init__()
            self.backbone   = AutoModel.from_pretrained(model_name)
            self.projection = nn.Linear(self.backbone.config.hidden_size, PROJ_DIM)

        def forward(self, input_ids, attention_mask):
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            m   = attention_mask.unsqueeze(-1).float()
            pooled = (out.last_hidden_state * m).sum(1) / m.sum(1).clamp(min=1e-9)
            return F.normalize(self.projection(pooled), dim=-1)

    class _DualEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.patent_encoder  = _Tower(PATENT_MODEL)
            self.product_encoder = _Tower(PRODUCT_MODEL)
            self.log_temperature = nn.Parameter(torch.tensor(0.07).log())

        def encode_patents(self, input_ids, attention_mask):
            return self.patent_encoder(input_ids, attention_mask)

        def encode_products(self, input_ids, attention_mask):
            return self.product_encoder(input_ids, attention_mask)

    model = _DualEncoder().to(device)
    ckpt  = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    logger.info(f"Loaded fine-tuned checkpoint (epoch {ckpt.get('epoch','?')})")

    patent_tok  = AutoTokenizer.from_pretrained(PATENT_MODEL)
    product_tok = AutoTokenizer.from_pretrained(PRODUCT_MODEL)

    def encode(texts, tokenizer, encode_fn):
        all_embs = []
        for i in range(0, len(texts), BATCH):
            batch = texts[i : i + BATCH]
            enc = tokenizer(batch, max_length=MAX_LEN, truncation=True,
                            padding=True, return_tensors="pt").to(device)
            with torch.no_grad():
                all_embs.append(encode_fn(**enc).cpu().float().numpy())
        return np.vstack(all_embs)

    logger.info("Encoding product corpus (fine-tuned)...")
    product_embs  = encode(product_corpus["product_description"].fillna("").tolist(),
                           product_tok, model.encode_products)
    product_names = product_corpus["company_name"].tolist()

    per_query = []
    for patent_id, group in benchmark.groupby("patent_id"):
        claims   = group["patent_claims"].iloc[0]
        relevant = set(group["company_name"].str.lower())
        p_emb    = encode([claims], patent_tok, model.encode_patents)
        sims     = (p_emb @ product_embs.T).squeeze(0)
        ranked   = np.argsort(sims)[::-1]
        retrieved = [product_names[i].lower() for i in ranked]
        per_query.append({
            "patent_id": patent_id,
            "Recall@10": recall_at_k(retrieved, relevant, 10),
            "Recall@50": recall_at_k(retrieved, relevant, 50),
            "MRR":       mrr(retrieved, relevant),
            "nDCG@10":   ndcg_at_k(retrieved, relevant, 10),
        })
    return aggregate_metrics(per_query)


# ---------------------------------------------------------------------------
# Per-vertical breakdown
# ---------------------------------------------------------------------------

def per_vertical_metrics(benchmark: pd.DataFrame, fn, product_corpus: pd.DataFrame) -> dict:
    out = {}
    for vertical in benchmark["vertical"].unique():
        subset = benchmark[benchmark["vertical"] == vertical]
        if len(subset) < 5:
            continue
        try:
            m = fn(subset, product_corpus)
            out[vertical] = m
        except Exception as e:
            logger.warning(f"Vertical {vertical} failed: {e}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Ablation experiment: zero-shot vs fine-tuned")
    parser.add_argument("--skip-zeroshot", action="store_true",
                        help="Skip PatentSBERTa zero-shot (no GPU / slow CPU)")
    args = parser.parse_args()

    bench_path = PROCESSED_DIR / "pae_bench.parquet"
    if not bench_path.exists():
        logger.error("PAE-Bench not found. Run make_dataset.py --assemble first.")
        return

    benchmark = pd.read_parquet(bench_path)
    logger.info(f"PAE-Bench: {len(benchmark)} rows, {benchmark['patent_id'].nunique()} patents")

    product_corpus = (
        benchmark[["company_name", "product_description"]]
        .drop_duplicates("company_name")
        .reset_index(drop=True)
    )

    # Use full benchmark for ablation (not just test split) so results are comparable
    # to the per-model evaluations already run
    patent_ids = benchmark["patent_id"].unique()
    _, test_ids = train_test_split(patent_ids, test_size=0.2, random_state=42)
    test_bench  = benchmark[benchmark["patent_id"].isin(test_ids)]

    conditions = {}

    logger.info("=== Condition 1: BM25 ===")
    try:
        conditions["bm25"] = eval_bm25(test_bench, product_corpus)
        logger.info(f"  R@10={conditions['bm25']['Recall@10']:.4f}  MRR={conditions['bm25']['MRR']:.4f}")
    except Exception as e:
        logger.error(f"BM25 failed: {e}")

    logger.info("=== Condition 2: TF-IDF + LogReg ===")
    try:
        conditions["classical"] = eval_classical(test_bench, product_corpus)
        logger.info(f"  R@10={conditions['classical']['Recall@10']:.4f}  MRR={conditions['classical']['MRR']:.4f}")
    except Exception as e:
        logger.error(f"Classical failed: {e}")

    if not args.skip_zeroshot:
        logger.info("=== Condition 3: PatentSBERTa zero-shot ===")
        try:
            conditions["zeroshot_patentsbert"] = eval_zeroshot(test_bench, product_corpus)
            logger.info(f"  R@10={conditions['zeroshot_patentsbert']['Recall@10']:.4f}  "
                        f"MRR={conditions['zeroshot_patentsbert']['MRR']:.4f}")
        except Exception as e:
            logger.error(f"Zero-shot failed: {e}")
    else:
        logger.info("Skipping zero-shot (--skip-zeroshot set)")

    logger.info("=== Condition 4: Fine-tuned Dual Encoder ===")
    result = eval_finetuned(test_bench, product_corpus)
    if result:
        conditions["dual_encoder_finetuned"] = result
        logger.info(f"  R@10={result['Recall@10']:.4f}  MRR={result['MRR']:.4f}")

    # Summary table
    logger.info("\n" + "=" * 65)
    logger.info(f"{'Model':<30} {'R@10':>7} {'R@50':>7} {'MRR':>7} {'nDCG@10':>9}")
    logger.info("=" * 65)
    labels = {
        "bm25":                    "BM25 (naive baseline)",
        "classical":               "TF-IDF + LogReg (classical)",
        "zeroshot_patentsbert":    "PatentSBERTa zero-shot",
        "dual_encoder_finetuned":  "Dual Encoder (fine-tuned)",
    }
    for key, label in labels.items():
        if key in conditions:
            m = conditions[key]
            logger.info(f"{label:<30} {m['Recall@10']:>7.4f} {m['Recall@50']:>7.4f} "
                        f"{m['MRR']:>7.4f} {m['nDCG@10']:>9.4f}")
    logger.info("=" * 65)

    out = {
        "experiment": "ablation_finetuning_impact",
        "hypothesis": "Fine-tuning on litigation-derived pairs outperforms zero-shot PatentSBERTa (RQ1)",
        "conditions": conditions,
    }
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / "ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
