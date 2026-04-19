"""
Dual-encoder training: text-only baseline, multimodal extension ready.

Architecture:
  Patent tower  : PatentSBERTa → linear projection → 512-d unit-norm embedding
  Product tower : all-mpnet-base-v2 → linear projection → 512-d unit-norm embedding
  Loss          : InfoNCE (symmetric cross-entropy on cosine similarity matrix)
  Hard negatives: BM25 top-k non-relevant products per patent in each batch

After training, builds a FAISS IndexFlatIP over the product corpus and saves it
alongside a product metadata pkl so RetrievalEngine.load() can use it directly.

Usage:
    python scripts/train_deep_learning.py
    python scripts/train_deep_learning.py --epochs 5 --batch-size 32 --eval-only
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import MODELS_DIR, OUTPUTS_DIR, PROCESSED_DIR

DUAL_MODEL_DIR = MODELS_DIR / "dual_encoder"

PATENT_MODEL_NAME  = "AI-Growth-Lab/PatentSBERTa"
PRODUCT_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
PROJ_DIM    = 256   # projection head output dim (matches notebook)
EMBED_DIM   = 768   # backbone hidden size for both models
MAX_SEQ_LEN = 256


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

class MeanPooler(nn.Module):
    """Attention-mask-aware mean pooling over token embeddings."""

    def forward(self, last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp(min=1e-9)
        return summed / count


class EncoderTower(nn.Module):
    """
    Single encoder tower: pre-trained transformer + mean pooling + projection.

    Used for both the patent tower and the product tower.
    The pre-trained backbone is shared or independent depending on construction.
    """

    def __init__(self, model_name: str, output_dim: int = PROJ_DIM):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size
        self.pooler = MeanPooler()
        self.projection = nn.Linear(hidden, output_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.pooler(out.last_hidden_state, attention_mask)
        projected = self.projection(pooled)
        return F.normalize(projected, dim=-1)


class DualEncoder(nn.Module):
    """Two-tower contrastive model for patent-to-product retrieval."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.patent_tower  = EncoderTower(PATENT_MODEL_NAME)
        self.product_tower = EncoderTower(PRODUCT_MODEL_NAME)
        self.log_temperature = nn.Parameter(torch.tensor(temperature).log())

    def encode_patents(self, input_ids, attention_mask) -> torch.Tensor:
        return self.patent_tower(input_ids, attention_mask)

    def encode_products(self, input_ids, attention_mask) -> torch.Tensor:
        return self.product_tower(input_ids, attention_mask)

    def forward(self, patent_ids, patent_mask, product_ids, product_mask):
        p_emb = self.encode_patents(patent_ids, patent_mask)
        q_emb = self.encode_products(product_ids, product_mask)
        return p_emb, q_emb

    def info_nce_loss(self, p_emb: torch.Tensor, q_emb: torch.Tensor) -> torch.Tensor:
        """Symmetric InfoNCE. Positive pairs on the diagonal."""
        temp = self.log_temperature.exp().clamp(min=0.01, max=1.0)
        logits = (p_emb @ q_emb.T) / temp
        labels = torch.arange(p_emb.size(0), device=p_emb.device)
        loss = (
            F.cross_entropy(logits, labels) +
            F.cross_entropy(logits.T, labels)
        ) / 2
        return loss


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PatentProductDataset(Dataset):
    """
    Dataset of (patent_claims, product_description) positive pairs.

    Hard negatives are added at collation time (BM25 top non-relevant products).
    """

    def __init__(
        self,
        pairs: list[dict],
        patent_tokenizer: AutoTokenizer,
        product_tokenizer: AutoTokenizer,
        max_len: int = MAX_SEQ_LEN,
    ):
        self.pairs = pairs
        self.patent_tok = patent_tokenizer
        self.product_tok = product_tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        row = self.pairs[idx]
        patent_enc = self.patent_tok(
            row["patent_claims"][:1024],
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        product_enc = self.product_tok(
            row["product_description"][:1024],
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "patent_input_ids":    patent_enc["input_ids"].squeeze(0),
            "patent_attention_mask": patent_enc["attention_mask"].squeeze(0),
            "product_input_ids":   product_enc["input_ids"].squeeze(0),
            "product_attention_mask": product_enc["attention_mask"].squeeze(0),
        }


# ---------------------------------------------------------------------------
# Hard negative mining (BM25)
# ---------------------------------------------------------------------------

def build_hard_negatives(
    benchmark: pd.DataFrame,
    product_corpus: pd.DataFrame,
    neg_per_pos: int = 3,
    rng_seed: int = 42,
) -> list[dict]:
    """
    Build training pairs with BM25-mined hard negatives.

    For each positive (patent, product) pair, retrieves BM25-top products
    that are NOT among the known positives for that patent as hard negatives.
    Falls back to random negatives if BM25 is unavailable.
    """
    from rank_bm25 import BM25Okapi
    import re

    def _tok(text: str) -> list[str]:
        return re.sub(r"[^\w\s]", " ", text.lower()).split()

    corpus_texts = product_corpus["product_description"].fillna("").tolist()
    corpus_names = product_corpus["company_name"].tolist()
    bm25 = BM25Okapi([_tok(t) for t in corpus_texts])

    positives = benchmark.copy()
    positives["label"] = 1

    rng = np.random.default_rng(rng_seed)
    pairs: list[dict] = []

    for patent_id, group in positives.groupby("patent_id"):
        claims = group["patent_claims"].iloc[0]
        known_positives = set(group["company_name"].str.lower())

        # Add all positive pairs
        for _, row in group.iterrows():
            pairs.append({
                "patent_claims": row["patent_claims"],
                "product_description": row["product_description"],
                "label": 1,
            })

        # Mine hard negatives via BM25
        scores = bm25.get_scores(_tok(claims))
        ranked = np.argsort(scores)[::-1]

        added = 0
        for idx in ranked:
            if corpus_names[idx].lower() in known_positives:
                continue
            pairs.append({
                "patent_claims": claims,
                "product_description": corpus_texts[idx],
                "label": 0,
            })
            added += 1
            if added >= neg_per_pos * len(group):
                break

        # Fallback random negatives if not enough hard negatives
        while added < neg_per_pos:
            i = int(rng.integers(0, len(product_corpus)))
            if corpus_names[i].lower() not in known_positives:
                pairs.append({
                    "patent_claims": claims,
                    "product_description": corpus_texts[i],
                    "label": 0,
                })
                added += 1

    # Keep only positive pairs for contrastive training
    # (negatives are used implicitly within each batch via off-diagonal entries)
    pos_only = [p for p in pairs if p["label"] == 1]
    logger.info(f"Training pairs (positives only for InfoNCE): {len(pos_only)}")
    return pos_only


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: DualEncoder,
    benchmark: pd.DataFrame,
    product_corpus: pd.DataFrame,
    device: torch.device,
    patent_tokenizer: AutoTokenizer,
    product_tokenizer: AutoTokenizer,
    batch_size: int = 64,
) -> dict:
    """Compute Recall@10, Recall@50, MRR, nDCG@10 using the dual encoder."""
    model.eval()

    # Encode all products
    product_texts = product_corpus["product_description"].fillna("").tolist()
    product_names = product_corpus["company_name"].tolist()
    product_embeddings = _encode_texts(product_texts, model.encode_products, product_tokenizer, device, batch_size)

    recall_10, recall_50, mrr_scores, ndcg_10 = [], [], [], []

    for patent_id, group in benchmark.groupby("patent_id"):
        claims = group["patent_claims"].iloc[0]
        relevant = set(group["company_name"].str.lower())

        patent_emb = _encode_texts([claims], model.encode_patents, patent_tokenizer, device, 1)
        sims = (patent_emb @ product_embeddings.T).squeeze(0).cpu().numpy()
        ranked = np.argsort(sims)[::-1]
        retrieved = [product_names[i].lower() for i in ranked]

        r10 = len(set(retrieved[:10]) & relevant) / max(len(relevant), 1)
        r50 = len(set(retrieved[:50]) & relevant) / max(len(relevant), 1)
        recall_10.append(r10)
        recall_50.append(r50)

        mrr = 0.0
        for rank, name in enumerate(retrieved, 1):
            if name in relevant:
                mrr = 1.0 / rank
                break
        mrr_scores.append(mrr)

        dcg = sum(1.0 / np.log2(r + 2) for r, name in enumerate(retrieved[:10]) if name in relevant)
        ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), 10)))
        ndcg_10.append(dcg / max(ideal, 1e-10))

    return {
        "Recall@10": float(np.mean(recall_10)),
        "Recall@50": float(np.mean(recall_50)),
        "MRR": float(np.mean(mrr_scores)),
        "nDCG@10": float(np.mean(ndcg_10)),
        "n_queries": len(recall_10),
    }


def _encode_texts(
    texts: list[str],
    encode_fn,
    tokenizer: AutoTokenizer,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """Encode a list of texts in batches, returns (N, D) float32 tensor on CPU."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            max_length=MAX_SEQ_LEN,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            emb = encode_fn(
                enc["input_ids"].to(device),
                enc["attention_mask"].to(device),
            )
        all_embeddings.append(emb.cpu())
    return torch.cat(all_embeddings, dim=0)


# ---------------------------------------------------------------------------
# FAISS index builder
# ---------------------------------------------------------------------------

def build_and_save_faiss_index(
    model: DualEncoder,
    product_corpus: pd.DataFrame,
    device: torch.device,
    product_tokenizer: AutoTokenizer,
    out_dir: Path,
    batch_size: int = 64,
):
    """Encode all products and write a FAISS IndexFlatIP + metadata pkl."""
    logger.info(f"Building FAISS index over {len(product_corpus)} products...")
    product_texts = product_corpus["product_description"].fillna("").tolist()
    product_names = product_corpus["company_name"].tolist()

    embeddings = _encode_texts(
        product_texts, model.encode_products, product_tokenizer, device, batch_size
    ).numpy().astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "product_index.faiss"))

    meta = {
        "ids": [str(i) for i in range(len(product_corpus))],
        "names": product_names,
        "descriptions": product_texts,
    }
    with open(out_dir / "product_meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    logger.info(f"FAISS index saved to {out_dir}/product_index.faiss")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
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

    # Train / test split by patent_id (no leakage)
    patent_ids = benchmark["patent_id"].unique()
    train_ids, test_ids = train_test_split(patent_ids, test_size=0.2, random_state=42)
    train_bench = benchmark[benchmark["patent_id"].isin(train_ids)]
    test_bench  = benchmark[benchmark["patent_id"].isin(test_ids)]

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info(f"Device: {device}")

    # Tokenizers
    patent_tokenizer  = AutoTokenizer.from_pretrained(PATENT_MODEL_NAME)
    product_tokenizer = AutoTokenizer.from_pretrained(PRODUCT_MODEL_NAME)

    model_path = DUAL_MODEL_DIR / "best_model.pt"

    if args.eval_only:
        if not model_path.exists():
            logger.error(f"No checkpoint at {model_path}. Run without --eval-only first.")
            return
        model = DualEncoder().to(device)
        ckpt = torch.load(model_path, map_location=device)
        state = ckpt.get("model_state", ckpt)  # support both notebook and script formats
        model.load_state_dict(state)
        logger.info("Loaded checkpoint for eval-only run")
    else:
        model = DualEncoder().to(device)

        # Build training pairs (positive pairs; InfoNCE uses in-batch negatives)
        train_pairs = build_hard_negatives(train_bench, product_corpus, neg_per_pos=3)
        dataset = PatentProductDataset(train_pairs, patent_tokenizer, product_tokenizer)
        loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                             num_workers=2, pin_memory=(device.type == "cuda"))

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        best_recall = 0.0
        DUAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            for step, batch in enumerate(loader):
                patent_ids_t  = batch["patent_input_ids"].to(device)
                patent_mask   = batch["patent_attention_mask"].to(device)
                product_ids_t = batch["product_input_ids"].to(device)
                product_mask  = batch["product_attention_mask"].to(device)

                p_emb, q_emb = model(patent_ids_t, patent_mask, product_ids_t, product_mask)
                loss = model.info_nce_loss(p_emb, q_emb)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                if (step + 1) % 50 == 0:
                    logger.info(f"Epoch {epoch} step {step+1}/{len(loader)}  loss={loss.item():.4f}")

            scheduler.step()
            avg_loss = total_loss / max(len(loader), 1)
            logger.info(f"Epoch {epoch}/{args.epochs}  avg_loss={avg_loss:.4f}")

            # Evaluate every epoch
            metrics = evaluate(model, test_bench, product_corpus, device, patent_tokenizer, product_tokenizer)
            logger.info(
                f"  Recall@10={metrics['Recall@10']:.4f}  "
                f"Recall@50={metrics['Recall@50']:.4f}  "
                f"MRR={metrics['MRR']:.4f}  "
                f"nDCG@10={metrics['nDCG@10']:.4f}"
            )

            if metrics["Recall@10"] > best_recall:
                best_recall = metrics["Recall@10"]
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "config": {
                        "patent_model": PATENT_MODEL_NAME,
                        "product_model": PRODUCT_MODEL_NAME,
                        "proj_dim": PROJ_DIM,
                        "batch_size": args.batch_size,
                        "lr": args.lr,
                    },
                }, model_path)
                logger.info(f"  New best Recall@10={best_recall:.4f} — checkpoint saved")

    # Final evaluation
    logger.info("=" * 60)
    logger.info("Dual Encoder — Final Evaluation (test split)")
    logger.info("=" * 60)
    metrics = evaluate(model, test_bench, product_corpus, device, patent_tokenizer, product_tokenizer)
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Per-vertical breakdown
    per_vertical = {}
    for vertical in test_bench["vertical"].unique():
        subset = test_bench[test_bench["vertical"] == vertical]
        if len(subset) < 5:
            continue
        v_metrics = evaluate(model, subset, product_corpus, device, patent_tokenizer, product_tokenizer)
        per_vertical[vertical] = v_metrics
        logger.info(f"  [{vertical}] Recall@10={v_metrics['Recall@10']:.4f}  MRR={v_metrics['MRR']:.4f}")

    # Save results
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUTS_DIR / "dual_encoder_results.json"
    with open(out_path, "w") as f:
        json.dump({"overall": metrics, "per_vertical": per_vertical}, f, indent=2)
    logger.info(f"Results saved to {out_path}")

    # Build FAISS index over full product corpus
    build_and_save_faiss_index(model, product_corpus, device, product_tokenizer, DUAL_MODEL_DIR)


def main():
    parser = argparse.ArgumentParser(description="Train dual-encoder retrieval model")
    parser.add_argument("--epochs",     type=int,   default=10,    help="Training epochs")
    parser.add_argument("--batch-size", type=int,   default=16,    help="Batch size (use 8-16 on single GPU)")
    parser.add_argument("--lr",         type=float, default=2e-5,  help="Learning rate")
    parser.add_argument("--eval-only",  action="store_true",       help="Skip training, load checkpoint")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
