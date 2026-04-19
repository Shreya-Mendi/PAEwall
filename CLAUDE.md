# PAEwall — Claude Code Instructions

## Project
Multimodal patent infringement discovery system (Duke MEng AIPI DL final project).
Given a patent, retrieves companies likely infringing it, generates claim charts, and
produces faithfulness-scored counter-arguments.

## Repo Layout
```
PAEwall/
├── scripts/
│   ├── config.py          # All paths, API keys, rate limits
│   ├── make_dataset.py    # Data collection + PAE-Bench assembly
│   ├── train_naive.py     # BM25 baseline
│   ├── train_classical.py # TF-IDF + LogReg
│   └── *.py
├── notebooks/
│   └── train_dual_encoder.ipynb  # GPU training (VSCode Remote SSH)
├── data/
│   ├── raw/               # litigation_dockets.json, patents.json, company_products.json
│   └── processed/         # pae_bench.parquet
├── models/
│   ├── naive/             # bm25.pkl
│   ├── classical/         # tfidf_logreg.pkl
│   └── dual_encoder/      # best_model.pt, product_index.faiss
├── app.py                 # FastAPI app
└── requirements.txt
```

## Environment
- Python venv: `.venv/` (activate with `source .venv/bin/activate`)
- Always run scripts from the repo root: `python scripts/make_dataset.py`
- GPU training: open `notebooks/train_dual_encoder.ipynb` via VSCode Remote SSH

## Credentials (.env — gitignored)
```
COURTLISTENER_TOKEN=43bde0c6ad5cf508a8f7d7b2fc3e332707041b43
GCP_PROJECT_ID=project-dbd93984-938b-4576-bf2
EDGAR_USER_AGENT=PAEwall shreya.mendi@duke.edu
```
BigQuery auth: `gcloud auth application-default login` (use system Python, not venv).

## Data Pipeline (run in order)
```bash
# 1. Collect litigation records from CourtListener
python scripts/make_dataset.py --litigation

# 2. Collect patents from Google Patents BigQuery
python scripts/make_dataset.py --patents

# 3. Collect product descriptions from SEC EDGAR
python scripts/make_dataset.py --products

# 4. Assemble PAE-Bench parquet
python scripts/make_dataset.py --assemble
```

## Training
```bash
# BM25 naive baseline
python scripts/train_naive.py

# TF-IDF + LogReg classical model
python scripts/train_classical.py

# Dual-encoder (run in notebook on GPU)
# Open notebooks/train_dual_encoder.ipynb via VSCode Remote SSH
```

## Evaluation Metrics
Recall@10, Recall@50, MRR, nDCG@10 — computed per-vertical and overall.

## Key Design Decisions
- CourtListener: use `/search/` endpoint (type=r, nature_of_suit=830) — not /dockets/
- Patent lookup: BigQuery assignee search, not PatentsView
- Product descriptions: SEC EDGAR 10-K Item 1, via company_tickers.json CIK lookup
- Train/test split by patent_id to prevent leakage
- Dual encoder: PatentSBERTa (patent side) + all-mpnet-base-v2 (product side), InfoNCE loss

## Git Workflow
- `main` — stable releases only
- `develop` — integration branch
- `feature/*` — all new work branches off develop
- Never commit directly to main
