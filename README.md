# PAEwall

A multimodal, faithfulness-grounded patent infringement discovery system.

PAEwall automates the patent assertion entity (PAE) pipeline: given a patent, it retrieves candidate infringing products, generates faithfulness-checked claim charts, and produces red-team counter-arguments — giving patent holders an honest view of enforcement viability.

## Features

- **Patent Intake** — Upload a USPTO patent number or PDF; claims are parsed and structured automatically
- **Infringement Target Discovery** — Cross-vertical dual-encoder retrieval matches patents to candidate products using both text and images
- **Faithfulness-Scored Claim Charts** — LLM-generated claim charts with per-limitation faithfulness verification
- **Red-Team Counter-Arguments** — Non-infringement arguments, invalidity risks, and enforcement probability estimates

## Three Modeling Approaches

| Model | Type | Description | Location |
|-------|------|-------------|----------|
| BM25 Baseline | Naive | Keyword retrieval over product descriptions | `scripts/train_naive.py` |
| TF-IDF + LogReg | Classical ML | TF-IDF features with logistic regression scoring | `scripts/train_classical.py` |
| Dual-Encoder | Deep Learning | Multimodal two-tower contrastive model (PatentSBERTa + SigLIP) | `scripts/train_deep_learning.py` |

All three are evaluated on PAE-Bench with Recall@10, Recall@50, MRR, and nDCG@10.

## Training

Model training is done on **Google Colab** (A100 GPU). Training notebooks are in `notebooks/` for Colab execution; trained model artifacts are downloaded to `models/` for local inference and the deployed app.

## Repository Structure

```
├── README.md                          <- This file
├── requirements.txt                   <- Python dependencies
├── setup.py                           <- Setup and training pipeline
├── app.py                             <- Web application (FastAPI)
├── scripts/
│   ├── make_dataset.py                <- Data collection pipeline
│   ├── build_features.py              <- Feature engineering
│   ├── train_naive.py                 <- Naive baseline (BM25)
│   ├── train_classical.py             <- Classical model (TF-IDF + LogReg)
│   ├── train_deep_learning.py         <- Deep learning model (Dual-Encoder)
│   └── evaluate.py                    <- Unified evaluation on PAE-Bench
├── src/
│   ├── patent_intake/                 <- Module A: claim parsing + CPC classification
│   ├── retrieval/                     <- Module B: dual-encoder + FAISS index
│   ├── claim_chart/                   <- Module C: LLM generation + faithfulness verifier
│   ├── red_team/                      <- Module D: counter-arguments + enforcement probability
│   └── app/                           <- Frontend templates and static assets
├── models/                            <- Trained model artifacts
├── data/
│   ├── raw/                           <- Raw downloaded data
│   ├── processed/                     <- Processed features and benchmarks
│   └── outputs/                       <- Evaluation results and figures
├── notebooks/                         <- Exploration notebooks (not graded)
└── .gitignore
```

## Setup

### Prerequisites

- Python 3.11+
- pip

### Installation

```bash
git clone https://github.com/<your-username>/paewall.git
cd paewall
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Data & Training

```bash
# Download raw data
python setup.py --data

# Build features and benchmarks
python setup.py --features

# Train all three models
python setup.py --train

# Or train individually
python setup.py --train-naive
python setup.py --train-classical
python setup.py --train-dl

# Or run the full pipeline
python setup.py --all
```

### Running the App

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Then open http://localhost:8000 in your browser.

## Evaluation

Run the unified evaluation script to compare all three models on PAE-Bench:

```bash
python scripts/evaluate.py
```

Results are saved to `data/outputs/`.

## Data Sources

All data is from public sources:

- [USPTO PatentsView](https://patentsview.org) — Patent full text, claims, CPC codes
- [Google Patents Public Datasets](https://cloud.google.com/bigquery/public-data) — BigQuery patent data
- [CourtListener RECAP](https://courtlistener.com/recap) — Litigation records and claim charts
- [PTAB Open Data](https://developer.uspto.gov) — Inter Partes Review outcomes
- [SEC EDGAR](https://sec.gov/edgar) — Company 10-K filings with product descriptions

## Experiment

**Text-only vs. Multimodal Retrieval (RQ2)**

Ablation study comparing:
1. Text-only dual-encoder (claims + product descriptions)
2. Full multimodal dual-encoder (claims + figures + descriptions + screenshots)

Per-vertical sub-splits reveal where visual grounding helps most (e.g., UI-heavy software patents vs. method patents).

See `scripts/evaluate.py` and `notebooks/` for experiment details.
