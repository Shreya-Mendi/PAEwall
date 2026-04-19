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

## Evaluation Results

All models evaluated on **PAE-Bench** — 522 patent–defendant pairs from CourtListener litigation records spanning five verticals (software, consumer electronics, medical devices, industrial, other).

| Model | Recall@10 | Recall@50 | MRR | nDCG@10 | n_queries |
|-------|-----------|-----------|-----|---------|-----------|
| BM25 Baseline | 0.130 | 0.468 | 0.168 | 0.108 | 483 |
| TF-IDF + LogReg | 0.241 | 0.485 | **0.332** | **0.229** | 97 |
| Dual-Encoder (text-only, fine-tuned) | **0.328** | **0.580** | 0.075 | 0.119 | 49 |

The dual encoder achieves the best **Recall@10** (+36% over classical), finding more true defendants in the top 10. Classical ML wins on **MRR/nDCG** — it ranks correctly when it finds the answer. The gap suggests more training epochs and a larger training set would close the ranking precision deficit.

Per-vertical breakdown and failure analysis: `data/outputs/`

```bash
python scripts/evaluate.py      # runs all available models
python scripts/error_analysis.py  # BM25 vs classical failure analysis
```

## Deployment

The app is deployed on [Railway](https://railway.app) from the `deploy` branch.

Required environment variables:
```
DUKE_LLM_API_KEY=<key>
DUKE_LLM_BASE_URL=https://litellm.oit.duke.edu/v1
DUKE_LLM_MODEL=GPT 4.1 Mini
```

To run locally:
```bash
source .venv/bin/activate
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

## Data Sources

All data is from public sources:

- [CourtListener RECAP](https://courtlistener.com/recap) — Litigation dockets (nature of suit 830 = patent)
- [Google Patents BigQuery](https://cloud.google.com/bigquery/public-data) — Patent full text and claims
- [SEC EDGAR](https://sec.gov/edgar) — Company 10-K filings (Item 1 product descriptions)

## Experiment: Fine-Tuning Impact (RQ1)

Ablation study comparing four retrieval conditions:
1. **BM25** — keyword baseline, no learning
2. **TF-IDF + LogReg** — classical ML with learned feature weights
3. **PatentSBERTa zero-shot** — pre-trained bi-encoder, no fine-tuning
4. **Dual-encoder fine-tuned** — PatentSBERTa + all-mpnet, InfoNCE loss on litigation pairs

Hypothesis: fine-tuning on litigation-derived positive pairs outperforms zero-shot PatentSBERTa.
Results saved to `data/outputs/ablation_results.json`.
