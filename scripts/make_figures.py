"""
Generate paper figures from data/outputs/*.json.

Outputs are written to paper/figures/ so the paper can reference them directly.
All data is read from existing evaluation JSON files — no model re-runs needed.

Usage:
    python scripts/make_figures.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT / "data" / "outputs"
FIG_DIR = ROOT / "paper" / "figures"

MODEL_NAMES = ["BM25", "Classical\n(TF-IDF+LR)", "Dual-Encoder\n(fine-tuned)"]
MODEL_COLORS = ["#8c8c8c", "#3b82f6", "#10b981"]
VERTICALS = ["software", "consumer_electronics", "medical_devices", "industrial", "other"]
VERTICAL_LABELS = {
    "software": "Software",
    "consumer_electronics": "Consumer\nElectronics",
    "medical_devices": "Medical\nDevices",
    "industrial": "Industrial",
    "other": "Other",
}


def load_results() -> dict:
    """Load the three per-model result JSONs into a single dict."""
    with open(OUTPUTS_DIR / "naive_bm25_results.json") as f:
        bm25 = json.load(f)
    with open(OUTPUTS_DIR / "classical_tfidf_logreg_results.json") as f:
        classical = json.load(f)
    with open(OUTPUTS_DIR / "dual_encoder_text_only_results.json") as f:
        dual = json.load(f)
    return {"BM25": bm25, "Classical": classical, "DualEncoder": dual}


def fig_overall_metrics(results: dict, out_path: Path) -> None:
    """Bar chart: four metrics × three models."""
    metrics = ["Recall@10", "Recall@50", "MRR", "nDCG@10"]
    bm25_vals = [results["BM25"]["overall"][m] for m in metrics]
    cls_vals = [results["Classical"]["overall"][m] for m in metrics]
    de_vals = [results["DualEncoder"]["overall"][m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.27

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(x - width, bm25_vals, width, label="BM25", color=MODEL_COLORS[0])
    ax.bar(x, cls_vals, width, label="Classical (TF-IDF+LR)", color=MODEL_COLORS[1])
    ax.bar(x + width, de_vals, width, label="Dual-Encoder (fine-tuned)", color=MODEL_COLORS[2])

    for i, (b, c, d) in enumerate(zip(bm25_vals, cls_vals, de_vals)):
        ax.text(i - width, b + 0.01, f"{b:.2f}", ha="center", fontsize=8)
        ax.text(i, c + 0.01, f"{c:.2f}", ha="center", fontsize=8)
        ax.text(i + width, d + 0.01, f"{d:.2f}", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_title("Overall retrieval performance on PAE-Bench")
    ax.set_ylim(0, 0.70)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def fig_per_vertical_recall10(results: dict, out_path: Path) -> None:
    """Grouped bar chart: Recall@10 by vertical for each model."""
    bm25_vals = [results["BM25"]["per_vertical"].get(v, {}).get("Recall@10", 0.0) for v in VERTICALS]
    cls_vals = [results["Classical"]["per_vertical"].get(v, {}).get("Recall@10", 0.0) for v in VERTICALS]
    de_vals = [results["DualEncoder"]["per_vertical"].get(v, {}).get("Recall@10", 0.0) for v in VERTICALS]

    x = np.arange(len(VERTICALS))
    width = 0.27

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(x - width, bm25_vals, width, label="BM25", color=MODEL_COLORS[0])
    ax.bar(x, cls_vals, width, label="Classical (TF-IDF+LR)", color=MODEL_COLORS[1])
    ax.bar(x + width, de_vals, width, label="Dual-Encoder (fine-tuned)", color=MODEL_COLORS[2])

    ax.set_xticks(x)
    ax.set_xticklabels([VERTICAL_LABELS[v] for v in VERTICALS])
    ax.set_ylabel("Recall@10")
    ax.set_title("Recall@10 by technology vertical")
    ax.set_ylim(0, 0.65)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def fig_training_curve(results: dict, out_path: Path) -> None:
    """Dual-encoder training vs validation loss."""
    history = results["DualEncoder"]["training_history"]
    epochs = [h["epoch"] for h in history]
    train = [h["train_loss"] for h in history]
    val = [h["val_loss"] for h in history]

    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(epochs, train, marker="o", label="Train loss", color="#3b82f6", linewidth=2)
    ax.plot(epochs, val, marker="s", label="Val loss", color="#ef4444", linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("InfoNCE loss")
    ax.set_title("Dual-encoder training curve")
    ax.set_xticks(epochs)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def fig_recall_curve(results: dict, out_path: Path) -> None:
    """Recall@K comparison showing K=10 vs K=50 for each model."""
    ks = [10, 50]
    fig, ax = plt.subplots(figsize=(6, 4))

    for name, color, label in [
        ("BM25", MODEL_COLORS[0], "BM25"),
        ("Classical", MODEL_COLORS[1], "Classical"),
        ("DualEncoder", MODEL_COLORS[2], "Dual-Encoder"),
    ]:
        vals = [results[name]["overall"]["Recall@10"], results[name]["overall"]["Recall@50"]]
        ax.plot(ks, vals, marker="o", label=label, color=color, linewidth=2)
        for k, v in zip(ks, vals):
            ax.annotate(f"{v:.2f}", (k, v), textcoords="offset points", xytext=(6, 6), fontsize=8)

    ax.set_xlabel("K (top-K candidates)")
    ax.set_ylabel("Recall@K")
    ax.set_title("Recall at K across retrieval models")
    ax.set_xticks(ks)
    ax.set_ylim(0, 0.70)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    results = load_results()

    figures = [
        ("overall_metrics.png", fig_overall_metrics),
        ("per_vertical_recall10.png", fig_per_vertical_recall10),
        ("training_curve.png", fig_training_curve),
        ("recall_vs_k.png", fig_recall_curve),
    ]

    for name, func in figures:
        out_path = FIG_DIR / name
        func(results, out_path)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
