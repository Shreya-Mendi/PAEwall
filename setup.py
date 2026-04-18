"""
PAEwall — Setup Script
Downloads data, builds features, and trains models.

Usage:
    python setup.py --all          # Run full pipeline
    python setup.py --data         # Download and prepare data only
    python setup.py --features     # Build features only
    python setup.py --train        # Train all models only
    python setup.py --train-naive  # Train naive baseline only
    python setup.py --train-classical  # Train classical model only
    python setup.py --train-dl     # Train deep learning model only
"""

import argparse
import sys


def download_data():
    """Download raw patent and product data from public sources."""
    from scripts.make_dataset import main as make_dataset
    make_dataset()


def build_features():
    """Run feature engineering pipeline."""
    from scripts.build_features import main as build_feat
    build_feat()


def train_naive():
    """Train naive baseline (BM25 retrieval)."""
    from scripts.train_naive import main as train
    train()


def train_classical():
    """Train classical ML model (TF-IDF + logistic regression)."""
    from scripts.train_classical import main as train
    train()


def train_deep_learning():
    """Train deep learning model (dual-encoder)."""
    from scripts.train_deep_learning import main as train
    train()


def train_all():
    """Train all three models."""
    print("=" * 60)
    print("Training naive baseline (BM25)")
    print("=" * 60)
    train_naive()

    print("\n" + "=" * 60)
    print("Training classical model (TF-IDF + Logistic Regression)")
    print("=" * 60)
    train_classical()

    print("\n" + "=" * 60)
    print("Training deep learning model (Dual-Encoder)")
    print("=" * 60)
    train_deep_learning()


def main():
    parser = argparse.ArgumentParser(description="PAEwall setup and training pipeline")
    parser.add_argument("--all", action="store_true", help="Run full pipeline")
    parser.add_argument("--data", action="store_true", help="Download data only")
    parser.add_argument("--features", action="store_true", help="Build features only")
    parser.add_argument("--train", action="store_true", help="Train all models")
    parser.add_argument("--train-naive", action="store_true", help="Train naive baseline")
    parser.add_argument("--train-classical", action="store_true", help="Train classical model")
    parser.add_argument("--train-dl", action="store_true", help="Train deep learning model")
    args = parser.parse_args()

    if args.all:
        download_data()
        build_features()
        train_all()
    elif args.data:
        download_data()
    elif args.features:
        build_features()
    elif args.train:
        train_all()
    elif args.train_naive:
        train_naive()
    elif args.train_classical:
        train_classical()
    elif args.train_dl:
        train_deep_learning()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
