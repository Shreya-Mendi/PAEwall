"""
Feature engineering pipeline.

Transforms raw patent, litigation, and product data into features
suitable for model training and evaluation.

Reads from data/raw/, outputs to data/processed/.
"""

from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


class ClaimParser:
    """Parses patent claims into structured elements."""

    def parse_claims(self, patent_text: str) -> list[dict]:
        """
        Decompose independent claims into structured elements:
        preamble, transitional phrase, and limitations.
        """
        # TODO: Rule-based parsing for claim structure
        # TODO: Fine-tuned claim-structure tagger for ambiguous cases
        raise NotImplementedError


class BenchmarkBuilder:
    """Constructs the PAE-Bench evaluation dataset."""

    def __init__(self, raw_dir: Path = RAW_DIR, output_dir: Path = PROCESSED_DIR):
        self.raw_dir = raw_dir
        self.output_dir = output_dir

    def build_retrieval_benchmark(self):
        """
        Build ~1,000 (patent, company, label) tuples from litigation records.
        Labels: litigated-infringement, licensed, uncontested-non-practiced, unrelated.
        """
        # TODO: Join litigation records to company product catalogs
        # TODO: Human verification on subset
        # TODO: Create per-vertical sub-splits
        raise NotImplementedError

    def build_faithfulness_benchmark(self):
        """
        Build ~1,000 (limitation, evidence, verdict) triples
        from actual litigation claim charts.
        """
        # TODO: Extract claim charts from PACER exhibits
        # TODO: Label as {supports, partially supports, does not support}
        raise NotImplementedError


class FeatureBuilder:
    """Builds features for the enforcement probability model."""

    def build_patent_features(self, patent_data: dict) -> dict:
        """
        Extract patent-level features: claim breadth, citation count,
        family size, examiner rejection history.
        """
        # TODO: Compute patent features from raw data
        raise NotImplementedError

    def build_defendant_features(self, company_data: dict) -> dict:
        """
        Extract defendant-level features: litigation history, revenue,
        prior settlements.
        """
        # TODO: Compute company features from EDGAR + litigation data
        raise NotImplementedError


def main():
    """Run the full feature engineering pipeline."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Building features from {RAW_DIR} -> {PROCESSED_DIR}")

    claim_parser = ClaimParser()
    benchmark_builder = BenchmarkBuilder()
    feature_builder = FeatureBuilder()

    # TODO: Orchestrate feature building
    # 1. Parse all patent claims
    # 2. Build PAE-Bench retrieval benchmark
    # 3. Build faithfulness benchmark
    # 4. Build enforcement probability features

    print("Feature building not yet implemented. See TODOs in build_features.py.")


if __name__ == "__main__":
    main()
