"""
Feature engineering pipeline.

Transforms raw patent, litigation, and product data into features
suitable for model training and evaluation.

Reads from data/raw/, outputs to data/processed/.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent))

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


class ClaimParser:
    """Parses patent claims into structured elements."""

    _TRANSITION_PHRASES = [
        "consisting essentially of", "consisting of", "comprising",
        "including", "having", "characterized by", "wherein",
    ]
    _DEPENDENT_RE = re.compile(r"\bclaim\s+\d+\b", re.IGNORECASE)

    def parse_claims(self, patent_text: str) -> list[dict]:
        """
        Decompose patent text into structured claim elements.

        Returns list of dicts: {claim_number, is_independent, preamble,
        transition, limitations, full_text}
        """
        claims_raw = self._extract_raw_claims(patent_text)
        parsed = []
        for raw in claims_raw:
            parsed.append(self._decompose(raw))
        return parsed

    def _extract_raw_claims(self, text: str) -> list[str]:
        section = re.split(r'\bCLAIMS?\b', text, maxsplit=1, flags=re.IGNORECASE)
        body = section[1] if len(section) > 1 else text
        parts = re.split(r'(?:^|\n)\s*(\d+)\.\s+', body)
        claims = []
        for i in range(1, len(parts) - 1, 2):
            num, content = parts[i].strip(), parts[i + 1].strip()
            if content:
                claims.append(f"{num}. {content}")
        return claims

    def _decompose(self, claim_text: str) -> dict:
        m = re.match(r'^(\d+)\.\s+', claim_text)
        claim_num = int(m.group(1)) if m else 0
        body = claim_text[m.end():] if m else claim_text
        is_independent = not bool(self._DEPENDENT_RE.search(claim_text))

        transition_pos, matched_phrase = None, None
        for phrase in self._TRANSITION_PHRASES:
            pos = body.lower().find(phrase)
            if pos != -1 and (transition_pos is None or pos < transition_pos):
                transition_pos, matched_phrase = pos, phrase

        preamble, transition, limitations = "", "", []
        if transition_pos is not None:
            preamble = body[:transition_pos].strip()
            transition = matched_phrase
            remainder = body[transition_pos + len(matched_phrase):].strip()
            for lim in re.split(r';\s*|\n+', remainder):
                lim = lim.strip().rstrip(';').strip()
                if lim:
                    limitations.append(lim)
        else:
            preamble = body

        return {
            "claim_number": claim_num,
            "is_independent": is_independent,
            "preamble": preamble,
            "transition": transition,
            "limitations": limitations,
            "full_text": claim_text,
        }


class BenchmarkBuilder:
    """Constructs the PAE-Bench evaluation dataset."""

    def __init__(self, raw_dir: Path = RAW_DIR, output_dir: Path = PROCESSED_DIR):
        self.raw_dir = raw_dir
        self.output_dir = output_dir

    def build_retrieval_benchmark(self) -> pd.DataFrame:
        """
        Build (patent_id, company_name, patent_claims, product_description, label) tuples.

        Positive labels come from litigation records.
        Negative labels are randomly sampled non-litigated pairs.
        """
        lit_path = self.raw_dir / "litigation_dockets.json"
        patents_path = self.raw_dir / "patents.json"
        products_path = self.raw_dir / "company_products.json"

        for p in [lit_path, patents_path, products_path]:
            if not p.exists():
                raise FileNotFoundError(f"Missing data file: {p}. Run make_dataset.py first.")

        with open(lit_path) as f:
            litigation = json.load(f)
        with open(patents_path) as f:
            patents = json.load(f)
        with open(products_path) as f:
            products = json.load(f)

        # Build indexes
        assignee_patents: dict[str, list] = {}
        for p in patents:
            key = p.get("assignee", "").lower().strip()
            if key:
                assignee_patents.setdefault(key, []).append(p)

        product_map: dict[str, dict] = {}
        for c in products:
            product_map[c["company_name"].lower()] = c

        def _find_patents(plaintiff: str) -> list:
            key = plaintiff.lower().strip()
            if key in assignee_patents:
                return assignee_patents[key]
            for k, v in assignee_patents.items():
                if key in k or k in key:
                    return v
            return []

        def _find_product(defendant: str) -> dict:
            key = defendant.lower().strip()
            if key in product_map:
                return product_map[key]
            for k, v in product_map.items():
                if key in k or k in key:
                    return v
            return {}

        rows = []
        for lit in litigation:
            plaintiff = lit.get("plaintiff", "")
            defendant = lit.get("defendant", "")
            if not plaintiff or not defendant:
                continue
            patent_list = _find_patents(plaintiff)
            product = _find_product(defendant)
            if not product:
                continue
            for pat in patent_list:
                claims_text = " ".join(c.get("claim_text", "") for c in pat.get("claims", []))
                rows.append({
                    "patent_id": pat["patent_number"],
                    "patent_claims": claims_text,
                    "company_name": defendant,
                    "product_description": product.get("business_description", ""),
                    "label": "litigated_infringement",
                    "label_binary": 1,
                })

        df_pos = pd.DataFrame(rows)
        if df_pos.empty:
            logger.warning("No positive pairs assembled. Check data quality.")
            return df_pos

        # Sample negatives (3:1 ratio)
        all_companies = df_pos[["company_name", "product_description"]].drop_duplicates("company_name")
        rng = np.random.default_rng(seed=42)
        negatives = []
        for _, row in df_pos.iterrows():
            sued = set(df_pos.loc[df_pos["patent_id"] == row["patent_id"], "company_name"])
            candidates = all_companies[~all_companies["company_name"].isin(sued)]
            if candidates.empty:
                continue
            sample = candidates.sample(n=min(3, len(candidates)), random_state=int(rng.integers(0, 9999)))
            for _, neg in sample.iterrows():
                negatives.append({
                    "patent_id": row["patent_id"],
                    "patent_claims": row["patent_claims"],
                    "company_name": neg["company_name"],
                    "product_description": neg["product_description"],
                    "label": "non_infringement",
                    "label_binary": 0,
                })

        df = pd.concat([df_pos, pd.DataFrame(negatives)], ignore_index=True).sample(frac=1, random_state=42)
        logger.info(f"Retrieval benchmark: {len(df_pos)} positives + {len(negatives)} negatives")
        return df

    def build_faithfulness_benchmark(self) -> pd.DataFrame:
        """
        Build (limitation, evidence, verdict) triples from claim charts.

        Verdict labels: supports | partially_supports | does_not_support.
        Requires existing retrieval benchmark.
        """
        bench_path = self.output_dir / "pae_bench.parquet"
        if not bench_path.exists():
            raise FileNotFoundError("Run make_dataset.py --assemble first.")

        df = pd.read_parquet(bench_path)
        parser = ClaimParser()
        rows = []
        for _, row in df.iterrows():
            claims = parser.parse_claims(row.get("patent_claims", ""))
            for claim in claims:
                if not claim["is_independent"]:
                    continue
                for lim in claim["limitations"]:
                    rows.append({
                        "patent_id": row["patent_id"],
                        "company_name": row["company_name"],
                        "limitation": lim,
                        "evidence": row.get("product_description", "")[:1000],
                        "verdict": "unknown",
                    })

        logger.info(f"Faithfulness benchmark: {len(rows)} (limitation, evidence) pairs (auto-labeled)")
        return pd.DataFrame(rows)


class FeatureBuilder:
    """Builds features for the enforcement probability model."""

    def build_patent_features(self, patent_data: dict) -> dict:
        """Extract patent-level features for enforcement probability prediction."""
        claims_text = " ".join(c.get("claim_text", "") for c in patent_data.get("claims", []))
        num_claims = len(patent_data.get("claims", []))
        cpc_codes = [c.get("cpc_subclass", "") for c in patent_data.get("cpc_codes", [])]

        # Claim breadth proxy: avg words per independent claim
        words_per_claim = len(claims_text.split()) / max(num_claims, 1)

        return {
            "patent_id": patent_data.get("patent_number", ""),
            "num_claims": num_claims,
            "words_per_claim": round(words_per_claim, 1),
            "citation_count": patent_data.get("metadata", {}).get("citation_count", 0),
            "num_cpc_codes": len(cpc_codes),
            "has_system_claim": int(any("system" in c.get("claim_text", "").lower() for c in patent_data.get("claims", []))),
            "has_method_claim": int(any("method" in c.get("claim_text", "").lower() for c in patent_data.get("claims", []))),
        }

    def build_defendant_features(self, company_data: dict) -> dict:
        """Extract defendant-level features for enforcement probability prediction."""
        desc = company_data.get("business_description", "")
        return {
            "company_name": company_data.get("company_name", ""),
            "sic_code": company_data.get("sic", ""),
            "sic_description": company_data.get("sic_description", ""),
            "description_length": len(desc.split()),
            "has_10k": int(bool(company_data.get("filing_date"))),
            "is_publicly_traded": int(bool(company_data.get("tickers"))),
        }


def main():
    """Run the full feature engineering pipeline."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Building features: {RAW_DIR} -> {PROCESSED_DIR}")

    builder = BenchmarkBuilder()
    feature_builder = FeatureBuilder()
    claim_parser = ClaimParser()

    # Build retrieval benchmark (train/test pairs)
    logger.info("Building retrieval benchmark...")
    try:
        retrieval_df = builder.build_retrieval_benchmark()
        if not retrieval_df.empty:
            out = PROCESSED_DIR / "retrieval_benchmark.parquet"
            retrieval_df.to_parquet(out, index=False)
            logger.info(f"Saved retrieval benchmark: {len(retrieval_df)} rows -> {out}")
    except FileNotFoundError as e:
        logger.error(f"Skipping retrieval benchmark: {e}")

    # Build faithfulness benchmark
    logger.info("Building faithfulness benchmark...")
    try:
        faith_df = builder.build_faithfulness_benchmark()
        if not faith_df.empty:
            out = PROCESSED_DIR / "faithfulness_benchmark.parquet"
            faith_df.to_parquet(out, index=False)
            logger.info(f"Saved faithfulness benchmark: {len(faith_df)} rows -> {out}")
    except FileNotFoundError as e:
        logger.error(f"Skipping faithfulness benchmark: {e}")

    # Build patent + company features from raw data
    logger.info("Building enforcement probability features...")
    try:
        with open(RAW_DIR / "patents.json") as f:
            patents = json.load(f)
        with open(RAW_DIR / "company_products.json") as f:
            companies = json.load(f)

        patent_feats = [feature_builder.build_patent_features(p) for p in patents]
        company_feats = [feature_builder.build_defendant_features(c) for c in companies]

        pd.DataFrame(patent_feats).to_parquet(PROCESSED_DIR / "patent_features.parquet", index=False)
        pd.DataFrame(company_feats).to_parquet(PROCESSED_DIR / "company_features.parquet", index=False)
        logger.info(f"Saved {len(patent_feats)} patent features, {len(company_feats)} company features")
    except FileNotFoundError as e:
        logger.error(f"Skipping enforcement features: {e}")

    logger.info("Feature engineering complete.")


if __name__ == "__main__":
    main()
