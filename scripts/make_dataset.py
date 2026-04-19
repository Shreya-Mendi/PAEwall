"""
Data collection pipeline for PAEwall.

Collects three data streams and joins them into PAE-Bench:
1. Patent infringement litigation records (CourtListener RECAP)
2. Patent full text, claims, CPC codes (Google Patents BigQuery)
3. Company product descriptions (SEC EDGAR)

Usage:
    python scripts/make_dataset.py --all
    python scripts/make_dataset.py --litigation
    python scripts/make_dataset.py --patents
    python scripts/make_dataset.py --products
    python scripts/make_dataset.py --assemble
"""

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests
from loguru import logger

from config import (
    COURTLISTENER_BASE_URL,
    COURTLISTENER_RATE_LIMIT,
    COURTLISTENER_TOKEN,
    EDGAR_EFTS_URL,
    EDGAR_RATE_LIMIT,
    EDGAR_SUBMISSIONS_URL,
    EDGAR_USER_AGENT,
    GCP_PROJECT_ID,
    PATENT_NOS_CODE,
    PATENTS_BQ_TABLE,
    PROCESSED_DIR,
    RAW_DIR,
)

# ---------------------------------------------------------------------------
# 1. CourtListener — Patent infringement litigation records
# ---------------------------------------------------------------------------


class LitigationCollector:
    """
    Pulls patent infringement dockets from CourtListener RECAP.

    Extracts (patent_number, defendant_company, court, outcome) tuples
    that form the positive labels for PAE-Bench.
    """

    def __init__(self, token: str = COURTLISTENER_TOKEN):
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Token {token}",
        })
        self.base = COURTLISTENER_BASE_URL

    def fetch_patent_dockets(self, page_limit: int = 50) -> list[dict]:
        """
        Fetch patent infringement cases using the CourtListener search endpoint.

        Uses type=r (RECAP/PACER) with nature_of_suit=830 (patent).
        The /dockets/ endpoint requires special PACER access; /search/ is open.

        Returns list of docket dicts with case_name, parties, court, date, etc.
        """
        dockets = []
        url = f"{self.base}/search/"
        params = {
            "type": "r",
            "nature_of_suit": PATENT_NOS_CODE,
            "order_by": "dateFiled desc",
            "page_size": 20,
        }

        for page in range(1, page_limit + 1):
            logger.info(f"Fetching patent cases page {page}/{page_limit}")
            resp = self.session.get(url, params=params, timeout=30)

            if resp.status_code == 401:
                logger.error("CourtListener auth failed. Set COURTLISTENER_TOKEN in .env")
                break
            if resp.status_code == 429:
                logger.warning("Rate limited. Sleeping 60s...")
                time.sleep(60)
                continue
            if resp.status_code != 200:
                logger.error(f"HTTP {resp.status_code}: {resp.text[:200]}")
                break

            data = resp.json()
            results = data.get("results", [])
            if not results:
                break

            for r in results:
                dockets.append({
                    "docket_id": r.get("docket_id"),
                    "case_name": r.get("caseName", ""),
                    "court": r.get("court", ""),
                    "date_filed": r.get("dateFiled", ""),
                    "date_terminated": r.get("dateTerminated", ""),
                    "cause": r.get("cause", ""),
                    "docket_number": r.get("docketNumber", ""),
                    "parties": r.get("party", []),
                    "docket_url": r.get("docket_absolute_url", ""),
                })

            # Follow cursor-based pagination
            next_url = data.get("next")
            if next_url:
                url = next_url
                params = {}
            else:
                break

            time.sleep(COURTLISTENER_RATE_LIMIT)

        logger.info(f"Collected {len(dockets)} patent cases")
        return dockets

    def extract_patent_numbers(self, case_name: str, cause: str) -> list[str]:
        """
        Extract US patent numbers from case name and cause fields.
        Patent numbers are typically 7-8 digit numbers, sometimes prefixed with US.
        """
        text = f"{case_name} {cause}"
        # Match patterns like: US7,654,321 or 7654321 or US 7,654,321
        patterns = [
            r"US\s*[\d,]{7,12}",
            r"Patent\s+(?:No\.?\s*)?[\d,]{7,12}",
            r"'\d{3}\s+patent",
            r"\b\d{1,2},\d{3},\d{3}\b",
        ]
        numbers = set()
        for pat in patterns:
            for match in re.findall(pat, text, re.IGNORECASE):
                digits = re.sub(r"[^\d]", "", match)
                if 6 <= len(digits) <= 8:
                    numbers.add(digits)
        return list(numbers)

    def extract_plaintiff_defendant(self, case_name: str) -> tuple[str, str]:
        """
        Extract plaintiff and defendant from case name.
        Standard format: "Plaintiff v. Defendant"
        The plaintiff is the patent asserter (PAE); defendant is the accused infringer.
        """
        parts = re.split(r"\s+v\.?\s+", case_name, maxsplit=1)
        if len(parts) == 2:
            plaintiff = re.split(r"\s*\(", parts[0].strip())[0].strip()
            defendant = re.split(r"\s*\(", parts[1].strip())[0].strip()
            return plaintiff, defendant
        return "", ""

    def collect(self, output_dir: Path = RAW_DIR, page_limit: int = 50) -> Path:
        """Run full litigation collection and save to JSON."""
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "litigation_dockets.json"

        dockets = self.fetch_patent_dockets(page_limit=page_limit)

        # Enrich with plaintiff + defendant from case name
        enriched = []
        for d in dockets:
            plaintiff, defendant = self.extract_plaintiff_defendant(d["case_name"])
            # Skip noise: cases with no recognizable company on either side
            if plaintiff and defendant and len(defendant) > 2:
                enriched.append({
                    **d,
                    "plaintiff": plaintiff,
                    "defendant": defendant,
                })

        logger.info(f"Enriched {len(enriched)}/{len(dockets)} dockets with plaintiff + defendant")

        with open(out_path, "w") as f:
            json.dump(enriched, f, indent=2)

        logger.info(f"Saved litigation data to {out_path}")
        return out_path


# ---------------------------------------------------------------------------
# 2. Google Patents BigQuery — Patent claims, CPC codes, metadata
# ---------------------------------------------------------------------------


class PatentCollector:
    """
    Pulls patent details from Google Patents Public Dataset on BigQuery.

    Free tier: 1TB/month of queries. A single batch query for hundreds
    of patents uses only a few GB — much faster than REST-per-patent.

    For each patent number found in litigation records, fetches:
    claims text, abstract, CPC codes, citation counts, priority date.
    """

    def __init__(self, project_id: str = GCP_PROJECT_ID):
        from google.cloud import bigquery
        self.client = bigquery.Client(project=project_id)
        self.table = PATENTS_BQ_TABLE

    def fetch_patents_by_assignees(self, assignee_names: list[str]) -> pd.DataFrame:
        """
        Fetch granted US patents for a list of assignee companies (PAEs/plaintiffs).

        Uses BigQuery assignee_harmonized array to match by company name.
        Returns DataFrame with patent details, claims, and CPC codes.
        """
        # Build LIKE conditions for each assignee
        conditions = " OR ".join(
            f"LOWER(a.name) LIKE '%{name.lower().replace(chr(39), '')}%'"
            for name in assignee_names
            if name.strip()
        )

        query = f"""
        SELECT
            publication_number,
            (SELECT text FROM UNNEST(title_localized) WHERE language = 'en' LIMIT 1) AS title,
            (SELECT text FROM UNNEST(abstract_localized) WHERE language = 'en' LIMIT 1) AS abstract,
            (SELECT text FROM UNNEST(claims_localized) WHERE language = 'en' LIMIT 1) AS claims_text,
            (SELECT name FROM UNNEST(assignee_harmonized) LIMIT 1) AS assignee,
            priority_date,
            filing_date,
            grant_date,
            ARRAY_TO_STRING(
                ARRAY(SELECT code FROM UNNEST(cpc) WHERE code IS NOT NULL), '|'
            ) AS cpc_all,
            ARRAY_LENGTH(citation) AS citation_count
        FROM `{self.table}`
        WHERE
            country_code = 'US'
            AND grant_date IS NOT NULL
            AND EXISTS (
                SELECT 1 FROM UNNEST(assignee_harmonized) a
                WHERE {conditions}
            )
        LIMIT 2000
        """

        logger.info(f"Running BigQuery for {len(assignee_names)} assignees...")
        df = self.client.query(query).to_dataframe()
        logger.info(f"BigQuery returned {len(df)} patents")
        return df

    def collect(self, plaintiff_companies: list[str], output_dir: Path = RAW_DIR) -> Path:
        """
        Fetch patents for plaintiff companies (PAEs) found in litigation data.

        Args:
            plaintiff_companies: List of company names that appear as plaintiffs.
            output_dir: Where to save the output JSON.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "patents.json"

        unique_plaintiffs = list(set(plaintiff_companies))
        logger.info(f"Fetching patents for {len(unique_plaintiffs)} plaintiff companies from BigQuery")

        # Query in chunks of 50 assignees to keep query size manageable
        all_dfs = []
        chunk_size = 50
        for i in range(0, len(unique_plaintiffs), chunk_size):
            chunk = unique_plaintiffs[i : i + chunk_size]
            df = self.fetch_patents_by_assignees(chunk)
            all_dfs.append(df)

        combined = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
        logger.info(f"Fetched {len(combined)} patents total")

        patents = []
        for _, row in combined.iterrows():
            patents.append({
                "patent_number": str(row["publication_number"]),
                "assignee": str(row.get("assignee", "")),
                "metadata": {
                    "patent_title": str(row.get("title", "") or ""),
                    "patent_abstract": str(row.get("abstract", "") or ""),
                    "patent_date": str(row.get("grant_date", "")),
                    "priority_date": str(row.get("priority_date", "")),
                    "filing_date": str(row.get("filing_date", "")),
                    "citation_count": int(row.get("citation_count", 0) or 0),
                },
                "claims": [{"claim_text": str(row.get("claims_text", "") or "")}],
                "cpc_codes": [
                    {"cpc_subclass": code[:4]}  # first 4 chars = subclass
                    for code in str(row.get("cpc_all", "")).split("|")
                    if code
                ],
            })

        with open(out_path, "w") as f:
            json.dump(patents, f, indent=2)

        logger.info(f"Saved {len(patents)} patents to {out_path}")
        return out_path


# ---------------------------------------------------------------------------
# 3. SEC EDGAR — Company product descriptions from 10-K filings
# ---------------------------------------------------------------------------


class ProductCollector:
    """
    Pulls product descriptions from SEC EDGAR 10-K filings.

    For each defendant company, searches EDGAR for their most recent 10-K
    and extracts the business description (Item 1).
    """

    def __init__(self, user_agent: str = EDGAR_USER_AGENT):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
        })

    def search_company(self, company_name: str) -> dict | None:
        """
        Search EDGAR for a company by name. Returns CIK and basic info.
        Uses the company search endpoint.
        """
        url = "https://efts.sec.gov/LATEST/search-index"
        params = {
            "q": f'"{company_name}"',
            "forms": "10-K",
            "dateRange": "custom",
            "startdt": "2020-01-01",
            "enddt": "2026-12-31",
            "size": 1,
        }

        resp = self.session.get(url, params=params)
        if resp.status_code != 200:
            logger.warning(f"EDGAR search failed for '{company_name}': HTTP {resp.status_code}")
            return None

        data = resp.json()
        hits = data.get("hits", {}).get("hits", [])
        if not hits:
            return None

        hit = hits[0]["_source"]
        return {
            "company_name": hit.get("entity_name", company_name),
            "cik": hit.get("entity_id", ""),
            "accession_number": hit.get("file_num", ""),
            "filing_date": hit.get("file_date", ""),
            "form_type": hit.get("form_type", ""),
        }

    def fetch_company_submissions(self, cik: str) -> dict | None:
        """
        Fetch company filing history from data.sec.gov submissions endpoint.
        Returns the most recent 10-K filing info.
        """
        # CIK must be zero-padded to 10 digits
        cik_padded = cik.zfill(10)
        url = f"{EDGAR_SUBMISSIONS_URL}/CIK{cik_padded}.json"

        resp = self.session.get(url)
        if resp.status_code != 200:
            logger.warning(f"EDGAR submissions failed for CIK {cik}: HTTP {resp.status_code}")
            return None

        data = resp.json()

        # Find most recent 10-K
        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        accession_numbers = recent.get("accessionNumber", [])
        filing_dates = recent.get("filingDate", [])
        primary_docs = recent.get("primaryDocument", [])

        for i, form in enumerate(forms):
            if form == "10-K":
                return {
                    "company_name": data.get("name", ""),
                    "cik": cik,
                    "accession_number": accession_numbers[i],
                    "filing_date": filing_dates[i],
                    "primary_document": primary_docs[i],
                    "sic": data.get("sic", ""),
                    "sic_description": data.get("sicDescription", ""),
                    "tickers": data.get("tickers", []),
                }

        return None

    def fetch_filing_text(self, cik: str, accession_number: str, primary_doc: str) -> str:
        """Download the actual 10-K filing text."""
        cik_padded = cik.zfill(10)
        accession_clean = accession_number.replace("-", "")
        url = f"https://www.sec.gov/Archives/edgar/data/{cik_padded}/{accession_clean}/{primary_doc}"

        resp = self.session.get(url)
        if resp.status_code != 200:
            logger.warning(f"Could not fetch filing text: HTTP {resp.status_code}")
            return ""

        return resp.text

    def extract_business_description(self, filing_html: str) -> str:
        """
        Extract Item 1 (Business Description) from a 10-K filing.
        This section typically contains product descriptions.
        """
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(filing_html, "lxml")
        text = soup.get_text(separator="\n", strip=True)

        # Find Item 1 section — look for "Item 1" header, stop at "Item 1A"
        item1_pattern = re.compile(
            r"(?:ITEM\s*1[\.\s]*(?:BUSINESS|Business))(.*?)(?:ITEM\s*1A)",
            re.DOTALL | re.IGNORECASE,
        )
        match = item1_pattern.search(text)
        if match:
            description = match.group(1).strip()
            # Truncate to reasonable length (first ~5000 chars)
            return description[:5000]

        return ""

    def collect(self, company_names: list[str], output_dir: Path = RAW_DIR) -> Path:
        """Fetch product descriptions for a list of companies and save."""
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "company_products.json"

        companies = []
        unique_names = list(set(company_names))
        logger.info(f"Fetching product data for {len(unique_names)} companies from EDGAR")

        for i, name in enumerate(unique_names):
            if i > 0 and i % 10 == 0:
                logger.info(f"  Progress: {i}/{len(unique_names)}")

            # Step 1: Search for company
            search_result = self.search_company(name)
            time.sleep(EDGAR_RATE_LIMIT)

            if not search_result or not search_result.get("cik"):
                logger.debug(f"No EDGAR results for '{name}'")
                continue

            cik = search_result["cik"]

            # Step 2: Get most recent 10-K
            submission = self.fetch_company_submissions(cik)
            time.sleep(EDGAR_RATE_LIMIT)

            if not submission:
                logger.debug(f"No 10-K found for '{name}' (CIK: {cik})")
                continue

            # Step 3: Fetch filing text and extract business description
            filing_text = self.fetch_filing_text(
                cik, submission["accession_number"], submission["primary_document"]
            )
            time.sleep(EDGAR_RATE_LIMIT)

            description = self.extract_business_description(filing_text) if filing_text else ""

            companies.append({
                "company_name": name,
                "edgar_name": submission.get("company_name", ""),
                "cik": cik,
                "sic": submission.get("sic", ""),
                "sic_description": submission.get("sic_description", ""),
                "tickers": submission.get("tickers", []),
                "filing_date": submission.get("filing_date", ""),
                "business_description": description,
            })

        logger.info(f"Fetched product data for {len(companies)}/{len(unique_names)} companies")

        with open(out_path, "w") as f:
            json.dump(companies, f, indent=2)

        logger.info(f"Saved company product data to {out_path}")
        return out_path


# ---------------------------------------------------------------------------
# 4. Assemble PAE-Bench
# ---------------------------------------------------------------------------


def assemble_benchmark(raw_dir: Path = RAW_DIR, output_dir: Path = PROCESSED_DIR) -> Path:
    """
    Join litigation, patent, and product data into PAE-Bench.

    Output: a DataFrame of (patent_id, patent_claims, company_name,
    product_description, label, vertical) tuples saved as parquet.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "pae_bench.parquet"

    # Load raw data
    with open(raw_dir / "litigation_dockets.json") as f:
        litigation = json.load(f)
    with open(raw_dir / "patents.json") as f:
        patents_raw = json.load(f)
    with open(raw_dir / "company_products.json") as f:
        products_raw = json.load(f)

    # Index patents and products by key
    patent_map = {}
    for p in patents_raw:
        pn = p["patent_number"]
        claims_text = " ".join(c.get("claim_text", "") for c in p.get("claims", []))
        cpc_codes = [c.get("cpc_subclass", "") for c in p.get("cpc_codes", [])]
        patent_map[pn] = {
            "title": p.get("metadata", {}).get("patent_title", ""),
            "abstract": p.get("metadata", {}).get("patent_abstract", ""),
            "claims_text": claims_text,
            "cpc_codes": cpc_codes,
            "date": p.get("metadata", {}).get("patent_date", ""),
            "num_citations": p.get("metadata", {}).get("patent_num_cited_by_us_patents", 0),
        }

    product_map = {}
    for c in products_raw:
        product_map[c["company_name"].lower()] = {
            "edgar_name": c.get("edgar_name", ""),
            "sic": c.get("sic", ""),
            "sic_description": c.get("sic_description", ""),
            "business_description": c.get("business_description", ""),
        }

    # Map CPC subclass to vertical
    def cpc_to_vertical(cpc_codes: list[str]) -> str:
        for code in cpc_codes:
            if code.startswith("G16H"):
                return "medical_devices"
            if code.startswith("G06"):
                return "software"
            if code.startswith("H04"):
                return "consumer_electronics"
            if code.startswith("G06Q"):
                return "fintech"
            if code.startswith("A61"):
                return "medical_devices"
            if code.startswith("B") or code.startswith("F"):
                return "industrial"
        return "other"

    # Build assignee -> patents lookup
    assignee_to_patents: dict[str, list] = {}
    for p in patents_raw:
        assignee = p.get("assignee", "").lower().strip()
        if assignee:
            assignee_to_patents.setdefault(assignee, []).append(p)

    def find_patents_for_plaintiff(plaintiff: str) -> list:
        """Fuzzy match plaintiff name to assignee in patent data."""
        plaintiff_lower = plaintiff.lower().strip()
        # Exact match
        if plaintiff_lower in assignee_to_patents:
            return assignee_to_patents[plaintiff_lower]
        # Substring match
        for key, patents in assignee_to_patents.items():
            if plaintiff_lower in key or key in plaintiff_lower:
                return patents
        return []

    def find_product_for_defendant(defendant: str) -> dict:
        """Fuzzy match defendant name to product corpus."""
        defendant_lower = defendant.lower().strip()
        if defendant_lower in product_map:
            return product_map[defendant_lower]
        for key, val in product_map.items():
            if defendant_lower in key or key in defendant_lower:
                return val
        return {"business_description": "", "edgar_name": "", "sic": "", "sic_description": ""}

    # Join: each litigation docket -> plaintiff patents x defendant product
    rows = []
    for lit in litigation:
        plaintiff = lit.get("plaintiff", "")
        defendant = lit.get("defendant", "")
        if not plaintiff or not defendant:
            continue

        plaintiff_patents = find_patents_for_plaintiff(plaintiff)
        product_info = find_product_for_defendant(defendant)

        for p in plaintiff_patents:
            patent_info = p.get("metadata", {})
            claims_text = " ".join(c.get("claim_text", "") for c in p.get("claims", []))
            cpc_codes = [c.get("cpc_subclass", "") for c in p.get("cpc_codes", [])]

            rows.append({
                "patent_id": p["patent_number"],
                "patent_title": patent_info.get("patent_title", ""),
                "patent_abstract": patent_info.get("patent_abstract", ""),
                "patent_claims": claims_text,
                "patent_date": patent_info.get("patent_date", ""),
                "patent_cpc_codes": "|".join(cpc_codes),
                "vertical": cpc_to_vertical(cpc_codes),
                "num_citations": patent_info.get("citation_count", 0),
                "plaintiff": plaintiff,
                "company_name": defendant,
                "product_description": product_info["business_description"],
                "sic": product_info["sic"],
                "sic_description": product_info["sic_description"],
                "label": "litigated_infringement",
                "case_name": lit["case_name"],
                "court": lit["court"],
                "date_filed": lit["date_filed"],
            })

    df = pd.DataFrame(rows)
    logger.info(f"Assembled PAE-Bench: {len(df)} rows, {df['vertical'].nunique()} verticals")
    logger.info(f"Vertical distribution:\n{df['vertical'].value_counts().to_string()}")

    df.to_parquet(out_path, index=False)
    logger.info(f"Saved PAE-Bench to {out_path}")

    # Also save a CSV for easy inspection
    csv_path = output_dir / "pae_bench.csv"
    df.to_csv(csv_path, index=False)

    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="PAEwall data collection pipeline")
    parser.add_argument("--all", action="store_true", help="Run full pipeline")
    parser.add_argument("--litigation", action="store_true", help="Collect litigation data only")
    parser.add_argument("--patents", action="store_true", help="Collect patent data only")
    parser.add_argument("--products", action="store_true", help="Collect product data only")
    parser.add_argument("--assemble", action="store_true", help="Assemble PAE-Bench only")
    parser.add_argument("--pages", type=int, default=50, help="Max pages of litigation dockets")
    args = parser.parse_args()

    if args.litigation or args.all:
        logger.info("=== Step 1: Collecting litigation records ===")
        lit_collector = LitigationCollector()
        lit_collector.collect(page_limit=args.pages)

    if args.patents or args.all:
        logger.info("=== Step 2: Collecting patent data from BigQuery ===")
        with open(RAW_DIR / "litigation_dockets.json") as f:
            litigation = json.load(f)
        # Extract plaintiff companies (index 0 in parties = plaintiff)
        plaintiffs = list({
            d["parties"][0].strip()
            for d in litigation
            if d.get("parties") and len(d["parties"]) > 0
        })
        plaintiffs = list({d["plaintiff"] for d in litigation if d.get("plaintiff")})
        logger.info(f"Found {len(plaintiffs)} unique plaintiff companies")
        patent_collector = PatentCollector()
        patent_collector.collect(plaintiffs)

    if args.products or args.all:
        logger.info("=== Step 3: Collecting product data ===")
        with open(RAW_DIR / "litigation_dockets.json") as f:
            litigation = json.load(f)
        defendants = list({d["defendant"] for d in litigation if d.get("defendant")})
        product_collector = ProductCollector()
        product_collector.collect(defendants)

    if args.assemble or args.all:
        logger.info("=== Step 4: Assembling PAE-Bench ===")
        assemble_benchmark()

    if not any([args.all, args.litigation, args.patents, args.products, args.assemble]):
        parser.print_help()


if __name__ == "__main__":
    main()
