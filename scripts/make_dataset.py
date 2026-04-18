"""
Data collection and preparation pipeline.

Downloads patent data from USPTO/Google Patents BigQuery, litigation records
from CourtListener/PACER, and product descriptions from SEC EDGAR and public web.

Outputs raw data to data/raw/.
"""

from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


class PatentDataCollector:
    """Collects patent data from USPTO PatentsView and Google Patents BigQuery."""

    def __init__(self, output_dir: Path = RAW_DIR):
        self.output_dir = output_dir

    def fetch_patents(self, cpc_subclasses: list[str] | None = None):
        """Fetch patent full text, claims, CPC codes, and citations."""
        # TODO: Implement BigQuery query for Google Patents Public Dataset
        # TODO: Fallback to PatentsView API
        raise NotImplementedError

    def fetch_patent_figures(self, patent_ids: list[str]):
        """Download patent figure images from USPTO."""
        # TODO: Implement USPTO figure download
        raise NotImplementedError


class LitigationDataCollector:
    """Collects litigation records from CourtListener RECAP and PTAB."""

    def __init__(self, output_dir: Path = RAW_DIR):
        self.output_dir = output_dir

    def fetch_infringement_cases(self):
        """Fetch patent infringement complaints and claim charts from PACER/RECAP."""
        # TODO: Query CourtListener API for patent cases
        # TODO: Extract (patent, defendant company, outcome) tuples
        raise NotImplementedError

    def fetch_ptab_outcomes(self):
        """Fetch Inter Partes Review outcomes from PTAB Open Data."""
        # TODO: Download PTAB dataset from developer.uspto.gov
        raise NotImplementedError


class ProductDataCollector:
    """Collects product descriptions from SEC EDGAR and company websites."""

    def __init__(self, output_dir: Path = RAW_DIR):
        self.output_dir = output_dir

    def fetch_edgar_filings(self, company_ciks: list[str]):
        """Extract product descriptions from 10-K filings."""
        # TODO: Query EDGAR XBRL API for 10-K filings
        # TODO: Parse product description sections
        raise NotImplementedError

    def scrape_product_pages(self, company_urls: list[str]):
        """Scrape product feature pages and screenshots."""
        # TODO: Implement targeted web scraper
        # TODO: Capture product UI screenshots via Selenium
        raise NotImplementedError


def main():
    """Run the full data collection pipeline."""
    raw_dir = RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"Collecting data to {raw_dir}")

    patent_collector = PatentDataCollector(raw_dir)
    litigation_collector = LitigationDataCollector(raw_dir)
    product_collector = ProductDataCollector(raw_dir)

    # TODO: Orchestrate data collection
    # 1. Fetch patents
    # 2. Fetch litigation records to get (patent, company) positive pairs
    # 3. Fetch product data for companies in litigation records
    # 4. Save all to data/raw/

    print("Data collection not yet implemented. See TODOs in make_dataset.py.")


if __name__ == "__main__":
    main()
