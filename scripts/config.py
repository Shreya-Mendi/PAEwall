"""
Configuration and API credentials for data collection.

API keys are loaded from environment variables or a .env file.
Copy .env.example to .env and fill in your keys.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# --- Directories ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "data" / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"

# --- CourtListener API ---
# Get a free token at: https://www.courtlistener.com/sign-in/
COURTLISTENER_TOKEN = os.getenv("COURTLISTENER_TOKEN", "")
COURTLISTENER_BASE_URL = "https://www.courtlistener.com/api/rest/v4"

# Patent infringement nature-of-suit code
PATENT_NOS_CODE = "830"

# --- Google Patents BigQuery ---
# Free: 1TB/month queries. Requires a GCP project (free tier works).
# Set up: https://cloud.google.com/bigquery/docs/quickstarts
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "")
PATENTS_BQ_TABLE = "patents-public-data.patents.publications"

# --- SEC EDGAR ---
# No API key needed, but must include User-Agent
EDGAR_USER_AGENT = os.getenv("EDGAR_USER_AGENT", "PAEwall shreya.mendi@duke.edu")
EDGAR_EFTS_URL = "https://efts.sec.gov/LATEST/search-index"
EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions"

# --- Rate limits ---
COURTLISTENER_RATE_LIMIT = 5.0   # seconds between requests
EDGAR_RATE_LIMIT = 0.15          # SEC asks for 10 req/sec max
