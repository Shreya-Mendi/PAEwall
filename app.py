"""
PAEwall — Interactive Patent Infringement Discovery Application

FastAPI backend serving the PAEwall web interface.
Runs inference only (no training).

Usage:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

import json
import os
import re
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from claim_chart.generator import ClaimChart, ClaimChartGenerator, FaithfulnessVerifier
from patent_intake.parser import PatentParser
from red_team.counter_arguments import (
    EnforcementProbabilityModel,
    InvalidityGenerator,
    NonInfringementGenerator,
)
from retrieval.dual_encoder import RetrievalEngine

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
TEMPLATES_DIR = ROOT / "src" / "app" / "templates"
STATIC_DIR = ROOT / "src" / "app" / "static"

# Global state loaded at startup
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and corpus once at startup."""
    # Retrieval engine
    engine = RetrievalEngine(MODELS_DIR)
    try:
        model_type = engine.load()
        print(f"[PAEwall] Loaded retrieval engine: {model_type}")
    except Exception as e:
        print(f"[PAEwall] WARNING: {e} — retrieval unavailable.")
        engine = None

    # Product corpus for classical/BM25 fallback
    corpus = None
    bench_path = PROCESSED_DIR / "pae_bench.parquet"
    if bench_path.exists():
        df = pd.read_parquet(bench_path)
        corpus = (
            df[["company_name", "product_description"]]
            .drop_duplicates("company_name")
            .reset_index(drop=True)
        )
        if engine is not None:
            engine.load_product_corpus(corpus)
        print(f"[PAEwall] Loaded product corpus: {len(corpus)} companies")

    # Patent lookup index — prefer full patents.json, fall back to bench subset
    patent_index: dict[str, dict] = {}
    patents_path = RAW_DIR / "patents.json"
    if not patents_path.exists():
        patents_path = RAW_DIR / "patents_bench.json"
    if patents_path.exists():
        with open(patents_path) as f:
            patents_raw = json.load(f)
        for p in patents_raw:
            num = str(p.get("patent_number", "")).replace("-", "").strip()
            patent_index[num] = p
        print(f"[PAEwall] Patent index: {len(patent_index)} patents ({patents_path.name})")

    _state.update({
        "engine": engine,
        "corpus": corpus,
        "patent_index": patent_index,
        "parser": PatentParser(),
        "chart_gen": ClaimChartGenerator(),
        "verifier": FaithfulnessVerifier(),
        "ni_gen": NonInfringementGenerator(),
        "inv_gen": InvalidityGenerator(),
        "enforcement_model": EnforcementProbabilityModel(),
    })

    yield

    _state.clear()


app = FastAPI(
    title="PAEwall",
    description="Multimodal patent infringement discovery system",
    version="0.1.0",
    lifespan=lifespan,
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR)) if TEMPLATES_DIR.exists() else None


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    if templates:
        return templates.TemplateResponse(request, "index.html")
    return HTMLResponse(_fallback_html())


@app.post("/api/analyze")
async def analyze_patent(
    patent_number: str = Form(None),
    patent_pdf: UploadFile = File(None),
):
    """
    Main analysis endpoint.

    Accepts a USPTO patent number or uploaded PDF.
    Returns: candidate infringers, claim charts, counter-arguments.
    """
    if not patent_number and not patent_pdf:
        return JSONResponse({"error": "Provide a patent_number or upload a patent_pdf."}, status_code=400)

    # Module A — Parse patent claims
    patent_data, claims_text, priority_date = await _load_patent(patent_number, patent_pdf)
    if not claims_text:
        return JSONResponse({"error": "Could not extract patent claims. Check patent number or PDF."}, status_code=422)

    parser: PatentParser = _state["parser"]
    parsed_claims = parser.parse_independent_claims(claims_text)

    # Module B — Retrieve candidate infringers
    engine: RetrievalEngine | None = _state.get("engine")
    if engine is None:
        return JSONResponse({"error": "Retrieval model not loaded. Run training scripts first."}, status_code=503)

    candidates = engine.retrieve(claims_text, top_k=10)

    # Module C + D — Claim charts and counter-arguments per top candidate
    results = []
    for candidate in candidates[:5]:
        evidence = [candidate.get("description", "")]

        # Claim chart
        chart_gen: ClaimChartGenerator = _state["chart_gen"]
        verifier: FaithfulnessVerifier = _state["verifier"]
        try:
            mappings = chart_gen.generate(parsed_claims, evidence)
        except Exception as e:
            mappings = []

        chart_obj = ClaimChart(
            patent_id=patent_number or "uploaded",
            product_id=candidate.get("company_name", ""),
            mappings=mappings,
        )
        if mappings:
            chart_obj = verifier.score_chart(chart_obj)

        chart_dict = {
            "patent_id": chart_obj.patent_id,
            "product_id": chart_obj.product_id,
            "overall_confidence": chart_obj.overall_confidence,
            "mappings": [
                {
                    "limitation": m.limitation,
                    "evidence": m.evidence,
                    "faithfulness_label": m.faithfulness_label,
                    "faithfulness_score": m.faithfulness_score,
                }
                for m in chart_obj.mappings
            ],
        }

        # Counter-arguments
        ni_gen: NonInfringementGenerator = _state["ni_gen"]
        inv_gen: InvalidityGenerator = _state["inv_gen"]
        enf_model: EnforcementProbabilityModel = _state["enforcement_model"]

        try:
            ni_args = ni_gen.generate(chart_dict, num_args=3)
        except Exception:
            ni_args = []
        try:
            inv_args = inv_gen.generate(
                [c.full_text for c in parsed_claims],
                priority_date=priority_date or "",
                num_args=2,
            )
        except Exception:
            inv_args = []

        num_met = sum(1 for m in chart_obj.mappings if m.faithfulness_label == "supports")
        enf_features = {
            "faithfulness_score": chart_obj.overall_confidence,
            "num_limitations_met": num_met,
            "num_limitations_total": max(len(chart_obj.mappings), 1),
            "citation_count": patent_data.get("metadata", {}).get("citation_count", 0),
            "non_infringement_strength": sum(a.strength for a in ni_args) / max(len(ni_args), 1),
            "invalidity_strength": sum(a.strength for a in inv_args) / max(len(inv_args), 1),
        }
        enforcement_prob = enf_model.predict(enf_features)

        results.append({
            "rank": candidate.get("rank", 0),
            "company_name": candidate.get("company_name", ""),
            "retrieval_score": candidate.get("score", 0.0),
            "claim_chart": chart_dict,
            "non_infringement_args": [
                {"summary": a.summary, "legal_basis": a.legal_basis, "strength": a.strength}
                for a in ni_args
            ],
            "invalidity_args": [
                {"summary": a.summary, "legal_basis": a.legal_basis, "strength": a.strength}
                for a in inv_args
            ],
            "enforcement_probability": enforcement_prob,
        })

    return {
        "patent_id": patent_number or "uploaded",
        "num_claims_parsed": len(parsed_claims),
        "candidates": results,
    }


@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "version": "0.1.0",
        "retrieval_model": _state.get("engine") and getattr(_state["engine"], "_retriever_type", None),
        "patent_index_size": len(_state.get("patent_index", {})),
        "product_corpus_size": len(_state["corpus"]) if _state.get("corpus") is not None else 0,
    }


async def _load_patent(patent_number: str | None, pdf: UploadFile | None) -> tuple[dict, str, str]:
    """
    Resolve patent claims text and priority date from a patent number or uploaded PDF.

    Returns (patent_data_dict, claims_text, priority_date).
    """
    patent_index: dict = _state.get("patent_index", {})

    if patent_number:
        key = patent_number.replace("-", "").strip()
        # Try exact match first
        data = patent_index.get(key) or patent_index.get(f"US{key}")
        if not data:
            # Partial match only for keys with 7+ digits (avoid "1234" matching real patents)
            if len(re.sub(r'\D', '', key)) >= 7:
                for k, v in patent_index.items():
                    if key in k:
                        data = v
                        break

        if data:
            claims = " ".join(c.get("claim_text", "") for c in data.get("claims", []))
            priority = data.get("metadata", {}).get("priority_date", "")
            return data, claims, str(priority)

    if pdf is not None:
        contents = await pdf.read()
        text = _extract_pdf_text(contents)
        parser: PatentParser = _state["parser"]
        claims_list = parser.extract_claims(text)
        claims_text = " ".join(claims_list)
        return {}, claims_text, ""

    return {}, "", ""


def _extract_pdf_text(contents: bytes) -> str:
    """Extract plain text from a PDF byte string."""
    try:
        import io
        import pypdf

        reader = pypdf.PdfReader(io.BytesIO(contents))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception:
        # Try pdfplumber as fallback
        try:
            import io
            import pdfplumber

            with pdfplumber.open(io.BytesIO(contents)) as pdf:
                return "\n".join(p.extract_text() or "" for p in pdf.pages)
        except Exception:
            return contents.decode("utf-8", errors="ignore")


def _fallback_html() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>PAEwall</title></head>
<body>
<h1>PAEwall</h1>
<p>Templates not found. Place index.html in src/app/templates/.</p>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
