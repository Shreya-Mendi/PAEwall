"""
PAEwall — Interactive Patent Infringement Discovery Application

FastAPI backend serving the PAEwall web interface.
Runs inference only (no training).

Usage:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

app = FastAPI(
    title="PAEwall",
    description="Multimodal patent infringement discovery system",
    version="0.1.0",
)

MODELS_DIR = Path(__file__).resolve().parent / "models"
TEMPLATES_DIR = Path(__file__).resolve().parent / "src" / "app" / "templates"
STATIC_DIR = Path(__file__).resolve().parent / "src" / "app" / "static"

# TODO: Mount static files and templates once frontend is built
# app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
# templates = Jinja2Templates(directory=TEMPLATES_DIR)


@app.get("/", response_class=HTMLResponse)
async def root():
    """Landing page."""
    return """
    <html>
        <head><title>PAEtrol</title></head>
        <body>
            <h1>PAEwall</h1>
            <p>Multimodal patent infringement discovery system.</p>
            <p>Under construction. Check back soon.</p>
        </body>
    </html>
    """


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
        return {"error": "Provide a patent number or upload a PDF."}

    # TODO: Module A — Parse patent claims
    # TODO: Module B — Retrieve candidate products
    # TODO: Module C — Generate faithfulness-scored claim charts
    # TODO: Module D — Generate counter-arguments + enforcement probability

    return {
        "status": "not_implemented",
        "message": "Analysis pipeline under development.",
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "version": "0.1.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
