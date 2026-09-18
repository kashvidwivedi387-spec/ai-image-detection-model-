"""
VerifAI demo API.

Run with:
    pip install -r requirements.txt
    uvicorn app:app --reload --port 8000

Then open http://localhost:8000 in a browser.
"""

import base64
import io
import time
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

import detector

app = FastAPI(title="VerifAI Demo API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/verify")
async def verify(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="File too large (max 15 MB for this demo).")

    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read this file as an image.")

    start = time.perf_counter()
    result = detector.analyze_image(image)
    elapsed_ms = round((time.perf_counter() - start) * 1000)

    heatmap_buf = io.BytesIO()
    result.ela["heatmap_image"].save(heatmap_buf, format="PNG")
    heatmap_b64 = base64.b64encode(heatmap_buf.getvalue()).decode("ascii")

    return {
        "filename": file.filename,
        "processing_time_ms": elapsed_ms,
        "score": result.score,
        "verdict": result.verdict,
        "evidence": result.evidence,
        "breakdown": {
            "metadata": {k: v for k, v in result.metadata.items()},
            "ela": {k: v for k, v in result.ela.items() if k != "heatmap_image"},
            "pattern": result.pattern,
        },
        "heatmap_data_url": f"data:image/png;base64,{heatmap_b64}",
    }


# Serve the single-page frontend directly from the API for a one-command demo.
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
