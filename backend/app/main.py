"""
backend/app/main.py
===================
FastAPI application entry point.

Routers are registered here. For now, only /health is live.
All other routers (wallets, alerts, model info) are added in later tasks.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import get_settings

settings = get_settings()

app = FastAPI(
    title="SIH26146 — Bitcoin Risk Analysis API",
    description=(
        "AI-powered monitoring and analysis of Bitcoin transaction traffic. "
        "Pipeline B produces model-derived risk rankings (0–100) for address "
        "prioritization. Scores are NOT calibrated probabilities."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow the React frontend (running on :5173) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["system"])
def health_check():
    """
    Docker health check endpoint.
    Returns 200 when the API process is alive.
    Database connectivity is checked separately via the db service health check.
    """
    return {"status": "ok", "offline_mode": settings.offline_mode}
