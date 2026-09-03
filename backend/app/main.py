"""
backend/app/main.py
===================
FastAPI application entry point.

Routers registered:
  health_router  — GET /api/v1/health, GET /api/v1/model/info
  alerts_router  — GET /api/v1/alerts
  wallets_router — GET /api/v1/wallets/{address_hash}
  graph_router   — GET /api/v1/graph
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import get_settings
from backend.app.routers import (
    alerts_router,
    graph_router,
    health_router,
    wallets_router,
)

settings = get_settings()

app = FastAPI(
    title="SIH26146 — Bitcoin Risk Analysis API",
    description=(
        "AI-powered monitoring and analysis of Bitcoin transaction traffic. "
        "Pipeline B (Bitcoin GCN) produces model-derived risk rankings (0–100) "
        "for address prioritization. "
        "Scores are NOT calibrated probabilities and do not constitute "
        "an enforcement or legal determination. "
        "Every response that contains a risk score includes a score_disclaimer field."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
# Allow the React frontend (running on :5173 in dev) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(health_router)
app.include_router(alerts_router)
app.include_router(wallets_router)
app.include_router(graph_router)
