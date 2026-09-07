# backend/app/routers/__init__.py
# Expose routers for registration in main.py
from backend.app.routers.alerts import router as alerts_router
from backend.app.routers.graph import router as graph_router
from backend.app.routers.health import router as health_router
from backend.app.routers.wallets import router as wallets_router

__all__ = ["health_router", "alerts_router", "wallets_router", "graph_router"]
