"""
RiskShield AI - Backend API (FastAPI)

Thin entrypoint: loads artifacts once at startup and mounts the route
modules under backend/api/. Route logic lives there, grouped by domain
(transactions, risk, cases, analytics) rather than in one file, so it
mirrors the repo layout used for the ML/graph/agent code.

Run:
    uvicorn backend.main:app --reload --port 8000

Interactive API docs (Swagger UI) are auto-served at /docs.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.state import load_artifacts
from backend.api import transactions, risk, cases, analytics


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield


app = FastAPI(
    title="RiskShield AI API",
    version="0.1.0",
    description="Agentic merchant-risk platform: transaction scoring, abuse-ring detection, "
                 "AI investigation, and policy-gated recommendations.",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

app.include_router(transactions.router)
app.include_router(risk.router)
app.include_router(cases.router)
app.include_router(analytics.router)


@app.get("/")
def root():
    return {"service": "RiskShield AI", "status": "ok", "docs": "/docs"}
