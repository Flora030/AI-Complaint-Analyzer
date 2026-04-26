"""
FastAPI backend for the AI Complaint Analyzer.

Receives a customer complaint, sends it to a local Ollama model with a strict
JSON schema, and returns structured analysis: summary, category, severity,
sentiment, and a suggested response.

Run:
    uvicorn backend:app --reload --port 8000

Environment variables:
    OLLAMA_URL     default http://localhost:11434
    OLLAMA_MODEL   default llama3.2
    REQUEST_TIMEOUT  default 120 (seconds)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("complaint-analyzer")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "120"))


Severity = Literal["Low", "Medium", "High"]
Category = Literal["Delivery", "Product", "Payment", "Service"]
Sentiment = Literal["Positive", "Neutral", "Negative"]


class ComplaintRequest(BaseModel):
    complaint: str = Field(..., min_length=1, max_length=5000)


class AnalysisResponse(BaseModel):
    summary: str
    category: Category
    severity: Severity
    sentiment: Sentiment
    response: str


# JSON schema the model is constrained to. Requires Ollama >= 0.5.0.
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "summary":   {"type": "string"},
        "category":  {"type": "string", "enum": ["Delivery", "Product", "Payment", "Service"]},
        "severity":  {"type": "string", "enum": ["Low", "Medium", "High"]},
        "sentiment": {"type": "string", "enum": ["Positive", "Neutral", "Negative"]},
        "response":  {"type": "string"},
    },
    "required": ["summary", "category", "severity", "sentiment", "response"],
}

SYSTEM_PROMPT = """You are an analyst for an e-commerce customer service team.
You will receive a single customer complaint. Return a JSON object with:

- summary: one or two sentences, neutral and factual, describing the issue
- category: exactly one of "Delivery", "Product", "Payment", "Service"
- severity: exactly one of "Low", "Medium", "High"
- sentiment: exactly one of "Positive", "Neutral", "Negative"
- response: a professional, empathetic two-to-four sentence reply the agent could send

Severity guidance:
- High: financial loss, safety risk, legal threats, repeated failures, or strong escalation language
- Medium: significant inconvenience but recoverable in one interaction
- Low: minor issue, easy to resolve

Category guidance:
- Delivery: shipping, tracking, lost or late packages
- Product: damaged, defective, wrong, or missing items
- Payment: charges, refunds, billing errors, double-charges
- Service: agent behavior, response quality, communication problems

Output ONLY the JSON object. No preamble, no markdown fences, no commentary."""


app = FastAPI(title="AI Complaint Analyzer", version="1.0.0")


@app.get("/health")
async def health() -> dict:
    """Probe Ollama and report whether the configured model is pulled."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{OLLAMA_URL}/api/tags")
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
        return {
            "status": "ok",
            "ollama_reachable": True,
            "model": OLLAMA_MODEL,
            "model_pulled": any(m.startswith(OLLAMA_MODEL) for m in models),
            "available_models": models,
        }
    except Exception as e:
        log.warning("Ollama health check failed: %s", e)
        return {
            "status": "degraded",
            "ollama_reachable": False,
            "model": OLLAMA_MODEL,
            "model_pulled": False,
            "available_models": [],
            "error": str(e),
        }


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(req: ComplaintRequest) -> AnalysisResponse:
    """Send the complaint to Ollama and return validated structured output."""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": req.complaint.strip()},
        ],
        "format": RESPONSE_SCHEMA,
        "stream": False,
        "options": {"temperature": 0.2},
    }

    log.info("Analyzing complaint (%d chars) with model=%s", len(req.complaint), OLLAMA_MODEL)

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            r = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()
    except httpx.ConnectError:
        log.error("Cannot connect to Ollama at %s", OLLAMA_URL)
        raise HTTPException(
            status_code=503,
            detail=f"Cannot reach Ollama at {OLLAMA_URL}. Start it with 'ollama serve'.",
        )
    except httpx.TimeoutException:
        log.error("Ollama request timed out after %ss", REQUEST_TIMEOUT)
        raise HTTPException(status_code=504, detail="Ollama request timed out.")
    except httpx.HTTPStatusError as e:
        body = e.response.text[:300]
        log.error("Ollama returned HTTP %s: %s", e.response.status_code, body)
        raise HTTPException(
            status_code=502,
            detail=f"Ollama error {e.response.status_code}: {body}",
        )

    raw = data.get("message", {}).get("content", "").strip()
    if not raw:
        raise HTTPException(status_code=502, detail="Ollama returned an empty response.")

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        log.error("Could not parse JSON from model. Raw output: %s", raw[:500])
        raise HTTPException(
            status_code=502,
            detail=f"Model did not return valid JSON. First 200 chars: {raw[:200]}",
        )

    try:
        return AnalysisResponse(**parsed)
    except ValidationError as e:
        log.error("Model JSON failed schema validation: %s", e)
        raise HTTPException(
            status_code=502,
            detail=f"Model output failed validation: {e.errors()}",
        )
