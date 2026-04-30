"""
FastAPI backend for the AI Complaint Analyzer.

Endpoints:
    GET  /health                  Ollama reachability + model availability
    POST /analyze                 Analyze a complaint, save it, and embed it
    POST /similar                 Cosine-similarity search over stored complaints
    POST /backfill_embeddings     Generate embeddings for older complaints
    POST /send_email              Send response email to a customer

Run:
    uvicorn backend:app --reload --port 8000

Environment variables:
    OLLAMA_URL          default http://localhost:11434
    OLLAMA_MODEL        default llama3.2
    OLLAMA_EMBED_MODEL  default nomic-embed-text
    REQUEST_TIMEOUT     default 120 (seconds)

    SMTP_HOST           e.g. smtp.gmail.com
    SMTP_PORT           default 587
    SMTP_USER           your sending address
    SMTP_PASS           app password or SMTP credential
    EMAIL_FROM          display name + address, e.g. "Support <support@company.com>"
                        (falls back to SMTP_USER if not set)
"""

from __future__ import annotations

import json
import logging
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Literal

import httpx
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError

load_dotenv()

from database import (
    init_db,
    save_complaint,
    save_embedding,
    fetch_all_embeddings,
    fetch_all_complaints,
    fetch_complaint,
    fetch_customer,
    fetch_customer_complaints,
    fetch_complaints_without_embeddings,
    mark_email_sent,
)

init_db()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("complaint-analyzer")

OLLAMA_URL         = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL       = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
REQUEST_TIMEOUT    = float(os.getenv("REQUEST_TIMEOUT", "120"))

SMTP_HOST  = os.getenv("SMTP_HOST", "")
SMTP_PORT  = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER  = os.getenv("SMTP_USER", "")
SMTP_PASS  = os.getenv("SMTP_PASS", "")
EMAIL_FROM = os.getenv("EMAIL_FROM", SMTP_USER)

Severity  = Literal["Low", "Medium", "High"]
Category  = Literal["Delivery", "Product", "Payment", "Service"]
Sentiment = Literal["Positive", "Neutral", "Negative"]
ResolutionSuccess = Literal["Successful", "Partial", "Unsuccessful"]


class ComplaintRequest(BaseModel):
    complaint: str = Field(..., min_length=1, max_length=5000)
    customer_id: int | None = None


class AnalysisResponse(BaseModel):
    id: int
    summary: str
    category: Category
    severity: Severity
    sentiment: Sentiment
    response: str


class SimilarRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    top_k: int = Field(default=5, ge=1, le=20)
    exclude_id: int | None = None


class SimilarComplaint(BaseModel):
    id: int
    complaint: str
    summary: str
    category: str
    severity: str
    sentiment: str
    response: str
    status: str
    created_at: str | None
    customer_name: str | None
    similarity: float


class SendEmailRequest(BaseModel):
    complaint_id: int
    to_email: str
    to_name: str | None = None
    subject: str
    body: str


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
You will receive a single customer complaint, sometimes preceded by a brief
[Customer context] block describing the customer and their prior complaint
history. Use this history to inform tone and personalize the response, but
analyze the CURRENT complaint, not past ones.

Return a JSON object with:

- Summary: One or two sentences, neutral and factual, describing the issue
- Category: Exactly one of "Delivery", "Product", "Payment", "Service"
- Severity: Exactly one of "Low", "Medium", "High"
- Sentiment: Exactly one of "Positive", "Neutral", "Negative"
- Response: A professional, empathetic two-to-four sentence reply the agent
  could send. If customer history is provided, acknowledge their loyalty or
  prior issues where appropriate.

Severity guidance:
- High: Financial loss, safety risk, legal threats, repeated failures, or strong escalation language.
        Repeat complaints from the same customer should bias toward higher severity.
- Medium: Significant inconvenience but recoverable in one interaction
- Low: Minor issue, easy to resolve

Category guidance:
- Delivery: Shipping, tracking, lost or late packages
- Product: Damaged, defective, wrong, or missing items
- Payment: Charges, refunds, billing errors, double-charges
- Service: Agent behavior, response quality, communication problems

Output ONLY the JSON object. No preamble, no markdown fences, no commentary."""


app = FastAPI(title="AI Complaint Analyzer", version="2.1.0")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_customer_context(customer_id: int | None) -> str:
    if not customer_id:
        return ""
    customer = fetch_customer(customer_id)
    if not customer:
        return ""

    history = fetch_customer_complaints(customer_id, limit=5)
    name = customer["name"]
    ltv  = customer.get("lifetime_value") or 0.0

    if not history:
        return (
            f"[Customer context]\n"
            f"Customer: {name}\n"
            f"Lifetime value: ${ltv:,.0f}\n"
            f"This is their first complaint with us.\n\n"
            f"[Current complaint]\n"
        )

    lines = []
    for h in history:
        date = (h.get("created_at") or "")[:10]
        lines.append(
            f"- {date}: {h.get('category','?')} ({h.get('severity','?')}) — {h.get('summary','')}"
        )

    return (
        f"[Customer context]\n"
        f"Customer: {name}\n"
        f"Lifetime value: ${ltv:,.0f}\n"
        f"Prior complaints: {len(history)}\n"
        f"Recent history:\n" + "\n".join(lines) + "\n\n"
        f"[Current complaint]\n"
    )


async def _get_embedding(text: str) -> list[float] | None:
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                f"{OLLAMA_URL}/api/embed",
                json={"model": OLLAMA_EMBED_MODEL, "input": text},
            )
            r.raise_for_status()
            data = r.json()
            embeddings = data.get("embeddings") or []
            if embeddings and len(embeddings[0]) > 0:
                return embeddings[0]
            single = data.get("embedding")
            if single:
                return single
            return None
    except Exception as e:
        log.warning("Embedding generation failed (model=%s): %s", OLLAMA_EMBED_MODEL, e)
        return None


def _smtp_configured() -> bool:
    return bool(SMTP_HOST and SMTP_USER and SMTP_PASS)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------
@app.get("/health")
async def health() -> dict:
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{OLLAMA_URL}/api/tags")
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
        return {
            "status": "ok",
            "ollama_reachable": True,
            "model": OLLAMA_MODEL,
            "embed_model": OLLAMA_EMBED_MODEL,
            "model_pulled": any(m.startswith(OLLAMA_MODEL) for m in models),
            "embed_model_pulled": any(m.startswith(OLLAMA_EMBED_MODEL) for m in models),
            "available_models": models,
            "email_configured": _smtp_configured(),
        }
    except Exception as e:
        log.warning("Ollama health check failed: %s", e)
        return {
            "status": "degraded",
            "ollama_reachable": False,
            "model": OLLAMA_MODEL,
            "embed_model": OLLAMA_EMBED_MODEL,
            "model_pulled": False,
            "embed_model_pulled": False,
            "available_models": [],
            "error": str(e),
            "email_configured": _smtp_configured(),
        }


# ---------------------------------------------------------------------------
# /analyze
# ---------------------------------------------------------------------------
@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(req: ComplaintRequest) -> AnalysisResponse:
    customer_block = _build_customer_context(req.customer_id)
    user_content = (customer_block + req.complaint.strip()) if customer_block else req.complaint.strip()

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ],
        "format": RESPONSE_SCHEMA,
        "stream": False,
        "options": {"temperature": 0.2},
    }

    log.info(
        "Analyzing complaint (%d chars, customer_id=%s) with model=%s",
        len(req.complaint), req.customer_id, OLLAMA_MODEL,
    )

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

    complaint_id = save_complaint({
        "complaint": req.complaint,
        "summary":   parsed["summary"],
        "category":  parsed["category"],
        "severity":  parsed["severity"],
        "sentiment": parsed["sentiment"],
        "response":  parsed["response"],
        "customer_id": req.customer_id,
    })
    parsed["id"] = complaint_id

    emb = await _get_embedding(req.complaint)
    if emb:
        try:
            save_embedding(complaint_id, emb, model=OLLAMA_EMBED_MODEL)
        except Exception as e:
            log.warning("Failed to persist embedding for complaint %s: %s", complaint_id, e)

    try:
        return AnalysisResponse(**parsed)
    except ValidationError as e:
        log.error("Model JSON failed schema validation: %s", e)
        raise HTTPException(
            status_code=502,
            detail=f"Model output failed validation: {e.errors()}",
        )


# ---------------------------------------------------------------------------
# /similar
# ---------------------------------------------------------------------------
@app.post("/similar", response_model=list[SimilarComplaint])
async def similar(req: SimilarRequest) -> list[dict]:
    emb = await _get_embedding(req.text)
    if not emb:
        return []

    query = np.asarray(emb, dtype=np.float32)
    qn = float(np.linalg.norm(query))
    if qn == 0:
        return []
    query = query / qn

    all_embeddings = fetch_all_embeddings()
    if not all_embeddings:
        return []

    sims: list[tuple[int, float]] = []
    for cid, vec in all_embeddings.items():
        if req.exclude_id and cid == req.exclude_id:
            continue
        if vec.size != query.size:
            continue
        n = float(np.linalg.norm(vec))
        if n == 0:
            continue
        sim = float(np.dot(query, vec / n))
        sims.append((cid, sim))

    sims.sort(key=lambda x: -x[1])
    top = sims[: req.top_k]

    out: list[dict] = []
    for cid, score in top:
        c = fetch_complaint(cid)
        if not c:
            continue
        out.append({
            "id": c["id"],
            "complaint": c["complaint"] or "",
            "summary": c["summary"] or "",
            "category": c["category"] or "",
            "severity": c["severity"] or "",
            "sentiment": c["sentiment"] or "",
            "response": c["response"] or "",
            "status": c["status"] or "",
            "created_at": c["created_at"],
            "customer_name": c.get("customer_name"),
            "similarity": score,
        })
    return out


# ---------------------------------------------------------------------------
# /backfill_embeddings
# ---------------------------------------------------------------------------
@app.post("/backfill_embeddings")
async def backfill_embeddings() -> dict:
    pending = fetch_complaints_without_embeddings()
    processed, failed = 0, 0
    for c in pending:
        text = c.get("complaint") or ""
        if not text.strip():
            failed += 1
            continue
        emb = await _get_embedding(text)
        if not emb:
            failed += 1
            continue
        try:
            save_embedding(c["id"], emb, model=OLLAMA_EMBED_MODEL)
            processed += 1
        except Exception as e:
            log.warning("Failed to save embedding for %s: %s", c["id"], e)
            failed += 1
    return {"total": len(pending), "processed": processed, "failed": failed}


# ---------------------------------------------------------------------------
# /send_email
# ---------------------------------------------------------------------------
@app.post("/send_email")
async def send_email(req: SendEmailRequest) -> dict:
    """Send a response email to a customer and mark the complaint as emailed."""
    if not _smtp_configured():
        raise HTTPException(
            status_code=503,
            detail=(
                "SMTP is not configured. Set SMTP_HOST, SMTP_USER, and SMTP_PASS "
                "environment variables (or in your .env file) and restart the backend."
            ),
        )

    # Build message
    msg = MIMEMultipart("alternative")
    msg["Subject"] = req.subject
    msg["From"]    = EMAIL_FROM
    msg["To"]      = f"{req.to_name} <{req.to_email}>" if req.to_name else req.to_email

    # Plain text part
    text_part = MIMEText(req.body, "plain")

    # Basic HTML part
    html_body = req.body.replace("\n", "<br>")
    html_part = MIMEText(
        f"""<html><body style="font-family:Arial,sans-serif;font-size:14px;
            color:#222;line-height:1.6;max-width:600px;margin:40px auto;padding:0 20px">
            <p>{html_body}</p>
            <hr style="border:none;border-top:1px solid #eee;margin:32px 0">
            <p style="font-size:12px;color:#888">This message was sent by our
            Customer Support team.</p>
        </body></html>""",
        "html",
    )

    msg.attach(text_part)
    msg.attach(html_part)

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
            server.ehlo()
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(EMAIL_FROM, req.to_email, msg.as_string())
        log.info("Email sent to %s for complaint %s", req.to_email, req.complaint_id)
    except smtplib.SMTPAuthenticationError:
        raise HTTPException(
            status_code=502,
            detail="SMTP authentication failed. Check your SMTP_USER and SMTP_PASS.",
        )
    except smtplib.SMTPException as e:
        log.error("SMTP error sending to %s: %s", req.to_email, e)
        raise HTTPException(status_code=502, detail=f"SMTP error: {e}")
    except OSError as e:
        log.error("Network error reaching SMTP host %s: %s", SMTP_HOST, e)
        raise HTTPException(status_code=502, detail=f"Cannot reach SMTP host: {e}")

    # Mark complaint as emailed
    try:
        mark_email_sent(req.complaint_id)
    except Exception as e:
        log.warning("Could not mark complaint %s as emailed: %s", req.complaint_id, e)

    return {"sent": True, "to": req.to_email, "complaint_id": req.complaint_id}