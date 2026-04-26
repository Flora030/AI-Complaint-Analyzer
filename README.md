# AI Complaint Analyzer

AI-powered tool for analyzing e-commerce customer complaints using a local LLM (Ollama). Generates summary, category, severity, sentiment, and a suggested response.

## Architecture

```
┌──────────────────┐     POST /analyze      ┌──────────────────┐    /api/chat   ┌──────────┐
│  Streamlit UI    │ ─────────────────────▶ │  FastAPI backend │ ─────────────▶│  Ollama  │
│  (frontend.py)   │ ◀───────────────────── │   (backend.py)   │ ◀──────────── │  llama3.2│
└──────────────────┘   structured JSON      └──────────────────┘                └──────────┘
```

The FastAPI backend constrains the model output with a JSON schema so the response is always:

```json
{
  "summary": "...",
  "category": "Delivery | Product | Payment | Service",
  "severity": "Low | Medium | High",
  "sentiment": "Positive | Neutral | Negative",
  "response": "..."
}
```

Validation happens twice: once by Ollama against the schema (>= 0.5.0), then again by Pydantic on the backend.

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) 0.5.0 or newer (required for structured outputs)
- ~3 GB disk for the default `llama3.2` model

## Setup

```bash
# 1. Install Ollama, then start the daemon
ollama serve            # leave running in its own terminal

# 2. Pull a model (in a new terminal)
ollama pull llama3.2    # default — about 2 GB
# or, for higher quality:
# ollama pull llama3.1:8b
# ollama pull qwen2.5:7b

# 3. Python deps
pip install -r requirements.txt
```

## Run

You need three things running. Use three terminals.

```bash
# Terminal 1 — Ollama
ollama serve

# Terminal 2 — backend
uvicorn backend:app --reload --port 8000

# Terminal 3 — frontend
streamlit run frontend.py
```

Then open the Streamlit URL it prints (usually `http://localhost:8501`).

### Verify the backend works

```bash
curl http://localhost:8000/health
# expects: ollama_reachable=true, model_pulled=true

curl -X POST http://localhost:8000/analyze \
     -H "Content-Type: application/json" \
     -d '{"complaint": "My package is two weeks late and nobody answers. I want a refund."}'
```

## Configuration

Override via environment variables:

| Variable          | Default                  | Purpose                           |
| ----------------- | ------------------------ | --------------------------------- |
| `OLLAMA_URL`      | `http://localhost:11434` | Where Ollama is listening         |
| `OLLAMA_MODEL`    | `llama3.2`               | Which model the backend uses      |
| `REQUEST_TIMEOUT` | `120`                    | Seconds to wait for Ollama        |
| `BACKEND_URL`     | `http://localhost:8000`  | Where the frontend calls (env for `frontend.py`) |

Example:

```bash
OLLAMA_MODEL=llama3.1:8b uvicorn backend:app --port 8000
```

## File map

```
backend.py        FastAPI + Ollama integration with JSON schema enforcement
frontend.py       Streamlit UI (dashboard, analyzer, history)
requirements.txt  Python dependencies
```

## Notes on behavior

- If the backend is unreachable, the frontend shows a yellow banner and falls back to canned demo data so the UI still demos cleanly. Real backend errors (500, 502, etc.) are surfaced verbatim instead of hidden.
- Analysis results are stored in `st.session_state` so they survive reruns triggered by editing the suggested response or clicking Save.
- "Save Reviewed Response" appends to an in-memory list (visible in the History tab). For production, replace that with a database or ticketing-system call.

## Evaluation

The numbers in the Evaluation Preview tab are placeholders. To produce real numbers, label a sample of complaints by hand, run them through `/analyze`, and compare. A simple version:

```python
import json, requests, pandas as pd
df = pd.read_csv("labeled.csv")  # columns: complaint, category, severity, sentiment
df["pred"] = df["complaint"].map(
    lambda c: requests.post("http://localhost:8000/analyze", json={"complaint": c}).json()
)
for field in ["category", "severity", "sentiment"]:
    acc = (df[field] == df["pred"].map(lambda r: r[field])).mean()
    print(field, acc)
```