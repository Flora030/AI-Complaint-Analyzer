"""
Complaint Intelligence — Streamlit frontend.

Tabs:
- Overview      KPIs, distribution charts, 30-day trend, recent feed
- Analyze       Single complaint with optional customer context + similarity search
- Bulk          CSV upload for batch analysis (with optional customer-email linking)
- Customers     CRUD for customer profiles
- History       Filterable card feed with inline status updates
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from database import (
    init_db,
    fetch_all_complaints,
    update_complaint,
    update_status,
    delete_complaint,
    create_customer,
    fetch_all_customers,
    fetch_customer,
    fetch_customer_complaints,
    update_customer,
    delete_customer,
    count_complaints_without_embeddings,
)

init_db()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Complaint Intelligence",
    page_icon="◐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Brand tokens
# ---------------------------------------------------------------------------
BG          = "#0b0d14"
SURFACE     = "#13161f"
SURFACE_2   = "#191d29"
BORDER      = "rgba(148, 163, 184, 0.12)"
BORDER_HOV  = "rgba(148, 163, 184, 0.24)"
TEXT        = "#f1f5f9"
MUTED       = "#94a3b8"
SUBTLE      = "#64748b"
PRIMARY     = "#a78bfa"
ACCENT      = "#fbbf24"
SUCCESS     = "#34d399"
WARNING     = "#fbbf24"
DANGER      = "#f87171"

CATEGORY_COLORS = {
    "Delivery": "#a78bfa",
    "Product":  "#22d3ee",
    "Payment":  "#fbbf24",
    "Service":  "#f472b6",
}
SEVERITY_COLORS = {
    "Low":    "#34d399",
    "Medium": "#fbbf24",
    "High":   "#f87171",
}
SENTIMENT_COLORS = {
    "Positive": "#34d399",
    "Neutral":  "#94a3b8",
    "Negative": "#f87171",
}
STATUS_COLORS = {
    "Needs Review": "#fbbf24",
    "In Progress":  "#22d3ee",
    "Resolved":     "#34d399",
}

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"]  {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: {TEXT};
}}

.stApp {{
    background:
        radial-gradient(1200px 600px at -10% -20%, rgba(167,139,250,0.10), transparent 60%),
        radial-gradient(900px 500px at 110% -10%, rgba(251,191,36,0.06), transparent 60%),
        {BG};
}}

#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ background: transparent; }}

/* Hero */
.hero {{
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 32px;
    padding: 24px 0 28px 0;
    border-bottom: 1px solid {BORDER};
    margin-bottom: 28px;
}}
.eyebrow {{
    display: inline-flex; align-items: center; gap: 8px;
    font-size: 11px; font-weight: 600;
    letter-spacing: 1.6px; text-transform: uppercase;
    color: {PRIMARY}; margin-bottom: 14px;
}}
.eyebrow::before {{
    content: ""; width: 6px; height: 6px; border-radius: 999px;
    background: {PRIMARY}; box-shadow: 0 0 10px {PRIMARY};
}}
.hero-title {{
    font-family: 'Fraunces', Georgia, serif;
    font-weight: 600; font-size: 52px; line-height: 1.05;
    letter-spacing: -1.5px; margin: 0 0 12px 0; color: {TEXT};
}}
.hero-title em {{ font-style: italic; color: {ACCENT}; font-weight: 500; }}
.hero-sub {{ font-size: 15px; line-height: 1.6; color: {MUTED}; max-width: 620px; }}
.hero-right {{ display: flex; flex-direction: column; gap: 8px; align-items: flex-end; }}

/* Pills / dots */
.pill {{
    display: inline-flex; align-items: center; gap: 8px;
    padding: 7px 13px; border-radius: 999px;
    background: {SURFACE}; border: 1px solid {BORDER};
    font-size: 11px; font-weight: 500; color: {MUTED};
    font-family: 'JetBrains Mono', monospace; letter-spacing: 0.3px;
}}
.dot {{ width: 7px; height: 7px; border-radius: 999px; }}
.dot-ok   {{ background: {SUCCESS}; box-shadow: 0 0 8px {SUCCESS}; }}
.dot-warn {{ background: {WARNING}; box-shadow: 0 0 8px {WARNING}; }}
.dot-err  {{ background: {DANGER};  box-shadow: 0 0 8px {DANGER}; }}

/* Section header */
.section {{
    margin: 28px 0 14px 0;
    display: flex; align-items: baseline; justify-content: space-between;
}}
.section h3 {{
    font-family: 'Fraunces', Georgia, serif;
    font-weight: 600; font-size: 24px; letter-spacing: -0.5px;
    margin: 0; color: {TEXT};
}}
.section .sub {{
    font-size: 12px; color: {SUBTLE};
    font-family: 'JetBrains Mono', monospace; letter-spacing: 0.5px;
    text-transform: uppercase;
}}

/* KPI */
.kpi {{
    padding: 22px; border-radius: 14px;
    background: linear-gradient(180deg, {SURFACE} 0%, {SURFACE_2} 100%);
    border: 1px solid {BORDER}; height: 100%;
    transition: border-color .15s ease;
}}
.kpi:hover {{ border-color: {BORDER_HOV}; }}
.kpi-label {{
    font-size: 11px; font-weight: 600;
    letter-spacing: 1.4px; text-transform: uppercase;
    color: {MUTED}; margin-bottom: 14px;
}}
.kpi-value {{
    font-family: 'Fraunces', Georgia, serif;
    font-size: 44px; font-weight: 600; line-height: 1;
    color: {TEXT}; margin-bottom: 6px;
}}
.kpi-meta {{ font-size: 12px; color: {SUBTLE}; }}
.kpi-accent {{ color: {PRIMARY}; }}
.kpi-warn   {{ color: {WARNING}; }}
.kpi-good   {{ color: {SUCCESS}; }}
.kpi-danger {{ color: {DANGER}; }}

/* Cards */
.card {{
    padding: 22px; border-radius: 14px;
    background: {SURFACE}; border: 1px solid {BORDER};
    margin-bottom: 12px; line-height: 1.6;
}}
.card-label {{
    font-size: 10px; font-weight: 700;
    letter-spacing: 1.4px; text-transform: uppercase;
    color: {MUTED}; margin-bottom: 10px;
}}
.card-body {{ font-size: 14.5px; color: {TEXT}; }}

/* Customer context card */
.ctx {{
    padding: 16px 20px; border-radius: 12px;
    background: linear-gradient(135deg, rgba(167,139,250,0.06), rgba(251,191,36,0.04));
    border: 1px solid {BORDER};
    margin-bottom: 14px;
}}
.ctx-name {{
    font-family: 'Fraunces', Georgia, serif;
    font-size: 18px; font-weight: 600; color: {TEXT};
    margin-bottom: 2px;
}}
.ctx-meta {{
    font-size: 12px; color: {MUTED};
    font-family: 'JetBrains Mono', monospace; letter-spacing: 0.4px;
}}

/* Chips */
.chip {{
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 10px; border-radius: 999px;
    font-size: 11px; font-weight: 600; letter-spacing: 0.3px;
    border: 1px solid;
}}
.chip-low    {{ color: {SUCCESS}; border-color: rgba(52,211,153,.35); background: rgba(52,211,153,.08); }}
.chip-medium {{ color: {WARNING}; border-color: rgba(251,191,36,.35); background: rgba(251,191,36,.08); }}
.chip-high   {{ color: {DANGER};  border-color: rgba(248,113,113,.4); background: rgba(248,113,113,.10); }}
.chip-pos {{ color: {SUCCESS}; border-color: rgba(52,211,153,.35); background: rgba(52,211,153,.08); }}
.chip-neu {{ color: {MUTED};  border-color: rgba(148,163,184,.3);  background: rgba(148,163,184,.08); }}
.chip-neg {{ color: {DANGER}; border-color: rgba(248,113,113,.4);  background: rgba(248,113,113,.10); }}

/* Issue cards */
.issue {{
    padding: 18px 20px; border-radius: 14px;
    background: {SURFACE}; border: 1px solid {BORDER};
    margin-bottom: 12px; transition: border-color .15s ease;
}}
.issue:hover {{ border-color: {BORDER_HOV}; }}
.issue-meta {{
    display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; color: {SUBTLE}; letter-spacing: 0.4px;
    margin-bottom: 10px;
}}
.issue-id {{ color: {PRIMARY}; font-weight: 600; }}
.issue-text {{ font-size: 14px; color: {TEXT}; line-height: 1.5; margin: 6px 0 8px 0; }}
.issue-summary {{
    font-size: 13px; color: {MUTED}; line-height: 1.55;
    border-left: 2px solid {BORDER_HOV};
    padding-left: 12px; margin: 4px 0 4px 0;
}}

/* Similar complaint card */
.sim {{
    padding: 14px 18px; border-radius: 12px;
    background: {SURFACE}; border: 1px solid {BORDER};
    margin-bottom: 10px;
}}
.sim-score {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px; color: {ACCENT}; font-weight: 600;
}}
.sim-bar {{
    height: 3px; border-radius: 999px;
    background: linear-gradient(90deg, {PRIMARY}, {ACCENT});
    margin-top: 4px;
}}

/* Empty state */
.empty {{
    text-align: center; padding: 60px 20px;
    border: 1px dashed {BORDER_HOV}; border-radius: 16px;
    background: rgba(255,255,255,0.01);
}}
.empty h4 {{
    font-family: 'Fraunces', Georgia, serif;
    font-size: 22px; font-weight: 600; color: {TEXT};
    margin: 0 0 8px 0;
}}
.empty p {{ color: {MUTED}; font-size: 14px; margin: 0; }}

/* Tabs */
button[data-baseweb="tab"] {{
    font-weight: 600 !important;
    letter-spacing: 0.3px;
    color: {MUTED} !important;
}}
button[data-baseweb="tab"][aria-selected="true"] {{ color: {TEXT} !important; }}
.stTabs [data-baseweb="tab-list"] {{
    gap: 4px;
    border-bottom: 1px solid {BORDER};
}}

div[data-testid="stMetricValue"] {{
    font-family: 'Fraunces', Georgia, serif;
    font-weight: 600;
}}

/* Buttons */
.stButton > button[kind="primary"] {{
    background: {PRIMARY}; color: #0b0d14; border: none;
    font-weight: 600; border-radius: 10px;
}}
.stButton > button[kind="primary"]:hover {{ background: #c4b5fd; }}

/* Inputs */
.stTextArea textarea, .stTextInput input, .stNumberInput input {{
    background: {SURFACE} !important;
    border: 1px solid {BORDER} !important;
    color: {TEXT} !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
}}
.stTextArea textarea:focus, .stTextInput input:focus, .stNumberInput input:focus {{
    border-color: {PRIMARY} !important;
}}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_data(ttl=15)
def check_backend() -> tuple[str, str]:
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=3)
        if r.status_code == 200:
            data = r.json()
            if data.get("ollama_reachable"):
                model = data.get("model", "model")
                embed_ok = data.get("embed_model_pulled", False)
                suffix = "" if embed_ok else " · no embeds"
                return "ok", f"live · {model}{suffix}"
            return "warn", "backend up · ollama down"
        return "warn", f"backend {r.status_code}"
    except requests.RequestException:
        return "err", "offline"


def call_analyze(text: str, customer_id: int | None = None) -> tuple[dict, str]:
    body: dict = {"complaint": text}
    if customer_id:
        body["customer_id"] = customer_id
    try:
        r = requests.post(f"{BACKEND_URL}/analyze", json=body, timeout=180)
        if r.status_code == 200:
            return r.json(), "live"
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        st.error(f"Backend returned {r.status_code}: {detail}")
        return _fallback_result(), "fallback"
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot reach the backend at {BACKEND_URL}. "
                 f"Run `uvicorn backend:app --port 8000` and start Ollama.")
        return _fallback_result(), "fallback"
    except requests.exceptions.Timeout:
        st.error("Backend request timed out. Try a smaller model or raise REQUEST_TIMEOUT.")
        return _fallback_result(), "fallback"


def call_similar(text: str, top_k: int = 3, exclude_id: int | None = None) -> list[dict]:
    body: dict = {"text": text, "top_k": top_k}
    if exclude_id:
        body["exclude_id"] = exclude_id
    try:
        r = requests.post(f"{BACKEND_URL}/similar", json=body, timeout=30)
        if r.status_code == 200:
            return r.json()
    except requests.RequestException:
        pass
    return []


def call_backfill_embeddings() -> dict | None:
    try:
        r = requests.post(f"{BACKEND_URL}/backfill_embeddings", timeout=600)
        if r.status_code == 200:
            return r.json()
    except requests.RequestException:
        pass
    return None


def _fallback_result() -> dict:
    return {
        "id": None,
        "summary": "Customer is frustrated about delayed delivery and requests a refund.",
        "category": "Delivery",
        "severity": "High",
        "sentiment": "Negative",
        "response": ("We apologize for the delay and understand your frustration. "
                     "We are investigating your order and can issue a refund or replacement if needed."),
    }


def style_fig(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=TEXT, size=12),
        margin=dict(l=10, r=10, t=20, b=10),
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=-0.22,
                    xanchor="center", x=0.5,
                    font=dict(size=11, color=MUTED),
                    bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="rgba(148,163,184,0.06)",
                   linecolor="rgba(148,163,184,0.15)",
                   zerolinecolor="rgba(148,163,184,0.15)"),
        yaxis=dict(gridcolor="rgba(148,163,184,0.06)",
                   linecolor="rgba(148,163,184,0.15)",
                   zerolinecolor="rgba(148,163,184,0.15)"),
    )
    return fig


def severity_chip(sev: str) -> str:
    cls = {"Low": "chip-low", "Medium": "chip-medium", "High": "chip-high"}.get(sev, "chip-low")
    color = SEVERITY_COLORS.get(sev, MUTED)
    return (f'<span class="chip {cls}"><span class="dot" '
            f'style="background:{color};box-shadow:0 0 6px {color}"></span>{sev}</span>')


def sentiment_chip(sent: str) -> str:
    cls = {"Positive": "chip-pos", "Neutral": "chip-neu", "Negative": "chip-neg"}.get(sent, "chip-neu")
    return f'<span class="chip {cls}">{sent}</span>'


def status_chip(status: str) -> str:
    color = STATUS_COLORS.get(status, MUTED)
    return (f'<span class="chip" style="color:{color};border-color:{color}55;'
            f'background:{color}15"><span class="dot" style="background:{color};'
            f'box-shadow:0 0 6px {color}"></span>{status}</span>')


def pretty_time(iso: str | None) -> str:
    if not iso:
        return "—"
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except ValueError:
        return iso[:19]
    delta = datetime.now(timezone.utc) - dt
    if delta < timedelta(minutes=1):  return "just now"
    if delta < timedelta(hours=1):    return f"{int(delta.total_seconds() // 60)}m ago"
    if delta < timedelta(days=1):     return f"{int(delta.total_seconds() // 3600)}h ago"
    if delta < timedelta(days=7):     return f"{delta.days}d ago"
    return dt.strftime("%b %d, %Y")


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
status_cls, status_label = check_backend()
dot_class = {"ok": "dot-ok", "warn": "dot-warn", "err": "dot-err"}[status_cls]

st.markdown(f"""
<div class="hero">
  <div>
    <div class="eyebrow">Complaint Intelligence</div>
    <h1 class="hero-title">Read between<br>the <em>lines</em>.</h1>
    <p class="hero-sub">
      Turn unstructured customer complaints into routed, summarized, and ready-to-send
      replies. Built on a local LLM, designed for support teams that don't have time
      to read every ticket twice.
    </p>
  </div>
  <div class="hero-right">
    <span class="pill"><span class="dot {dot_class}"></span>{status_label}</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Load data once per render
# ---------------------------------------------------------------------------
complaints = fetch_all_complaints()
customers  = fetch_all_customers()
df = pd.DataFrame(complaints)
if not df.empty:
    df["created_dt"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_overview, tab_analyze, tab_bulk, tab_customers, tab_history = st.tabs(
    ["  Overview  ", "  Analyze  ", "  Bulk  ", "  Customers  ", "  History  "]
)

# ===========================================================================
# OVERVIEW
# ===========================================================================
with tab_overview:
    total = len(df)

    if total == 0:
        st.markdown("""
        <div class="empty">
            <h4>No complaints yet</h4>
            <p>Head to the <b>Analyze</b> tab — or batch-import a CSV from <b>Bulk</b> — to get started.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        high_count    = int((df["severity"] == "High").sum())
        neg_pct       = (df["sentiment"] == "Negative").mean() * 100
        unresolved    = int((df["status"] != "Resolved").sum())
        last_7_cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        recent_count  = int((df["created_dt"] >= last_7_cutoff).sum())

        st.markdown('<div class="section"><h3>At a glance</h3>'
                    f'<span class="sub">{total} total · last 7 days</span></div>',
                    unsafe_allow_html=True)

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""
            <div class="kpi">
                <div class="kpi-label">Total Complaints</div>
                <div class="kpi-value">{total}</div>
                <div class="kpi-meta"><span class="kpi-accent">{recent_count}</span> in the last 7 days</div>
            </div>""", unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="kpi">
                <div class="kpi-label">High Severity</div>
                <div class="kpi-value kpi-danger">{high_count}</div>
                <div class="kpi-meta">{(high_count/total*100):.0f}% of all tickets</div>
            </div>""", unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
            <div class="kpi">
                <div class="kpi-label">Negative Sentiment</div>
                <div class="kpi-value kpi-warn">{neg_pct:.0f}%</div>
                <div class="kpi-meta">customers expressing frustration</div>
            </div>""", unsafe_allow_html=True)
        with k4:
            st.markdown(f"""
            <div class="kpi">
                <div class="kpi-label">Open Queue</div>
                <div class="kpi-value">{unresolved}</div>
                <div class="kpi-meta">awaiting review or in progress</div>
            </div>""", unsafe_allow_html=True)

        # Charts
        st.markdown('<div class="section"><h3>Where the noise is coming from</h3>'
                    '<span class="sub">distribution</span></div>', unsafe_allow_html=True)

        c1, c2 = st.columns([1, 1.2])
        with c1:
            cat_df = (df.groupby("category").size()
                        .reset_index(name="count")
                        .sort_values("count", ascending=False))
            fig_cat = px.pie(cat_df, names="category", values="count", hole=0.62,
                             color="category", color_discrete_map=CATEGORY_COLORS)
            fig_cat.update_traces(
                textposition="outside", textinfo="label+percent",
                marker=dict(line=dict(color=BG, width=2)),
                hovertemplate="<b>%{label}</b><br>%{value} complaints<extra></extra>",
            )
            fig_cat.add_annotation(
                text=f"<b style='font-size:28px'>{total}</b><br>"
                     f"<span style='font-size:11px;color:{MUTED}'>TICKETS</span>",
                showarrow=False, font=dict(family="Fraunces, serif", color=TEXT),
            )
            st.plotly_chart(style_fig(fig_cat, height=340), use_container_width=True,
                            config={"displayModeBar": False})

        with c2:
            cross = (df.groupby(["severity", "sentiment"]).size()
                       .reset_index(name="count"))
            sev_order = ["Low", "Medium", "High"]
            cross["severity"] = pd.Categorical(cross["severity"], categories=sev_order, ordered=True)
            cross = cross.sort_values("severity")
            fig_sev = px.bar(
                cross, x="severity", y="count", color="sentiment",
                color_discrete_map=SENTIMENT_COLORS,
                category_orders={"severity": sev_order,
                                 "sentiment": ["Negative", "Neutral", "Positive"]},
            )
            fig_sev.update_traces(
                marker=dict(line=dict(width=0)),
                hovertemplate="<b>%{x}</b> · %{fullData.name}<br>%{y} complaints<extra></extra>",
            )
            fig_sev.update_layout(barmode="stack", xaxis_title=None, yaxis_title=None)
            st.plotly_chart(style_fig(fig_sev, height=340), use_container_width=True,
                            config={"displayModeBar": False})

        # Trend
        st.markdown('<div class="section"><h3>Volume over time</h3>'
                    '<span class="sub">last 30 days</span></div>', unsafe_allow_html=True)

        end_dt   = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        start_dt = end_dt - timedelta(days=29)
        days = pd.date_range(start_dt, end_dt, freq="D", tz="UTC")
        trend = (df.assign(day=df["created_dt"].dt.floor("D"))
                   .groupby("day").size()
                   .reindex(days, fill_value=0)
                   .reset_index())
        trend.columns = ["day", "count"]

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=trend["day"], y=trend["count"],
            mode="lines",
            line=dict(color=PRIMARY, width=2.5, shape="spline", smoothing=0.6),
            fill="tozeroy", fillcolor="rgba(167,139,250,0.12)",
            hovertemplate="%{x|%b %d}<br>%{y} complaints<extra></extra>",
            name="Complaints",
        ))
        fig_trend.update_layout(showlegend=False, xaxis_title=None, yaxis_title=None)
        st.plotly_chart(style_fig(fig_trend, height=260), use_container_width=True,
                        config={"displayModeBar": False})

        # Recent feed
        st.markdown('<div class="section"><h3>Most recent</h3>'
                    '<span class="sub">latest 5</span></div>', unsafe_allow_html=True)
        for c in complaints[:5]:
            cat_color = CATEGORY_COLORS.get(c["category"], MUTED)
            text_preview = (c["complaint"] or "")[:160]
            if len(c["complaint"] or "") > 160:
                text_preview += "…"
            cust_bit = (f' · <span style="color:{ACCENT}">{c["customer_name"]}</span>'
                        if c.get("customer_name") else "")
            st.markdown(f"""
            <div class="issue">
                <div class="issue-meta">
                    <span class="issue-id">#{c['id']:04d}</span>
                    <span style="color:{cat_color}">●</span>
                    <span>{c['category']}</span>{cust_bit}
                    <span>·</span>
                    {severity_chip(c['severity'])}
                    {sentiment_chip(c['sentiment'])}
                    {status_chip(c['status'])}
                    <span style="margin-left:auto">{pretty_time(c['created_at'])}</span>
                </div>
                <div class="issue-text">{text_preview}</div>
            </div>
            """, unsafe_allow_html=True)


# ===========================================================================
# ANALYZE
# ===========================================================================
with tab_analyze:
    st.markdown('<div class="section"><h3>New complaint</h3>'
                '<span class="sub">paste · analyze · review</span></div>',
                unsafe_allow_html=True)

    # Customer selector
    customer_labels = ["— No customer —"] + [
        f"{c['name']}" + (f" · {c['email']}" if c['email'] else "")
        for c in customers
    ]
    customer_choice = st.selectbox(
        "Customer (optional)", customer_labels,
        help="Linking a customer feeds their prior complaint history into the model "
             "so the suggested response is personalized.",
    )
    selected_customer = None
    if customer_choice != customer_labels[0]:
        selected_customer = customers[customer_labels.index(customer_choice) - 1]

    # Customer context preview
    if selected_customer:
        prior = fetch_customer_complaints(selected_customer["id"], limit=5)
        prior_html = ""
        if prior:
            prior_lines = []
            for p in prior[:3]:
                prior_lines.append(
                    f"<div style='font-size:12px;color:{MUTED};margin-top:4px'>"
                    f"<span style='color:{CATEGORY_COLORS.get(p['category'], MUTED)}'>●</span> "
                    f"{p['category']} ({p['severity']}) — {p['summary']}</div>"
                )
            prior_html = "".join(prior_lines)
            if len(prior) > 3:
                prior_html += (f"<div style='font-size:11px;color:{SUBTLE};margin-top:6px'>"
                               f"+ {len(prior) - 3} more</div>")
        else:
            prior_html = (f"<div style='font-size:12px;color:{MUTED};margin-top:4px'>"
                          f"First complaint with us.</div>")

        st.markdown(f"""
        <div class="ctx">
            <div class="ctx-name">{selected_customer['name']}</div>
            <div class="ctx-meta">
                {selected_customer['email'] or 'no email'} ·
                ${selected_customer['lifetime_value']:,.0f} LTV ·
                {selected_customer['complaint_count']} prior complaint{'s' if selected_customer['complaint_count'] != 1 else ''}
            </div>
            {prior_html}
        </div>
        """, unsafe_allow_html=True)

    # Sample selector
    examples = {
        "Choose a sample…": "",
        "Delayed delivery": "My package has been delayed for two weeks and customer service has not "
                            "responded. I want a refund immediately.",
        "Damaged product":  "I received my laptop today and the screen was cracked. This is unacceptable "
                            "and I need a replacement as soon as possible.",
        "Payment issue":    "I was charged twice for the same order and still have not received my refund.",
        "Poor service":     "The support agent was rude and did not help me solve my issue.",
    }
    choice = st.selectbox("Sample", list(examples.keys()), label_visibility="collapsed")

    complaint_text = st.text_area(
        "Complaint",
        value=examples[choice],
        height=180,
        placeholder="Paste a customer complaint, transcript, or chat message here…",
        label_visibility="collapsed",
        key="complaint_input",
    )
    char_count = len(complaint_text)
    st.markdown(f'<div style="text-align:right;color:{SUBTLE};font-size:11px;'
                f'font-family:JetBrains Mono,monospace;margin-top:-12px;margin-bottom:14px">'
                f'{char_count:,} chars</div>', unsafe_allow_html=True)

    if st.button("Analyze complaint", type="primary", use_container_width=True):
        if not complaint_text.strip():
            st.warning("Enter a complaint first.")
        else:
            with st.spinner("Reading the room…"):
                result, source = call_analyze(
                    complaint_text,
                    customer_id=selected_customer["id"] if selected_customer else None,
                )
            st.session_state["analysis_result"]    = result
            st.session_state["analysis_source"]    = source
            st.session_state["analysis_complaint"] = complaint_text
            st.session_state["analysis_id"]        = result.get("id")
            st.session_state["edited_response_value"] = result.get("response", "")

            # Pull similar past complaints (best-effort)
            if result.get("id"):
                with st.spinner("Finding similar past complaints…"):
                    sims = call_similar(complaint_text, top_k=3, exclude_id=result["id"])
                st.session_state["similar_complaints"] = sims
            else:
                st.session_state["similar_complaints"] = []

    # Results
    if "analysis_result" in st.session_state:
        result = st.session_state["analysis_result"]
        source = st.session_state["analysis_source"]

        if source == "live":
            st.success("Analysis complete.")
        else:
            st.warning("Showing fallback demo data — the backend is unreachable.")

        st.markdown('<div class="section"><h3>Analysis</h3>'
                    f'<span class="sub">model · {result.get("category","").lower()}</span></div>',
                    unsafe_allow_html=True)

        cat = result.get("category", "—")
        cat_color = CATEGORY_COLORS.get(cat, MUTED)

        st.markdown(f"""
        <div class="card" style="display:flex;gap:18px;align-items:center;flex-wrap:wrap">
            <div style="flex:1;min-width:180px">
                <div class="card-label">Category</div>
                <div style="display:flex;align-items:center;gap:10px;font-size:22px;
                            font-family:'Fraunces',serif;font-weight:600">
                    <span style="display:inline-block;width:10px;height:10px;border-radius:999px;
                                 background:{cat_color};box-shadow:0 0 10px {cat_color}"></span>
                    {cat}
                </div>
            </div>
            <div style="flex:1;min-width:180px">
                <div class="card-label">Severity</div>
                <div style="margin-top:4px">{severity_chip(result.get("severity","Low"))}</div>
            </div>
            <div style="flex:1;min-width:180px">
                <div class="card-label">Sentiment</div>
                <div style="margin-top:4px">{sentiment_chip(result.get("sentiment","Neutral"))}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        c_sum, c_resp = st.columns(2)
        with c_sum:
            st.markdown(f"""
            <div class="card">
                <div class="card-label">Summary</div>
                <div class="card-body">{result.get("summary","—")}</div>
            </div>""", unsafe_allow_html=True)
        with c_resp:
            st.markdown(f"""
            <div class="card">
                <div class="card-label">Suggested response</div>
                <div class="card-body">{result.get("response","—")}</div>
            </div>""", unsafe_allow_html=True)

        # Similar past complaints
        sims = st.session_state.get("similar_complaints", [])
        if sims:
            st.markdown('<div class="section"><h3>Similar past complaints</h3>'
                        f'<span class="sub">cosine similarity · top {len(sims)}</span></div>',
                        unsafe_allow_html=True)
            for s in sims:
                pct = max(0.0, min(1.0, float(s.get("similarity", 0))))
                cat_c = CATEGORY_COLORS.get(s["category"], MUTED)
                cust_bit = (f' · <span style="color:{ACCENT}">{s["customer_name"]}</span>'
                            if s.get("customer_name") else "")
                st.markdown(f"""
                <div class="sim">
                    <div class="issue-meta" style="margin-bottom:8px">
                        <span class="issue-id">#{s['id']:04d}</span>
                        <span style="color:{cat_c}">●</span>
                        <span>{s['category']}</span>{cust_bit}
                        <span>·</span>
                        {severity_chip(s['severity'])}
                        {status_chip(s['status'])}
                        <span style="margin-left:auto" class="sim-score">{pct*100:.0f}% match</span>
                    </div>
                    <div class="sim-bar" style="width:{pct*100:.1f}%"></div>
                    <div style="font-size:13px;color:{MUTED};margin-top:10px;line-height:1.5">
                        {s['summary']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                with st.expander(f"How #{s['id']:04d} was answered"):
                    st.write(s.get("response", "—"))
        elif st.session_state.get("analysis_id"):
            st.caption("No similar past complaints found yet — either nothing in the database "
                       "is close, or the embedding model isn't pulled (`ollama pull nomic-embed-text`).")

        # Review & send
        st.markdown('<div class="section"><h3>Review &amp; send</h3>'
                    '<span class="sub">human-in-the-loop</span></div>',
                    unsafe_allow_html=True)

        edited = st.text_area(
            "Edit before sending",
            value=st.session_state.get("edited_response_value", result.get("response", "")),
            height=140,
            key="edited_response_widget",
            label_visibility="collapsed",
        )

        if st.button("Save reviewed response", use_container_width=True):
            cid = st.session_state.get("analysis_id")
            if cid:
                update_complaint({
                    "id":        cid,
                    "complaint": st.session_state.get("analysis_complaint", ""),
                    "summary":   result.get("summary"),
                    "category":  result.get("category"),
                    "severity":  result.get("severity"),
                    "sentiment": result.get("sentiment"),
                    "response":  edited,
                })
                st.toast("Saved.", icon="✅")
            else:
                st.error("No complaint ID — analysis may have come from fallback data.")


# ===========================================================================
# BULK
# ===========================================================================
with tab_bulk:
    st.markdown('<div class="section"><h3>Bulk import</h3>'
                '<span class="sub">CSV · batch analyze</span></div>',
                unsafe_allow_html=True)

    st.markdown(
        f'<p style="color:{MUTED};font-size:14px;margin-bottom:14px">'
        f'Drop a CSV with at least a complaints column. Optionally include a column '
        f'with customer emails to auto-link to existing customers.</p>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "CSV", type=["csv"], label_visibility="collapsed",
        key="bulk_csv_uploader",
    )

    if uploaded:
        try:
            df_upload = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            df_upload = None

        if df_upload is not None and len(df_upload) > 0:
            st.markdown(f'<div class="card-label" style="margin-top:14px">Preview '
                        f'({len(df_upload)} rows)</div>', unsafe_allow_html=True)
            st.dataframe(df_upload.head(5), use_container_width=True, height=220)

            cols = list(df_upload.columns)

            cc1, cc2 = st.columns(2)
            with cc1:
                default_idx = cols.index("complaint") if "complaint" in cols else 0
                complaint_col = st.selectbox(
                    "Complaint text column",
                    cols, index=default_idx, key="bulk_complaint_col",
                )
            with cc2:
                email_options = ["— don't link customers —"] + cols
                # Auto-detect email-like column
                default_email_idx = 0
                for i, col in enumerate(cols, start=1):
                    if "email" in col.lower():
                        default_email_idx = i
                        break
                customer_col_choice = st.selectbox(
                    "Customer email column (optional)",
                    email_options, index=default_email_idx,
                    key="bulk_customer_col",
                )
            customer_col = None if customer_col_choice == email_options[0] else customer_col_choice

            # Calculate linkability
            customers_by_email = {
                (c["email"] or "").lower().strip(): c["id"]
                for c in customers if c["email"]
            }
            total_rows = len(df_upload)
            linkable = 0
            if customer_col:
                emails = df_upload[customer_col].astype(str).str.lower().str.strip()
                linkable = int(sum(1 for e in emails if e in customers_by_email))

            link_msg = ""
            if customer_col:
                link_msg = (f', linking <b style="color:{TEXT}">{linkable}</b> '
                            f'to existing customers')

            st.markdown(
                f'<div style="color:{MUTED};font-size:13px;margin:14px 0">'
                f'Will analyze <b style="color:{TEXT}">{total_rows}</b> '
                f'complaint{"s" if total_rows != 1 else ""}{link_msg}.'
                f'</div>',
                unsafe_allow_html=True,
            )

            if total_rows > 0 and st.button(
                f"Analyze {total_rows} complaint{'s' if total_rows != 1 else ''}",
                type="primary", use_container_width=True, key="bulk_run",
            ):
                progress = st.progress(0.0, text="Starting…")
                ok, fail, linked = 0, 0, 0

                for i, row in enumerate(df_upload.itertuples(index=False)):
                    text = str(getattr(row, complaint_col, "") or "").strip()
                    if not text:
                        fail += 1
                        progress.progress((i + 1) / total_rows,
                                          text=f"Skipping empty row {i+1} of {total_rows}")
                        continue

                    cid = None
                    if customer_col:
                        email = str(getattr(row, customer_col, "") or "").lower().strip()
                        cid = customers_by_email.get(email)
                        if cid:
                            linked += 1

                    try:
                        body = {"complaint": text}
                        if cid:
                            body["customer_id"] = cid
                        r = requests.post(f"{BACKEND_URL}/analyze", json=body, timeout=180)
                        if r.status_code == 200:
                            ok += 1
                        else:
                            fail += 1
                    except requests.RequestException:
                        fail += 1

                    progress.progress(
                        (i + 1) / total_rows,
                        text=f"Processed {i+1} of {total_rows} · "
                             f"{ok} ok, {fail} failed",
                    )

                progress.empty()
                if ok:
                    msg = f"Analyzed {ok} complaint{'s' if ok != 1 else ''}."
                    if linked:
                        msg += f" Linked {linked} to existing customers."
                    if fail:
                        msg += f" {fail} failed."
                    st.success(msg)
                else:
                    st.error(f"All {fail} complaints failed to analyze. "
                             f"Check that the backend and Ollama are running.")

    # Embeddings backfill
    st.markdown('<div class="section"><h3>Embedding maintenance</h3>'
                '<span class="sub">similarity index</span></div>',
                unsafe_allow_html=True)

    pending = count_complaints_without_embeddings()
    if pending == 0:
        st.markdown(
            f'<div style="color:{MUTED};font-size:13px">'
            f'All complaints have embeddings. Similarity search is fully indexed.'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div style="color:{MUTED};font-size:13px;margin-bottom:10px">'
            f'<b style="color:{TEXT}">{pending}</b> complaint{"s" if pending != 1 else ""} '
            f'predate the similarity feature and have no embedding yet. '
            f'Generating them populates the search index.'
            f'</div>',
            unsafe_allow_html=True,
        )
        if st.button(f"Generate embeddings for {pending} older complaint{'s' if pending != 1 else ''}",
                     use_container_width=True, key="backfill_btn"):
            with st.spinner("Generating embeddings…"):
                res = call_backfill_embeddings()
            if res:
                st.success(
                    f"Processed {res['processed']} of {res['total']}. "
                    f"{res.get('failed', 0)} failed."
                )
                st.rerun()
            else:
                st.error("Backfill failed. Make sure the embedding model is pulled "
                         "(`ollama pull nomic-embed-text`).")


# ===========================================================================
# CUSTOMERS
# ===========================================================================
with tab_customers:
    st.markdown('<div class="section"><h3>Customers</h3>'
                f'<span class="sub">{len(customers)} on file</span></div>',
                unsafe_allow_html=True)

    # New customer form
    with st.expander("Add new customer", expanded=(len(customers) == 0)):
        with st.form("new_customer", clear_on_submit=True):
            n1, n2 = st.columns(2)
            with n1:
                new_name = st.text_input("Name *", key="new_cust_name")
            with n2:
                new_email = st.text_input("Email", key="new_cust_email")
            n3, n4 = st.columns([1, 2])
            with n3:
                new_ltv = st.number_input("Lifetime value ($)", min_value=0.0,
                                          step=100.0, key="new_cust_ltv")
            with n4:
                new_notes = st.text_input("Notes", key="new_cust_notes")
            submitted = st.form_submit_button("Create customer", type="primary")
            if submitted:
                if not new_name.strip():
                    st.error("Name is required.")
                else:
                    create_customer(
                        new_name.strip(),
                        new_email.strip() or None,
                        float(new_ltv),
                        new_notes.strip() or None,
                    )
                    st.toast("Customer created.", icon="✅")
                    st.rerun()

    if not customers:
        st.markdown("""
        <div class="empty">
            <h4>No customers yet</h4>
            <p>Add customers above to link complaints and give the AI better context.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        search_cust = st.text_input(
            "Search customers", placeholder="Search by name or email…",
            label_visibility="collapsed", key="customer_search",
        )

        filtered = customers
        if search_cust.strip():
            q = search_cust.lower().strip()
            filtered = [c for c in customers
                        if q in (c["name"] or "").lower()
                        or q in (c["email"] or "").lower()]

        st.markdown(
            f'<div style="color:{SUBTLE};font-size:12px;'
            f'font-family:JetBrains Mono,monospace;margin:6px 0 14px 0">'
            f'{len(filtered)} matching</div>',
            unsafe_allow_html=True,
        )

        for cust in filtered:
            email_html = (f'<span>{cust["email"]}</span>' if cust["email"]
                          else f'<span style="color:{SUBTLE}">no email</span>')
            notes_html = (f'<div class="issue-summary">{cust["notes"]}</div>'
                          if cust["notes"] else "")

            st.markdown(f"""
            <div class="issue">
                <div class="issue-meta">
                    <span class="issue-id">CUST-{cust['id']:04d}</span>
                    <span>·</span>
                    <span style="color:{TEXT};font-weight:600;font-family:Inter">{cust['name']}</span>
                    <span>·</span>
                    {email_html}
                    <span style="margin-left:auto">
                        {cust['complaint_count']} complaint{'s' if cust['complaint_count'] != 1 else ''}
                        · ${cust['lifetime_value']:,.0f} LTV
                    </span>
                </div>
                {notes_html}
            </div>
            """, unsafe_allow_html=True)

            cc1, cc2, cc3 = st.columns([1, 1, 0.5])
            with cc1:
                with st.expander(f"View {cust['complaint_count']} "
                                 f"complaint{'s' if cust['complaint_count'] != 1 else ''}"):
                    cust_complaints = fetch_customer_complaints(cust["id"], limit=20)
                    if not cust_complaints:
                        st.caption("No complaints yet.")
                    else:
                        for cc in cust_complaints:
                            cat_color = CATEGORY_COLORS.get(cc["category"], MUTED)
                            st.markdown(
                                f"**#{cc['id']:04d}** · "
                                f"<span style='color:{cat_color}'>●</span> "
                                f"{cc['category']} ({cc['severity']}) · "
                                f"{pretty_time(cc['created_at'])}",
                                unsafe_allow_html=True,
                            )
                            st.caption(cc["summary"])

            with cc2:
                with st.expander("Edit"):
                    with st.form(f"edit_customer_{cust['id']}"):
                        e_name = st.text_input("Name", value=cust["name"])
                        e_email = st.text_input("Email", value=cust["email"] or "")
                        e_ltv = st.number_input(
                            "LTV ($)", value=float(cust["lifetime_value"]),
                            min_value=0.0, step=100.0,
                        )
                        e_notes = st.text_area("Notes", value=cust["notes"] or "")
                        if st.form_submit_button("Save", type="primary"):
                            update_customer(
                                cust["id"], e_name.strip(),
                                e_email.strip() or None,
                                float(e_ltv),
                                e_notes.strip() or None,
                            )
                            st.toast("Customer updated.", icon="✅")
                            st.rerun()

            with cc3:
                if st.button("Delete", key=f"delcust_{cust['id']}",
                             use_container_width=True):
                    delete_customer(cust["id"])
                    st.toast("Customer deleted.", icon="🗑️")
                    st.rerun()


# ===========================================================================
# HISTORY
# ===========================================================================
with tab_history:
    st.markdown('<div class="section"><h3>All complaints</h3>'
                f'<span class="sub">{len(complaints)} records</span></div>',
                unsafe_allow_html=True)

    if not complaints:
        st.markdown("""
        <div class="empty">
            <h4>Nothing here yet</h4>
            <p>Analyzed complaints show up here with their full audit trail.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        f1, f2, f3, f4 = st.columns([1.6, 1, 1, 1])
        with f1:
            search = st.text_input("Search", placeholder="Search complaint text…",
                                   label_visibility="collapsed", key="hist_search")
        with f2:
            f_status = st.selectbox(
                "Status",
                ["All statuses", "Needs Review", "In Progress", "Resolved"],
                label_visibility="collapsed", key="hist_status",
            )
        with f3:
            f_sev = st.selectbox(
                "Severity",
                ["All severities", "High", "Medium", "Low"],
                label_visibility="collapsed", key="hist_sev",
            )
        with f4:
            f_cat = st.selectbox(
                "Category",
                ["All categories", "Delivery", "Product", "Payment", "Service"],
                label_visibility="collapsed", key="hist_cat",
            )

        filtered = complaints
        if search.strip():
            q = search.lower().strip()
            filtered = [c for c in filtered
                        if q in (c["complaint"] or "").lower()
                        or q in (c["summary"] or "").lower()
                        or q in (c.get("customer_name") or "").lower()]
        if f_status != "All statuses":
            filtered = [c for c in filtered if c["status"] == f_status]
        if f_sev != "All severities":
            filtered = [c for c in filtered if c["severity"] == f_sev]
        if f_cat != "All categories":
            filtered = [c for c in filtered if c["category"] == f_cat]

        st.markdown(
            f'<div style="color:{SUBTLE};font-size:12px;'
            f'font-family:JetBrains Mono,monospace;margin:6px 0 14px 0">'
            f'{len(filtered)} matching</div>',
            unsafe_allow_html=True,
        )

        for c in filtered:
            cat_color = CATEGORY_COLORS.get(c["category"], MUTED)
            cust_bit = (f' · <span style="color:{ACCENT}">{c["customer_name"]}</span>'
                        if c.get("customer_name") else "")
            with st.container():
                st.markdown(f"""
                <div class="issue">
                    <div class="issue-meta">
                        <span class="issue-id">#{c['id']:04d}</span>
                        <span style="color:{cat_color}">●</span>
                        <span>{c['category']}</span>{cust_bit}
                        <span>·</span>
                        {severity_chip(c['severity'])}
                        {sentiment_chip(c['sentiment'])}
                        {status_chip(c['status'])}
                        <span style="margin-left:auto">{pretty_time(c['created_at'])}</span>
                    </div>
                    <div class="issue-text">{c['complaint']}</div>
                    <div class="issue-summary">{c['summary']}</div>
                </div>
                """, unsafe_allow_html=True)

                a1, a2, a3 = st.columns([1.2, 1.2, 0.6])
                with a1:
                    new_status = st.selectbox(
                        "Status",
                        ["Needs Review", "In Progress", "Resolved"],
                        index=["Needs Review", "In Progress", "Resolved"].index(c["status"]),
                        key=f"status_{c['id']}",
                        label_visibility="collapsed",
                    )
                    if new_status != c["status"]:
                        update_status(c["id"], new_status)
                        st.rerun()
                with a2:
                    with st.expander("View suggested response"):
                        st.write(c["response"])
                with a3:
                    if st.button("Delete", key=f"del_{c['id']}",
                                 use_container_width=True):
                        delete_complaint(c["id"])
                        st.rerun()