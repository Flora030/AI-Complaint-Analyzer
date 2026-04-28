import os
import pandas as pd
import requests
import streamlit as st
from database import save_complaint
import sqlite3

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="AI Complaint Analyzer",
    layout="wide",
)

st.markdown("""
<style>
.header-container {
    padding: 36px 44px;
    border-radius: 18px;
    background: linear-gradient(135deg, #111827, #1f2937);
    border: 1px solid #374151;
    margin-bottom: 25px;
}

.main-title {
    font-size: 54px;
    font-weight: 900;
    letter-spacing: -1px;
    margin-bottom: 12px;
}

.subtitle {
    font-size: 18px;
    color: #cbd5e1;
    max-width: 900px;
    line-height: 1.6;
}

.section-title {
    font-size: 27px;
    font-weight: 800;
    margin-top: 10px;
    margin-bottom: 15px;
}

.card {
    padding: 20px;
    border-radius: 14px;
    background-color: #1e1e2f;
    border: 1px solid #374151;
    margin-bottom: 14px;
    line-height: 1.6;
}

.badge-high {
    color: #fecaca;
    background-color: #7f1d1d;
    padding: 6px 12px;
    border-radius: 999px;
    font-weight: 700;
}

.badge-medium {
    color: #fef3c7;
    background-color: #78350f;
    padding: 6px 12px;
    border-radius: 999px;
    font-weight: 700;
}

.badge-low {
    color: #d1fae5;
    background-color: #064e3b;
    padding: 6px 12px;
    border-radius: 999px;
    font-weight: 700;
}

.demo-banner {
    padding: 10px 14px;
    border-radius: 10px;
    background-color: #422006;
    border: 1px solid #92400e;
    color: #fde68a;
    margin-bottom: 14px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-container">
    <div class="main-title">AI Complaint Analyzer</div>
    <div class="subtitle">
        A generative AI proof of concept for e-commerce customer service.
        The system analyzes unstructured complaints, identifies issue category, severity, and sentiment,
        summarizes the customer concern, and generates a professional suggested response.
    </div>
</div>
""", unsafe_allow_html=True)

# Backend call for analyzing complaint
def analyze_complaint(complaint_text: str) -> tuple[dict, str]:
    try:
        r = requests.post(f"{BACKEND_URL}/analyze", json={"complaint": complaint_text}, timeout=180)
        if r.status_code == 200:
            return r.json(), "live"
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        st.error(f"Backend returned {r.status_code}: {detail}")
        return _fallback_result(), "fallback"
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot reach the backend at {BACKEND_URL}. Make sure `uvicorn backend:app --port 8000` is running and Ollama is started.")
        return _fallback_result(), "fallback"
    except requests.exceptions.Timeout:
        st.error("Backend request timed out. Try a smaller model or increase REQUEST_TIMEOUT.")
        return _fallback_result(), "fallback"

def _fallback_result() -> dict:
    return {
        "summary": "Customer is frustrated about delayed delivery and requests a refund.",
        "category": "Delivery",
        "severity": "High",
        "sentiment": "Negative",
        "response": "We apologize for the delay and understand your frustration. We are investigating your order and can issue a refund or replacement if needed."
    }

def fetch_complaints():
    try:
        conn = sqlite3.connect("complaints.db") 
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM complaints ORDER BY id DESC")
        rows = cursor.fetchall()

        conn.close()

        complaints = [
            {
                "id": row[0],
                "complaint": row[1],
                "summary": row[2],
                "category": row[3],
                "severity": row[4],
                "sentiment": row[5],
                "response": row[6],
            }
            for row in rows
        ]

        return complaints

    except sqlite3.Error as e:
        st.error(f"Database error: {e}")
        return []
    
# Tabs for Dashboard, Analyze Complaint, History
tab1, tab2, tab3 = st.tabs(["Dashboard", "Analyze Complaint", "Complaint History"])

history_data = fetch_complaints()
#print("145 Complaints History:", history_data)
total_complaints = len(history_data)
high_severity = sum(1 for complaint in history_data if complaint['severity'] == 'High')
negative_sentiment = sum(1 for complaint in history_data if complaint['sentiment'] == 'Negative')
avg_time_saved = 3 

# Dashboard Tab
with tab1:
    st.markdown('<div class="section-title">Complaint Dashboard</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Complaints", str(total_complaints))
    col2.metric("High Severity", str(high_severity))  
    if total_complaints > 0:
        negative_sentiment_percentage = (negative_sentiment / total_complaints) * 100
    else:
        negative_sentiment_percentage = 0 
    col3.metric("Negative Sentiment", f"{negative_sentiment_percentage:.2f}%")
    col4.metric("Avg Time Saved", f"{avg_time_saved} min")  

    st.divider()

    left_chart, right_chart = st.columns(2)
    with left_chart:
        st.markdown("### Complaints by Category")
        # Category count
        category_counts = pd.DataFrame({
            "Category": ["Delivery", "Product", "Payment", "Service"],
            "Complaints": [sum(1 for complaint in history_data if complaint['category'] == category) for category in ["Delivery", "Product", "Payment", "Service"]],
        }).set_index("Category")
        st.bar_chart(category_counts)
    
    with right_chart:
        st.markdown("### Severity Breakdown")
        # Severity count
        severity_counts = pd.DataFrame({
            "Severity": ["Low", "Medium", "High"],
            "Count": [sum(1 for complaint in history_data if complaint['severity'] == severity) for severity in ["Low", "Medium", "High"]],
        }).set_index("Severity")
        st.bar_chart(severity_counts)

    st.divider()

    st.markdown('<div class="section-title">Business Impact</div>', unsafe_allow_html=True)
    b1, b2, b3 = st.columns(3)
    with b1:
        st.markdown("<div class='card'><b>Faster Processing</b><br>Reduces the time support teams spend manually reading and categorizing complaints.</div>", unsafe_allow_html=True)
    with b2:
        st.markdown("<div class='card'><b>Better Prioritization</b><br>Flags high-severity complaints so urgent customer issues can be handled first.</div>", unsafe_allow_html=True)
    with b3:
        st.markdown("<div class='card'><b>Consistent Responses</b><br>Helps customer service teams provide professional and standardized replies.</div>", unsafe_allow_html=True)

# Analyze Tab
with tab2:
    st.markdown('<div class="section-title">Analyze a New Complaint</div>', unsafe_allow_html=True)

    example_complaints = {
        "Select an example": "",
        "Delayed delivery": "My package has been delayed for two weeks and customer service has not responded. I want a refund immediately.",
        "Damaged product": "I received my laptop today and the screen was cracked. This is unacceptable and I need a replacement as soon as possible.",
        "Payment issue": "I was charged twice for the same order and still have not received my refund.",
        "Poor service": "The support agent was rude and did not help me solve my issue.",
    }

    selected_example = st.selectbox("Sample complaint", list(example_complaints.keys()))

    complaint = st.text_area("Enter customer complaint", value=example_complaints[selected_example], height=170, placeholder="Type or paste a customer complaint here...", key="complaint_input")

    if st.button("Analyze Complaint", width="stretch"):
        if not complaint.strip():
            st.warning("Please enter a complaint.")
        else:
            with st.spinner("Processing complaint..."):
                result, source = analyze_complaint(complaint)
            st.session_state["analysis_result"] = result
            st.session_state["analysis_source"] = source
            st.session_state["analysis_complaint"] = complaint

            # Resetting the session state for edited_response after analyzing a new complaint
            if "edited_response" in st.session_state:
                del st.session_state["edited_response"]

            # Set the AI response as the default in case of a new complaint
            st.session_state["edited_response"] = result.get("response", "")  # AI response as default

    if "analysis_result" in st.session_state:
        result = st.session_state["analysis_result"]
        source = st.session_state["analysis_source"]

        if source == "live":
            st.success("Analysis complete")
        else:
            st.markdown('<div class="demo-banner">Showing fallback demo data because the backend is unreachable.</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-title">AI Analysis Results</div>', unsafe_allow_html=True)

        r1, r2, r3 = st.columns(3)
        r1.metric("Category", result.get("category", "N/A"))
        r2.metric("Severity", result.get("severity", "N/A"))
        r3.metric("Sentiment", result.get("sentiment", "N/A"))

        severity = result.get("severity", "Low")
        badge_class = {
            "High": "badge-high",
            "Medium": "badge-medium",
            "Low": "badge-low",
        }.get(severity, "badge-low")

        st.markdown(f'<p><span class="{badge_class}">Severity: {severity}</span></p>', unsafe_allow_html=True)

        st.markdown(f"<div class='card'><b>Complaint Summary</b><br>{result.get('summary', 'No summary generated.')}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='card'><b>Suggested Customer Response</b><br>{result.get('response', 'No response generated.')}</div>", unsafe_allow_html=True)

        st.markdown("### Human Review")
        edited_response = st.text_area(
            "Support agent can edit the AI response before sending", 
            value=st.session_state["edited_response"], 
            height=130, 
            key="edited_response", 
            width="stretch"
        )

        if st.button("Save Reviewed Response", width="stretch"):
            # Ensure only the edited response is saved
            saved_data = {
                "complaint": st.session_state.get("analysis_complaint", ""),
                "summary": result.get("summary"),
                "category": result.get("category"),
                "severity": result.get("severity"),
                "sentiment": result.get("sentiment"),
                "response": edited_response  # Save the human-edited response
            }
            save_complaint(saved_data)  
            st.toast("Reviewed response saved.", icon="✅")

# History Tab
with tab3:
    st.markdown('<div class="section-title">Recent Complaint History</div>', unsafe_allow_html=True)

    history_data = fetch_complaints()
    
    status_filter = st.selectbox("Filter by status", ["All", "Needs Review", "In Progress", "Resolved"])

    filtered_data = (
        history_data
        if status_filter == "All"
        else [complaint for complaint in history_data if complaint['status'] == status_filter]
    )

    st.dataframe(pd.DataFrame(filtered_data), use_container_width=True)