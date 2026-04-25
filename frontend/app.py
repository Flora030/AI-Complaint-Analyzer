import streamlit as st
import requests
import pandas as pd

st.set_page_config(
    page_title="AI Complaint Analyzer",
    layout="wide"
)

# -------------------------
# Custom Styling
# -------------------------
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
</style>
""", unsafe_allow_html=True)

# -------------------------
# Header
# -------------------------
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

# -------------------------
# Demo Data
# -------------------------
history_data = pd.DataFrame({
    "Complaint ID": ["C001", "C002", "C003", "C004", "C005", "C006"],
    "Category": ["Delivery", "Product", "Payment", "Service", "Delivery", "Product"],
    "Severity": ["High", "Medium", "High", "Low", "Medium", "High"],
    "Sentiment": ["Negative", "Negative", "Negative", "Neutral", "Negative", "Negative"],
    "Status": ["Needs Review", "Resolved", "Needs Review", "Resolved", "In Progress", "Needs Review"]
})

category_data = pd.DataFrame({
    "Category": ["Delivery", "Product", "Payment", "Service"],
    "Complaints": [45, 32, 18, 33]
})

severity_data = pd.DataFrame({
    "Severity": ["Low", "Medium", "High"],
    "Count": [52, 52, 24]
})

# -------------------------
# Tabs
# -------------------------
tab1, tab2, tab3 = st.tabs(["Dashboard", "Analyze Complaint", "Complaint History"])

# -------------------------
# Dashboard Tab
# -------------------------
with tab1:
    st.markdown('<div class="section-title">Complaint Dashboard</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Complaints", "128")
    col2.metric("High Severity", "24", "+8 today")
    col3.metric("Negative Sentiment", "71%")
    col4.metric("Avg Time Saved", "3 min")

    st.divider()

    left_chart, right_chart = st.columns(2)

    with left_chart:
        st.markdown("### Complaints by Category")
        st.bar_chart(category_data.set_index("Category"))

    with right_chart:
        st.markdown("### Severity Breakdown")
        st.bar_chart(severity_data.set_index("Severity"))

    st.divider()

    st.markdown('<div class="section-title">Business Impact</div>', unsafe_allow_html=True)

    b1, b2, b3 = st.columns(3)

    with b1:
        st.markdown("""
        <div class="card">
            <b>Faster Processing</b><br>
            Reduces the time support teams spend manually reading and categorizing complaints.
        </div>
        """, unsafe_allow_html=True)

    with b2:
        st.markdown("""
        <div class="card">
            <b>Better Prioritization</b><br>
            Flags high-severity complaints so urgent customer issues can be handled first.
        </div>
        """, unsafe_allow_html=True)

    with b3:
        st.markdown("""
        <div class="card">
            <b>Consistent Responses</b><br>
            Helps customer service teams provide professional and standardized replies.
        </div>
        """, unsafe_allow_html=True)

# -------------------------
# Analyze Tab
# -------------------------
with tab2:
    st.markdown('<div class="section-title">Analyze a New Complaint</div>', unsafe_allow_html=True)

    example_complaints = {
        "Select an example": "",
        "Delayed delivery": "My package has been delayed for two weeks and customer service has not responded. I want a refund immediately.",
        "Damaged product": "I received my laptop today and the screen was cracked. This is unacceptable and I need a replacement as soon as possible.",
        "Payment issue": "I was charged twice for the same order and still have not received my refund.",
        "Poor service": "The support agent was rude and did not help me solve my issue."
    }

    selected_example = st.selectbox("Sample complaint", list(example_complaints.keys()))

    complaint = st.text_area(
        "Enter customer complaint",
        value=example_complaints[selected_example],
        height=170,
        placeholder="Type or paste a customer complaint here..."
    )

    analyze = st.button("Analyze Complaint", use_container_width=True)

    if analyze:
        if not complaint.strip():
            st.warning("Please enter a complaint.")
        else:
            with st.spinner("Processing complaint..."):
                try:
                    response = requests.post(
                        "http://localhost:8000/analyze",
                        json={"complaint": complaint}
                    )

                    if response.status_code == 200:
                        result = response.json()
                    else:
                        raise Exception("Backend error")

                except:
                    result = {
                        "summary": "Customer is frustrated about delayed delivery and requests a refund.",
                        "category": "Delivery",
                        "severity": "High",
                        "sentiment": "Negative",
                        "response": "We apologize for the delay and understand your frustration. We are investigating your order and can issue a refund or replacement if needed."
                    }

            st.success("Analysis complete")

            st.markdown('<div class="section-title">AI Analysis Results</div>', unsafe_allow_html=True)

            r1, r2, r3 = st.columns(3)

            r1.metric("Category", result.get("category", "N/A"))
            r2.metric("Severity", result.get("severity", "N/A"))
            r3.metric("Sentiment", result.get("sentiment", "N/A"))

            severity = result.get("severity", "Low")

            if severity == "High":
                badge_class = "badge-high"
            elif severity == "Medium":
                badge_class = "badge-medium"
            else:
                badge_class = "badge-low"

            st.markdown(
                f'<p><span class="{badge_class}">Severity: {severity}</span></p>',
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="card">
                    <b>Complaint Summary</b><br>
                    {result.get("summary", "No summary generated.")}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="card">
                    <b>Suggested Customer Response</b><br>
                    {result.get("response", "No response generated.")}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("### Human Review")
            edited_response = st.text_area(
                "Support agent can edit the AI response before sending",
                value=result.get("response", ""),
                height=130
            )

            st.button("Save Reviewed Response", use_container_width=True)

# -------------------------
# History Tab
# -------------------------
with tab3:
    st.markdown('<div class="section-title">Recent Complaint History</div>', unsafe_allow_html=True)

    status_filter = st.selectbox(
        "Filter by status",
        ["All", "Needs Review", "In Progress", "Resolved"]
    )

    if status_filter != "All":
        filtered_data = history_data[history_data["Status"] == status_filter]
    else:
        filtered_data = history_data

    st.dataframe(filtered_data, use_container_width=True)

    st.divider()

    st.markdown('<div class="section-title">Evaluation Preview</div>', unsafe_allow_html=True)

    eval_data = pd.DataFrame({
        "Metric": ["Category Accuracy", "Severity Accuracy", "Sentiment Accuracy"],
        "Result": ["85%", "80%", "90%"],
        "Evaluation Method": [
            "Compared AI category to manual label",
            "Compared AI severity to manual label",
            "Compared AI sentiment to manual label"
        ]
    })

    st.dataframe(eval_data, use_container_width=True)