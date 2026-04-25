import streamlit as st
import requests

st.set_page_config(
    page_title="AI Complaint Analyzer",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍️ AI Complaint Analyzer")
st.write("Analyze e-commerce customer complaints using generative AI.")

complaint = st.text_area(
    "Enter customer complaint:",
    height=180,
    placeholder="Example: My package has been delayed for two weeks and customer service has not responded."
)

if st.button("Analyze Complaint"):
    if not complaint.strip():
        st.warning("Please enter a complaint first.")
    else:
        with st.spinner("Analyzing complaint..."):
            try:
                response = requests.post(
                    "http://localhost:8000/analyze",
                    json={"complaint": complaint}
                )

                if response.status_code == 200:
                    result = response.json()

                    st.subheader("Analysis Results")

                    col1, col2, col3 = st.columns(3)

                    col1.metric("Category", result.get("category", "N/A"))
                    col2.metric("Severity", result.get("severity", "N/A"))
                    col3.metric("Sentiment", result.get("sentiment", "N/A"))

                    st.markdown("### Summary")
                    st.info(result.get("summary", "No summary generated."))

                    st.markdown("### Suggested Response")
                    st.success(result.get("response", "No suggested response generated."))

                else:
                    st.error("Backend error. Make sure the backend is running.")

            except requests.exceptions.ConnectionError:
                st.error("Could not connect to backend. Make sure backend is running at localhost:8000.")