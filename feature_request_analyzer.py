
import streamlit as st
import pandas as pd
import json

st.set_page_config(page_title="Feature Request Analyzer", layout="wide")
st.title("🤖 Intelligent Feature Request Analyzer")

st.markdown("Upload a CSV with a column of feature requests. This app sends them to GPT-4 and categorizes them into themes with impact levels.")

# Input: Upload CSV
uploaded_file = st.file_uploader("Upload a CSV of feature requests", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("📋 Uploaded Data Preview:")
    st.dataframe(df.head())

    if st.button("Analyze with AI"):
        requests_text = "\n".join(df.iloc[:, 0].dropna().astype(str).tolist())

        prompt = f'''
You are an AI product analyst. Categorize the following customer feature requests into themes and assess business impact (low/medium/high). Output in a JSON list like:

[
  {{
    "theme": "Exporting Data",
    "requests": [...],
    "frequency": "High",
    "sentiment": "Positive",
    "business_impact": "High"
  }},
  ...
]

Requests:
{requests_text}
        '''

        st.code(prompt, language='markdown')
        st.info("🔑 Replace this section with your OpenAI API call to generate categorized results using the prompt above.")

        # Placeholder for AI output
        st.markdown("**💡 Example Output Structure:**")
        example_json = [
            {
                "theme": "Exporting Data",
                "requests": [
                    "Can I export results to CSV?",
                    "Need a PDF export option."
                ],
                "frequency": "High",
                "sentiment": "Neutral",
                "business_impact": "High"
            },
            {
                "theme": "Integrations",
                "requests": [
                    "Would love a Klaviyo integration.",
                    "Can you add Mailchimp?"
                ],
                "frequency": "Medium",
                "sentiment": "Positive",
                "business_impact": "Medium"
            }
        ]
        st.json(example_json)
