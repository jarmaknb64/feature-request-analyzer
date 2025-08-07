import streamlit as st
import pandas as pd
import openai
import json

# Load API key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.title("Feature Request Analyzer")

uploaded_file = st.file_uploader("Upload a CSV with a 'Requests' column", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Detected request column:", df.columns.tolist())
    st.dataframe(df)

    if st.button("Analyze"):
        requests_text = df["Requests"].tolist()
        prompt = f"Analyze these feature requests: {requests_text}"
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            st.write("Analysis:", response.choices[0].message["content"])
        except Exception as e:
            st.error(f"Error calling OpenAI API: {e}")
