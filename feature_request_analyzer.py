
import streamlit as st
import pandas as pd
import json
from openai import OpenAI

st.set_page_config(page_title="Feature Request Analyzer", layout="wide")
st.title("🤖 Intelligent Feature Request Analyzer")

st.markdown("Upload a CSV with a single column of feature requests (one per row). The app groups them into themes and estimates frequency and business impact.")

# File upload
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

def call_openai(prompt: str) -> str:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[
            {"role": "system", "content": "You are an AI product analyst who writes concise, valid JSON only."},
            {"role": "user", "content": prompt}
        ]
    )
    return resp.choices[0].message.content

def build_prompt(requests_text: str) -> str:
    return f"""
Read the following customer feature requests. Cluster them into coherent THEMES.
For each theme, return:
- "theme": short theme name
- "requests": list of representative original lines (3-6 max)
- "frequency": Low | Medium | High (based on count share)
- "sentiment": overall tone toward product: Positive | Neutral | Negative
- "business_impact": Low | Medium | High (estimate based on frequency + likely revenue impact in a Shopify A/B testing tool)

Return ONLY a JSON array. No commentary. Example shape:
[{"theme":"Exporting Data","requests":["..."],"frequency":"High","sentiment":"Neutral","business_impact":"High"}]

Requests:
{requests_text}
""".strip()

def normalize_to_df(items):
    # Flatten into a nice table
    rows = []
    for it in items:
        rows.append({
            "Theme": it.get("theme", ""),
            "Frequency": it.get("frequency", ""),
            "Impact": it.get("business_impact", ""),
            "Sentiment": it.get("sentiment", ""),
            "Examples": " • ".join(it.get("requests", [])[:5])
        })
    return pd.DataFrame(rows)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    # Use the first column by default
    col = df.columns[0]
    st.caption(f"Detected request column: **{col}**")
    st.dataframe(df.head())

    if st.button("Analyze"):
        requests_text = "\n".join(df[col].dropna().astype(str).tolist())
        prompt = build_prompt(requests_text)

        with st.spinner("Asking AI for clusters and impact..."):
            raw = call_openai(prompt)

        # Safety: try plain JSON, else try to extract from code fences
        parsed = None
        try:
            parsed = json.loads(raw)
        except Exception:
            import re
            m = re.search(r"```(?:json)?\s*(\[\s*{.*?}\s*\])\s*```", raw, flags=re.S)
            if m:
                try:
                    parsed = json.loads(m.group(1))
                except Exception:
                    pass

        if parsed is None:
            st.error("Couldn't parse AI response as JSON. Showing raw text below:")
            st.code(raw)
        else:
            table = normalize_to_df(parsed)
            st.success("Analysis complete!")
            st.dataframe(table, use_container_width=True)

            # Simple bar chart of frequency by theme for demo
            freq_map = {"Low": 1, "Medium": 2, "High": 3}
            viz = table.copy()
            viz["FrequencyScore"] = viz["Frequency"].map(freq_map).fillna(0)
            st.bar_chart(viz.set_index("Theme")["FrequencyScore"])

            # Download
            st.download_button(
                "Download results (CSV)",
                data=table.to_csv(index=False).encode("utf-8"),
                file_name="feature_request_analysis.csv",
                mime="text/csv"
            )
else:
    st.info("Tip: create a simple CSV with one column named `request` and a few rows of sample feedback to try it out.")
