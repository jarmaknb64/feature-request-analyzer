
import streamlit as st
import pandas as pd
import json
from openai import OpenAI

st.set_page_config(page_title="Feature Request Analyzer", layout="wide")
st.title("🤖 Feature Request Analyzer")

st.caption("Upload a CSV with one column of requests (e.g., 'Requests').")

# Create client using Streamlit secret
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

def build_prompt(requests_text: str) -> str:
    return f"""
Read the following customer feature requests. Cluster them into coherent THEMES.
For each theme, return:
- "theme": short theme name
- "requests": list of representative original lines (3–6 max)
- "frequency": Low | Medium | High (based on share of lines)
- "sentiment": Positive | Neutral | Negative
- "business_impact": Low | Medium | High (estimate for a Shopify A/B testing tool)

Return ONLY a JSON array. No commentary. Example shape:
[{{"theme":"Exporting Data","requests":["..."],"frequency":"High","sentiment":"Neutral","business_impact":"High"}}]

Requests:
{requests_text}
""".strip()

def analyze(text: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[
            {"role": "system", "content": "You output concise, valid JSON only."},
            {"role": "user", "content": text}
        ],
    )
    return resp.choices[0].message.content

def to_table(items):
    rows = []
    for it in items:
        rows.append({
            "Theme": it.get("theme", ""),
            "Frequency": it.get("frequency", ""),
            "Impact": it.get("business_impact", ""),
            "Sentiment": it.get("sentiment", ""),
            "Examples": " • ".join((it.get("requests") or [])[:5]),
        })
    return pd.DataFrame(rows)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    col = df.columns[0]
    st.success(f"Detected request column: {col}")
    st.dataframe(df.head())

    if st.button("Analyze"):
        requests_text = "\n".join(df[col].dropna().astype(str).tolist())
        prompt = build_prompt(requests_text)

        with st.spinner("Asking AI..."):
            raw = analyze(prompt)

        # Try to parse JSON; fallback to code-fence extraction
        try:
            data = json.loads(raw)
        except Exception:
            import re
            m = re.search(r"```(?:json)?\s*(\[\s*{.*?}\s*\])\s*```", raw, flags=re.S)
            data = json.loads(m.group(1)) if m else None

        if not data:
            st.error("Could not parse AI response as JSON. Showing raw text:")
            st.code(raw)
        else:
            table = to_table(data)
            st.dataframe(table, use_container_width=True)
            # simple frequency bar
            score = {"Low": 1, "Medium": 2, "High": 3}
            tmp = table.copy()
            tmp["FreqScore"] = tmp["Frequency"].map(score).fillna(0)
            st.bar_chart(tmp.set_index("Theme")["FreqScore"])
            st.download_button(
                "Download results CSV",
                data=table.to_csv(index=False).encode("utf-8"),
                file_name="feature_request_analysis.csv",
                mime="text/csv",
            )
else:
    st.info("Upload a CSV to begin.")
