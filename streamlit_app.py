import re
import json
import time
import requests
import pandas as pd
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
import streamlit as st
import os
# pyrefly: ignore [missing-import]
from ollama import chat
# pyrefly: ignore [missing-import]
from ollama import Client

# ============================================================
# Config
# ============================================================
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")
DATA_PATH = "Dataset/x_test_raw_with_id.csv"      # raw features + SK_ID_CURR
LABELS_PATH = "Dataset/y_test_with_id.csv"   
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
ollama_client = Client(host=OLLAMA_HOST)      # actual TARGET + SK_ID_CURR (optional)
OLLAMA_MODEL = "mistral:7b-instruct-v0.3-q4_K_M"   # match whatever tag you pulled
BATCH_SAMPLE_SIZE = 100                             # how many rows to scan for batch high-risk queries

RISK_COLORS = {"Low Risk": "🟢", "Medium Risk": "🟡", "High Risk": "🔴"}

SYSTEM_PROMPT = """You are a credit risk assistant for loan analysts.

You will be given structured JSON data about a loan applicant with these fields:
- "probability": the model's predicted default probability, expressed as a decimal
  fraction between 0 and 1 (e.g. 0.0402 means a 4.02% chance of default — always
  convert to a percentage by multiplying by 100 when you state it).
- "risk_category": Low Risk, Medium Risk, or High Risk.
- "action": the recommended lending action for this risk tier.
- "shap_explanation": a list of the top factors driving this specific prediction.
  Each entry has:
    - "feature": the feature name
    - "raw_value": the applicant's actual value for that feature (this is just
      their data point, NOT a measure of impact)
    - "shap": how much this feature pushed the prediction, in either direction.
      A positive shap value increases predicted default risk; a negative shap
      value decreases it. The MAGNITUDE of shap (not raw_value) tells you how
      important that factor was to this prediction — a raw_value of 0 does NOT
      mean the feature had no effect; check the shap value instead.
    - "direction": a plain-language label already stating whether this factor
      increases or decreases risk — trust this field.

Write a 2-4 sentence explanation in plain language for a credit analyst.
State the default probability as a percentage. Reference 2-3 of the most
influential factors (largest absolute shap values), using their "direction"
label to describe whether each increases or decreases risk. Only use numbers
and facts present in the JSON — never invent or estimate figures. Risk tiers
map to actions: Low = auto-approve, Medium = manual review, High = reject or
request additional verification.

If the risk_category is "Medium Risk", explicitly note that this is a
borderline case that warrants human judgment rather than a fully automated
decision."""

st.set_page_config(page_title="Credit Risk Assistant", page_icon="💳", layout="centered")

# ============================================================
# Data loading (cached)
# ============================================================
@st.cache_data
def load_test_data():
    return pd.read_csv(DATA_PATH)

@st.cache_data
def load_labels():
    try:
        return pd.read_csv(LABELS_PATH)
    except FileNotFoundError:
        return None

df = load_test_data()
labels_df = load_labels()

@st.cache_data
def get_sample_ids(n=4):
    return df["SK_ID_CURR"].sample(n, random_state=42).tolist()

# ============================================================
# Intent detection (rule-based)
# ============================================================
def extract_customer_id(text: str) -> int | None:
    match = re.search(r'\b\d{6,}\b', text)
    return int(match.group()) if match else None

def classify_intent(text: str) -> str:
    text_lower = text.lower()
    if any(kw in text_lower for kw in ["all high-risk", "list high risk", "show all", "high risk applicants"]):
        return "batch_high_risk"
    elif extract_customer_id(text) is not None:
        return "single_customer"
    else:
        return "unknown"

# ============================================================
# Lookup + API call
# ============================================================
def get_applicant_row(customer_id: int) -> dict | None:
    row = df[df["SK_ID_CURR"] == customer_id]
    if row.empty:
        return None
    row_dict = row.drop(columns=["SK_ID_CURR"]).iloc[0].to_dict()
    cleaned = {k: (None if pd.isna(v) else v) for k, v in row_dict.items()}
    return cleaned

def get_actual_outcome(customer_id: int) -> str | None:
    """Returns 'Defaulted', 'Repaid', or None if labels aren't available / ID not found."""
    if labels_df is None:
        return None
    row = labels_df[labels_df["SK_ID_CURR"] == customer_id]
    if row.empty:
        return None
    target = row.iloc[0]["TARGET"]
    return "Defaulted" if target == 1 else "Repaid"

def call_predict(payload: dict) -> dict:
    response = requests.post(API_URL, json=payload)
    response.raise_for_status()
    return response.json()

def stream_narration(prediction: dict):
    """Generator yielding text chunks from Ollama, for use with st.write_stream."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Explain this prediction:\n{json.dumps(prediction, indent=2)}"}
    ]
    for chunk in ollama_client.chat(model=OLLAMA_MODEL, messages=messages, stream=True):
        piece = chunk.message.content
        if piece:
            yield piece

# ============================================================
# Rendering helpers
# ============================================================
def build_shap_chart(shap_explanation: list[dict]):
    sorted_feats = sorted(shap_explanation, key=lambda f: abs(f["shap"]), reverse=True)
    features = [f["feature"] for f in sorted_feats][::-1]
    values = [f["shap"] for f in sorted_feats][::-1]
    colors = ["#d9534f" if v > 0 else "#5cb85c" for v in values]

    fig, ax = plt.subplots(figsize=(6, max(2, 0.4 * len(features))))
    ax.barh(features, values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value (impact on default risk)")
    ax.set_title("Top contributing factors")
    fig.tight_layout()
    return fig

def render_prediction_details(prediction: dict, customer_id: int | None = None, elapsed: float | None = None):
    prob_pct = prediction["probability"] * 100
    badge = RISK_COLORS.get(prediction["risk_category"], "⚪")

    col1, col2, col3 = st.columns(3)
    col1.metric("Default Probability", f"{prob_pct:.1f}%")
    col2.metric("Risk Category", f"{badge} {prediction['risk_category']}")
    col3.metric("Recommended Action", prediction["action"])

    # predicted vs actual outcome, if we have ground truth for this ID
    if customer_id is not None:
        actual = get_actual_outcome(customer_id)
        if actual is not None:
            match = (actual == "Defaulted") == (prediction["risk_category"] == "High Risk")
            st.caption(f"📋 Actual outcome on record: **{actual}**")

    fig = build_shap_chart(prediction["shap_explanation"])
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("View raw JSON"):
        st.json(prediction)

    if elapsed is not None:
        st.caption(f"⏱️ Explained in {elapsed:.1f}s")

def render_batch_results(results: list[dict]):
    if not results:
        st.write("No high-risk applicants found in the sampled batch.")
        return
    table = pd.DataFrame([
        {
            "Customer ID": r["customer_id"],
            "Default Probability": f"{r['probability']*100:.1f}%",
            "Risk Category": f"{RISK_COLORS.get(r['risk_category'], '⚪')} {r['risk_category']}",
            "Action": r["action"],
        }
        for r in results
    ])
    st.dataframe(table, hide_index=True, use_container_width=True)

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.header("💳 Credit Risk Assistant")
    st.caption("AI-assisted loan default prediction for credit analysts")

    st.subheader("Try a sample applicant")
    sample_ids = get_sample_ids()
    for sid in sample_ids:
        if st.button(f"Customer {sid}", key=f"sample_{sid}", use_container_width=True):
            st.session_state.pending_query = f"Will customer {sid} default?"
            st.rerun()

    st.subheader("Other example queries")
    st.markdown(
        "- \"Should we approve loan 173992?\"\n"
        "- \"Show all high-risk applicants\""
    )

    st.subheader("About")
    st.caption(
        "Predictions come from a LightGBM model trained on historical Home Credit "
        "applications. Explanations are generated by a local LLM (Mistral 7B) "
        "based only on the model's own output — it does not have independent "
        "knowledge of the applicant."
    )
    st.caption(
        "⚠️ This demo does not incorporate live credit bureau history. "
        "Bureau-related features fall back to the model's first-time-borrower "
        "defaults for every applicant."
    )
    st.caption("⏱️ Explanations stream in real time but can still take up to ~40s to finish on local hardware.")

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

# ============================================================
# Main chat UI
# ============================================================
st.title("Credit Risk Assistant")
st.caption("Ask about a loan applicant, e.g. \"Will customer 307396 default?\"")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("raw_json"):
            render_prediction_details(
                msg["raw_json"],
                customer_id=msg.get("customer_id"),
                elapsed=msg.get("elapsed"),
            )
        if msg.get("batch_results") is not None:
            render_batch_results(msg["batch_results"])

# support the sidebar sample-ID buttons pre-filling a query
pending = st.session_state.pop("pending_query", None)
user_input = st.chat_input("Ask about an applicant...")
if pending and not user_input:
    user_input = pending

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    intent = classify_intent(user_input)
    raw_json = None
    elapsed = None
    customer_id = None
    batch_results = None

    with st.chat_message("assistant"):
        if intent == "single_customer":
            customer_id = extract_customer_id(user_input)
            applicant_features = get_applicant_row(customer_id)

            if applicant_features is None:
                response_text = f"Customer ID {customer_id} not found in the test dataset."
                st.write(response_text)
            else:
                try:
                    with st.spinner("Getting prediction..."):
                        prediction = call_predict(applicant_features)
                    raw_json = prediction

                    start = time.time()
                    response_text = st.write_stream(stream_narration(prediction))
                    elapsed = time.time() - start

                    render_prediction_details(raw_json, customer_id=customer_id, elapsed=elapsed)

                except requests.exceptions.HTTPError as e:
                    try:
                        detail = e.response.json()
                        response_text = f"API rejected the request:\n```json\n{json.dumps(detail, indent=2)}\n```"
                    except Exception:
                        response_text = f"Error calling prediction API: {e}"
                    st.write(response_text)
                except requests.exceptions.RequestException as e:
                    response_text = f"Error calling prediction API: {e}"
                    st.write(response_text)
                except Exception as e:
                    response_text = (
                        f"Got a prediction but couldn't generate an explanation ({e}). "
                        f"Raw result is below."
                    )
                    st.write(response_text)
                    if raw_json:
                        render_prediction_details(raw_json, customer_id=customer_id)

        elif intent == "batch_high_risk":
            response_text = f"Scanning a sample of {BATCH_SAMPLE_SIZE} test applicants for high risk..."
            st.write(response_text)

            sample_df = df.sample(min(BATCH_SAMPLE_SIZE, len(df)), random_state=7)
            results = []
            progress = st.progress(0.0)
            for i, (_, row) in enumerate(sample_df.iterrows()):
                cid = row["SK_ID_CURR"]
                payload = {k: (None if pd.isna(v) else v) for k, v in row.drop("SK_ID_CURR").to_dict().items()}
                try:
                    pred = call_predict(payload)
                    if pred["risk_category"] == "High Risk":
                        results.append({**pred, "customer_id": int(cid)})
                except requests.exceptions.RequestException:
                    pass  # skip rows the API rejects rather than aborting the whole batch
                progress.progress((i + 1) / len(sample_df))
            progress.empty()

            batch_results = results
            response_text = (
                f"Found {len(results)} high-risk applicant(s) out of {len(sample_df)} sampled."
            )
            st.write(response_text)
            render_batch_results(batch_results)

        else:
            response_text = "I couldn't find a customer ID in your message. Try something like \"will customer 307396 default?\""
            st.write(response_text)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text,
        "raw_json": raw_json,
        "customer_id": customer_id,
        "elapsed": elapsed,
        "batch_results": batch_results,
    })