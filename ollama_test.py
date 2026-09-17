import time
# pyrefly: ignore [missing-import]
from ollama import chat

MODEL = "mistral:7b-instruct-v0.3-q4_K_M"  # match whatever tag you pulled

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
      Describe SHAP magnitude as 'a strong/moderate/minor factor' rather than stating it as a percentage.
    - "direction": a plain-language label already stating whether this factor
      increases or decreases risk — trust this field.

Write a 2-4 sentence explanation in plain language for a credit analyst.
State the default probability as a percentage. Reference 2-3 of the most
influential factors (largest absolute shap values), using their "direction"
label to describe whether each increases or decreases risk. Only use numbers
and facts present in the JSON — never invent or estimate figures. Risk tiers
map to actions: Low = auto-approve, Medium = manual review, High = reject or
request additional verification."""

# a fake version of the real JSON structure from your /predict endpoint,
# just to test with something realistic
fake_prediction = """
{
  "probability": 0.0402,
  "risk_category": "Low Risk",
  "action": "Auto Approve",
  "shap_explanation": [
    {"feature": "EXT_SOURCE_2", "raw_value": 0.6873, "shap": -0.2748, "direction": "decreases default risk"},
    {"feature": "BUREAU_AMT_CREDIT_MEAN", "raw_value": 0.0, "shap": 0.1985, "direction": "increases default risk"},
    {"feature": "CREDIT_TERM", "raw_value": 25.1485, "shap": -0.1985, "direction": "decreases default risk"}
  ]
}
"""

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": f"Explain this prediction:\n{fake_prediction}"}
]

print("Sending request to Ollama...")
start = time.time()

response = chat(model=MODEL, messages=messages)

elapsed = time.time() - start

print("\n--- Response ---")
print(response.message.content)
print(f"\n--- Took {elapsed:.1f} seconds ---")