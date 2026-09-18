"""
tests/test_api.py

A minimal correctness check for the deployed FastAPI /predict endpoint.
Run against a container that's already up (see the CI workflow, or run
locally with `docker run -p 8000:8000 credit-risk-api` first).

This is NOT a unit test in the strict sense — it's an integration smoke
test: "given a real applicant, does the live API return a sane response."
"""

import sys
import time

import pandas as pd
import requests

API_URL = "http://localhost:8000/predict"
HEALTH_URL = "http://localhost:8000/health"
DATA_PATH = "Dataset/x_test_raw_with_id.csv"

REQUIRED_RESPONSE_FIELDS = {"probability", "risk_category", "action", "shap_explanation"}
VALID_RISK_CATEGORIES = {"Low Risk", "Medium Risk", "High Risk"}


def wait_for_api(timeout_seconds=30):
    """Poll /health until the API is up, or fail after timeout."""
    start = time.time()
    while time.time() - start < timeout_seconds:
        try:
            r = requests.get(HEALTH_URL, timeout=2)
            if r.status_code == 200:
                print("API is up.")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
    return False


def get_sample_payload():
    """Pull one real applicant row from the test set, matching what the
    Streamlit app sends in production — same cleaning logic (NaN -> None)."""
    df = pd.read_csv(DATA_PATH)
    row = df.iloc[0].drop("SK_ID_CURR").to_dict()
    return {k: (None if pd.isna(v) else v) for k, v in row.items()}


def test_predict_returns_valid_response():
    payload = get_sample_payload()
    response = requests.post(API_URL, json=payload, timeout=15)

    assert response.status_code == 200, (
        f"Expected 200, got {response.status_code}. Body: {response.text}"
    )

    body = response.json()

    missing_fields = REQUIRED_RESPONSE_FIELDS - body.keys()
    assert not missing_fields, f"Response missing expected fields: {missing_fields}"

    assert 0.0 <= body["probability"] <= 1.0, (
        f"probability out of expected [0,1] range: {body['probability']}"
    )
    assert body["risk_category"] in VALID_RISK_CATEGORIES, (
        f"Unexpected risk_category: {body['risk_category']}"
    )
    assert isinstance(body["shap_explanation"], list) and len(body["shap_explanation"]) > 0, (
        "shap_explanation should be a non-empty list"
    )

    print("test_predict_returns_valid_response passed.")
    print(f"  probability={body['probability']:.4f}, risk_category={body['risk_category']}")


def main():
    if not wait_for_api():
        print("API never became healthy within timeout.")
        sys.exit(1)

    try:
        test_predict_returns_valid_response()
    except AssertionError as e:
        print(f"TEST FAILED: {e}")
        sys.exit(1)

    print("\nAll tests passed.")


if __name__ == "__main__":
    main()