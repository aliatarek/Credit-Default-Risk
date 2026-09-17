# preprocess.py
# ------------------------------------------------------------------
# Standalone preprocessing pipeline for inference
# Takes raw applicant data, returns model-ready feature vector
# ------------------------------------------------------------------

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# ------------------------------------------------------------------
# Load all artifacts at module level (loaded once, reused per request)
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent # adjust .parent count based on actual file depth
DATASET_PATH = BASE_DIR / "artifacts"
MODEL_PATH = BASE_DIR / "artifacts" / "lgbm_model.pkl"

with open(DATASET_PATH / "imputation_values.json")  as f: IMPUTATION   = json.load(f)
with open(DATASET_PATH / "capping_values.json")     as f: CAPPING      = json.load(f)
with open(DATASET_PATH / "org_type_encoding.json")  as f: ORG_ENCODING = json.load(f)
with open(DATASET_PATH / "education_encoding.json") as f: EDU_ENCODING = json.load(f)
with open(DATASET_PATH / "ohe_reference.json")      as f: OHE_REF      = json.load(f)
with open(DATASET_PATH / "bureau_fill_values.json") as f: BUREAU_FILL  = json.load(f)
with open(DATASET_PATH / "final_features.json")     as f: FINAL_FEATS  = json.load(f)

SCALER  = joblib.load(DATASET_PATH / "robust_scaler.pkl")
MODEL   = joblib.load(MODEL_PATH)
EXPLAINER = joblib.load(DATASET_PATH / "shap_explainer.pkl")

with open(DATASET_PATH / "risk_thresholds.json") as f:
    RISK_THRESHOLDS = json.load(f)

print(" All artifacts loaded.")

# ------------------------------------------------------------------
# Risk category assignment
# ------------------------------------------------------------------
def assign_risk_category(probability: float) -> str:
    if probability < RISK_THRESHOLDS["low_risk_max"]:
        return "Low Risk"
    elif probability < RISK_THRESHOLDS["medium_risk_max"]:
        return "Medium Risk"
    else:
        return "High Risk"

# ------------------------------------------------------------------
# Main preprocessing function
# ------------------------------------------------------------------
def preprocess(raw: dict) -> pd.DataFrame:
    """
    Takes a raw applicant dict and returns a scaled,
    model-ready single-row DataFrame with exactly 64 features.
    """
    df = pd.DataFrame([raw])

    # ==============================================================
    # STEP 1 — Drop XNA gender rows (replace with mode if XNA)
    # ==============================================================
    if df["CODE_GENDER"].iloc[0] == "XNA":
        df["CODE_GENDER"] = "F"   # default to majority class

    # ==============================================================
    # STEP 2 — DAYS_EMPLOYED anomaly (365243 sentinel)
    # ==============================================================
    df["EMPLOYED_SENTINEL"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)

    # ==============================================================
    # STEP 3 — EXT_SOURCE_1 missing flag + imputation
    # ==============================================================
    df["EXT_SOURCE_1_MISSING"] = df["EXT_SOURCE_1"].isna().astype(int)
    df["EXT_SOURCE_1_IMPUTED"] = df["EXT_SOURCE_1"].fillna(IMPUTATION["EXT_SOURCE_1"])
    df.drop(columns=["EXT_SOURCE_1"], inplace=True)

    # ==============================================================
    # STEP 4 — CAR_AGE_IMPUTED
    # FLAG_OWN_CAR == N → sentinel -1
    # FLAG_OWN_CAR == Y and OWN_CAR_AGE missing → median of car owners
    # FLAG_OWN_CAR == Y and OWN_CAR_AGE present → use as is
    # ==============================================================
    df["FLAG_OWN_CAR_NUM"] = (df["FLAG_OWN_CAR"] == "Y").astype(int)

    if df["FLAG_OWN_CAR_NUM"].iloc[0] == 0:
        df["CAR_AGE_IMPUTED"] = -1
    else:
        car_age = df["OWN_CAR_AGE"].iloc[0]
        df["CAR_AGE_IMPUTED"] = car_age if not pd.isna(car_age) \
                                else IMPUTATION["CAR_AGE_MEDIAN_FOR_CAR_OWNERS"]

    df.drop(columns=["FLAG_OWN_CAR", "OWN_CAR_AGE"], inplace=True, errors="ignore")

    # ==============================================================
    # STEP 5 — BUREAU_REQ_MISSING flag + impute bureau enquiry cols
    # ==============================================================
    df["BUREAU_REQ_MISSING"] = df["AMT_REQ_CREDIT_BUREAU_YEAR"].isna().astype(int) \
                               if "AMT_REQ_CREDIT_BUREAU_YEAR" in df.columns else 0

    for col, val in {
        "AMT_REQ_CREDIT_BUREAU_MON" : IMPUTATION["AMT_REQ_CREDIT_BUREAU_MON"],
    }.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    # ==============================================================
    # STEP 6 — Impute remaining numeric columns with training medians
    # ==============================================================
    numeric_imputations = {
        "DAYS_EMPLOYED"          : IMPUTATION["DAYS_EMPLOYED"],
        "AMT_ANNUITY"            : IMPUTATION["AMT_ANNUITY"],
        "AMT_GOODS_PRICE"        : IMPUTATION["AMT_GOODS_PRICE"],
        "EXT_SOURCE_2"           : IMPUTATION["EXT_SOURCE_2"],
        "EXT_SOURCE_3"           : IMPUTATION["EXT_SOURCE_3"],
        "DAYS_LAST_PHONE_CHANGE" : IMPUTATION["DAYS_LAST_PHONE_CHANGE"],
    }
    for col, val in numeric_imputations.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    # ==============================================================
    # STEP 7 — OCCUPATION_TYPE imputation
    # ==============================================================
    if "OCCUPATION_TYPE" not in df.columns or pd.isna(df["OCCUPATION_TYPE"].iloc[0]):
        # Same conditional logic as training
        if df["EMPLOYED_SENTINEL"].iloc[0] == 1:
            df["OCCUPATION_TYPE"] = "Not_Employed"
        else:
            df["OCCUPATION_TYPE"] = "Unknown"

    df.drop(columns=["EMPLOYED_SENTINEL"], inplace=True, errors="ignore")

    # ==============================================================
    # STEP 8 — NAME_TYPE_SUITE imputation
    # ==============================================================
    if "NAME_TYPE_SUITE" in df.columns:
        df["NAME_TYPE_SUITE"] = df["NAME_TYPE_SUITE"].fillna(
            IMPUTATION["NAME_TYPE_SUITE_MODE"]
        )

    # ==============================================================
    # STEP 9 — Outlier capping
    # ==============================================================
    for col, upper in CAPPING.items():
        if col == "CAR_AGE_IMPUTED":
            # Cap only real values, not the -1 sentinel
            if df["CAR_AGE_IMPUTED"].iloc[0] != -1:
                df["CAR_AGE_IMPUTED"] = df["CAR_AGE_IMPUTED"].clip(upper=upper)
        elif col == "CNT_CHILDREN":
            df["CNT_CHILDREN"] = df["CNT_CHILDREN"].clip(upper=upper)
        elif col in df.columns:
            df[col] = df[col].clip(upper=upper)

    # ==============================================================
    # STEP 10 — Binary encoding
    # ==============================================================
    df["CODE_GENDER"]        = (df["CODE_GENDER"] == "M").astype(int)
    df["FLAG_OWN_REALTY"]    = (df["FLAG_OWN_REALTY"] == "Y").astype(int) \
                               if "FLAG_OWN_REALTY" in df.columns else 0
    df["NAME_CONTRACT_TYPE"] = (df["NAME_CONTRACT_TYPE"] == "Cash loans").astype(int)

    # ==============================================================
    # STEP 11 — Label encode NAME_EDUCATION_TYPE
    # ==============================================================
    df["NAME_EDUCATION_TYPE"] = df["NAME_EDUCATION_TYPE"].map(EDU_ENCODING).fillna(0)

    # ==============================================================
    # STEP 12 — One-hot encode categorical columns
    # ==============================================================
    for base_col, expected_dummies in OHE_REF.items():
        if base_col not in df.columns:
            # Column missing entirely — add all dummies as 0
            for dummy in expected_dummies:
                df[dummy] = 0
            continue

        # Get dummies for this column
        dummies = pd.get_dummies(df[[base_col]], columns=[base_col],
                                 drop_first=False, dtype=int)

        # Add any missing dummy columns (unseen categories)
        for dummy in expected_dummies:
            if dummy not in dummies.columns:
                dummies[dummy] = 0

        # Keep only the expected columns (drop base + any unexpected)
        df = df.drop(columns=[base_col], errors="ignore")
        df = pd.concat([df, dummies[expected_dummies]], axis=1)

    # ==============================================================
    # STEP 13 — Target encode ORGANIZATION_TYPE
    # ==============================================================
    if "ORGANIZATION_TYPE" in df.columns:
        org_val = df["ORGANIZATION_TYPE"].iloc[0]
        # Use encoding map; fallback to global mean if unseen category
        global_mean = float(np.mean(list(ORG_ENCODING.values())))
        df["ORGANIZATION_TYPE"] = ORG_ENCODING.get(org_val, global_mean)

    # ==============================================================
    # STEP 14 — Flip DAYS_* columns to positive
    # ==============================================================
    days_cols = [
        "DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION",
        "DAYS_ID_PUBLISH", "DAYS_LAST_PHONE_CHANGE"
    ]
    for col in days_cols:
        if col in df.columns:
            df[col] = df[col].abs()

    # ==============================================================
    # STEP 15 — Bureau features
    # (fill with 0 if no bureau history provided)
    # ==============================================================
    for col, val in BUREAU_FILL.items():
        if col not in df.columns:
            df[col] = val

    # ==============================================================
    # STEP 16 — Feature engineering
    # ==============================================================
    df["ANNUITY_INCOME_RATIO"]  = df["AMT_ANNUITY"]   / df["AMT_INCOME_TOTAL"]
    df["CREDIT_TERM"]           = df["AMT_CREDIT"]     / df["AMT_ANNUITY"]
    df["EMPLOYED_TO_AGE_RATIO"] = df["DAYS_EMPLOYED"]  / df["DAYS_BIRTH"]
    df["CREDIT_TO_GOODS_RATIO"] = df["AMT_CREDIT"]     / df["AMT_GOODS_PRICE"]
    df["ADDRESS_MISMATCH"]      = (
        df.get("REG_CITY_NOT_WORK_CITY", 0) +
        df.get("REG_CITY_NOT_LIVE_CITY", 0) +
        df.get("LIVE_CITY_NOT_WORK_CITY", 0)
    )
    df["SOCIAL_DEFAULT_RATE_30"] = (
        df.get("DEF_30_CNT_SOCIAL_CIRCLE", 0) /
        (df.get("OBS_30_CNT_SOCIAL_CIRCLE", 0) + 1e-5)
    )
    df["SOCIAL_DEFAULT_RATE_60"] = (
        df.get("DEF_60_CNT_SOCIAL_CIRCLE", 0) /
        (df.get("OBS_60_CNT_SOCIAL_CIRCLE", 0) + 1e-5)
    )
    df["BUREAU_ACTIVE_RATIO"] = (
        df.get("BUREAU_ACTIVE_COUNT", 0) /
        (df.get("BUREAU_LOAN_COUNT", 0) + 1e-5)
    )

    # ==============================================================
    # STEP 17 — Align to final feature set (exact 64 columns, right order)
    # ==============================================================
    for feat in FINAL_FEATS:
        if feat not in df.columns:
            df[feat] = 0   # missing feature → safe default of 0

    df = df[FINAL_FEATS]   # select and reorder

    # ==============================================================
    # STEP 18 — Scale
    # ==============================================================
    df_scaled = pd.DataFrame(
        SCALER.transform(df),
        columns=FINAL_FEATS
    )

    return df_scaled


# ------------------------------------------------------------------
# Full inference function — preprocess + predict + explain
# ------------------------------------------------------------------
def predict(raw: dict) -> dict:
    # Preprocess
    X = preprocess(raw)

    # Store unscaled values for human-readable explanation
    X_unscaled = pd.DataFrame(
        SCALER.inverse_transform(X),
        columns=FINAL_FEATS
    )

    # Predict probability
    probability = float(MODEL.predict_proba(X)[:, 1][0])

    # Risk category
    risk_category = assign_risk_category(probability)
    action        = RISK_THRESHOLDS["actions"][risk_category]

    # SHAP explanation
    shap_vals = EXPLAINER.shap_values(X)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    shap_series = pd.Series(shap_vals[0], index=FINAL_FEATS)
    top_shap    = shap_series.abs().sort_values(ascending=False).head(10)

    shap_explanation = [
        {
            "feature"   : feat,
            "raw_value" : round(float(X_unscaled[feat].iloc[0]), 4),  # unscaled
            "shap"      : round(float(shap_series[feat]), 4),
            "direction" : "increases default risk" if shap_series[feat] > 0
                          else "decreases default risk"
        }
        for feat in top_shap.index
    ]

    return {
        "probability"     : round(probability, 4),
        "risk_category"   : risk_category,
        "action"          : action,
        "shap_explanation": shap_explanation
    }

# ------------------------------------------------------------------
# Quick test — run this file directly to verify pipeline works
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Minimal raw applicant example
    test_applicant = {
        "NAME_CONTRACT_TYPE"         : "Cash loans",
        "CODE_GENDER"                : "M",
        "FLAG_OWN_CAR"               : "N",
        "FLAG_OWN_REALTY"            : "Y",
        "CNT_CHILDREN"               : 0,
        "AMT_INCOME_TOTAL"           : 180000,
        "AMT_CREDIT"                 : 450000,
        "AMT_ANNUITY"                : 22500,
        "AMT_GOODS_PRICE"            : 450000,
        "NAME_TYPE_SUITE"            : "Unaccompanied",
        "NAME_INCOME_TYPE"           : "Working",
        "NAME_EDUCATION_TYPE"        : "Higher education",
        "NAME_FAMILY_STATUS"         : "Married",
        "NAME_HOUSING_TYPE"          : "House / apartment",
        "REGION_POPULATION_RELATIVE" : 0.035,
        "DAYS_BIRTH"                 : -12000,
        "DAYS_EMPLOYED"              : -3000,
        "DAYS_REGISTRATION"          : -4000,
        "DAYS_ID_PUBLISH"            : -2000,
        "OWN_CAR_AGE"                : None,
        "FLAG_EMP_PHONE"             : 1,
        "FLAG_WORK_PHONE"            : 0,
        "FLAG_PHONE"                 : 1,
        "FLAG_EMAIL"                 : 0,
        "OCCUPATION_TYPE"            : "Managers",
        "CNT_FAM_MEMBERS"            : 2,
        "REGION_RATING_CLIENT"       : 2,
        "REGION_RATING_CLIENT_W_CITY": 2,
        "WEEKDAY_APPR_PROCESS_START" : "TUESDAY",
        "HOUR_APPR_PROCESS_START"    : 10,
        "REG_REGION_NOT_LIVE_REGION" : 0,
        "REG_REGION_NOT_WORK_REGION" : 0,
        "LIVE_REGION_NOT_WORK_REGION": 0,
        "REG_CITY_NOT_LIVE_CITY"     : 0,
        "REG_CITY_NOT_WORK_CITY"     : 0,
        "LIVE_CITY_NOT_WORK_CITY"    : 0,
        "EXT_SOURCE_1"               : 0.65,
        "EXT_SOURCE_2"               : 0.72,
        "EXT_SOURCE_3"               : 0.68,
        "OBS_30_CNT_SOCIAL_CIRCLE"   : 2,
        "DEF_30_CNT_SOCIAL_CIRCLE"   : 0,
        "OBS_60_CNT_SOCIAL_CIRCLE"   : 2,
        "DEF_60_CNT_SOCIAL_CIRCLE"   : 0,
        "DAYS_LAST_PHONE_CHANGE"     : -500,
        "FLAG_DOCUMENT_3"            : 1,
        "FLAG_DOCUMENT_6"            : 0,
        "FLAG_DOCUMENT_13"           : 0,
        "FLAG_DOCUMENT_14"           : 0,
        "FLAG_DOCUMENT_16"           : 0,
        "AMT_REQ_CREDIT_BUREAU_HOUR" : 0,
        "AMT_REQ_CREDIT_BUREAU_DAY"  : 0,
        "AMT_REQ_CREDIT_BUREAU_WEEK" : 0,
        "AMT_REQ_CREDIT_BUREAU_MON"  : 0,
        "AMT_REQ_CREDIT_BUREAU_QRT"  : 0,
        "AMT_REQ_CREDIT_BUREAU_YEAR" : 1,
        # Bureau features (0 if first-time borrower)
        "BUREAU_LOAN_COUNT"          : 3,
        "BUREAU_ACTIVE_COUNT"        : 1,
        "BUREAU_DAYS_CREDIT_MEAN"    : -800,
        "BUREAU_DAYS_ENDDATE_MEAN"   : 200,
        "BUREAU_AMT_CREDIT_MEAN"     : 150000,
        "ORGANIZATION_TYPE"          : "Business Entity Type 3",
    }

    result = predict(test_applicant)

    print("\n=== Inference Result ===")
    print(f"Probability    : {result['probability']}")
    print(f"Risk Category  : {result['risk_category']}")
    print(f"Action         : {result['action']}")
    print(f"\nTop SHAP features:")
    for item in result["shap_explanation"]:
        direction = "↑" if item["shap"] > 0 else "↓"
        print(f"  {direction} {item['feature']:<35} "
              f"shap={item['shap']:+.4f}  raw_value={item['raw_value']:.3f}")
    



