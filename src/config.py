# src/config.py
from pathlib import Path

# ------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "applications_dataset.csv"

# Output folders – the notebook will write CSVs / PNGs here
EDA_OUTPUT_PATH = PROJECT_ROOT / "reports" / "eda"
FIGURE_PATH     = PROJECT_ROOT / "reports" / "figures"


RANDOM_STATE = 94
TARGET_COL   = "TARGET"
ID_COL       = "SK_ID_CURR"
DAYS_EMPLOYED_ANOMALY = 365_243


COLUMN_GROUPS = {
    "target": [TARGET_COL],
    "loan": [
        "NAME_CONTRACT_TYPE", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
    ],
    "demographics": [
        "CODE_GENDER", "DAYS_BIRTH", "CNT_CHILDREN", "CNT_FAM_MEMBERS",
    ],
    "income_employment": [
        "AMT_INCOME_TOTAL", "NAME_INCOME_TYPE", "OCCUPATION_TYPE", "DAYS_EMPLOYED",
    ],
    "education_family": ["NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS"],
    "housing": [
        "NAME_HOUSING_TYPE", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "OWN_CAR_AGE",
    ],
    "region": [
        "REGION_RATING_CLIENT", "REGION_RATING_CLIENT_W_CITY",
        "REGION_POPULATION_RELATIVE", "REG_REGION_NOT_LIVE_REGION",
        "REG_REGION_NOT_WORK_REGION", "LIVE_REGION_NOT_WORK_REGION",
        "REG_CITY_NOT_LIVE_CITY", "REG_CITY_NOT_WORK_CITY", "LIVE_CITY_NOT_WORK_CITY",
    ],
    "external_scores": ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"],
    "social_circle": [
        "OBS_30_CNT_SOCIAL_CIRCLE", "DEF_30_CNT_SOCIAL_CIRCLE",
        "OBS_60_CNT_SOCIAL_CIRCLE", "DEF_60_CNT_SOCIAL_CIRCLE",
    ],
    "documents": [f"FLAG_DOCUMENT_{i}" for i in range(2, 22)],
    "bureau_requests": [
        "AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_DAY",
        "AMT_REQ_CREDIT_BUREAU_WEEK", "AMT_REQ_CREDIT_BUREAU_MON",
        "AMT_REQ_CREDIT_BUREAU_QRT", "AMT_REQ_CREDIT_BUREAU_YEAR",
    ],
    "process_meta": ["WEEKDAY_APPR_PROCESS_START", "HOUR_APPR_PROCESS_START"],
}

BUILDING_PREFIXES = ("APARTMENTS_", "BASEMENTAREA_", "YEARS_BEGINEXPLUATATION_",
                     "YEARS_BUILD_", "COMMONAREA_", "ELEVATORS_", "ENTRANCES_",
                     "FLOORSMAX_", "FLOORSMIN_", "LANDAREA_", "LIVINGAPARTMENTS_",
                     "LIVINGAREA_", "NONLIVINGAPARTMENTS_", "NONLIVINGAREA_")

RISK_TIER_THRESHOLDS = {"low": 0.15, "medium": 0.35}
