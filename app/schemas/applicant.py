# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field
from typing import Optional


#REQUEST MODEL

class ApplicantData(BaseModel):
    # Core application fields
    NAME_CONTRACT_TYPE          : str   = Field(..., example="Cash loans")
    CODE_GENDER                 : str   = Field(..., example="M")
    FLAG_OWN_CAR                : str   = Field(..., example="N")
    FLAG_OWN_REALTY             : str   = Field(..., example="Y")
    CNT_CHILDREN                : int   = Field(..., example=0)
    AMT_INCOME_TOTAL            : float = Field(..., example=180000)
    AMT_CREDIT                  : float = Field(..., example=450000)
    AMT_ANNUITY                 : float = Field(..., example=22500)
    AMT_GOODS_PRICE             : float = Field(..., example=450000)
    NAME_INCOME_TYPE            : str   = Field(..., example="Working")
    NAME_EDUCATION_TYPE         : str   = Field(..., example="Higher education")
    NAME_FAMILY_STATUS          : str   = Field(..., example="Married")
    NAME_HOUSING_TYPE           : str   = Field(..., example="House / apartment")
    DAYS_BIRTH                  : int   = Field(..., example=-12000)
    DAYS_EMPLOYED               : int   = Field(..., example=-3000)
    DAYS_REGISTRATION           : float = Field(..., example=-4000)
    DAYS_ID_PUBLISH             : int   = Field(..., example=-2000)
    REGION_POPULATION_RELATIVE  : float = Field(..., example=0.035)
    REGION_RATING_CLIENT_W_CITY : int   = Field(..., example=2)
    HOUR_APPR_PROCESS_START     : int   = Field(..., example=10)
    WEEKDAY_APPR_PROCESS_START  : str   = Field(..., example="TUESDAY")
    ORGANIZATION_TYPE           : str   = Field(..., example="Business Entity Type 3")

    # Optional fields — imputed if missing
    OWN_CAR_AGE                 : Optional[float] = Field(None,  example=None)
    NAME_TYPE_SUITE             : Optional[str]   = Field(None,  example="Unaccompanied")
    OCCUPATION_TYPE             : Optional[str]   = Field(None,  example="Managers")
    EXT_SOURCE_1                : Optional[float] = Field(None,  example=0.65)
    EXT_SOURCE_2                : Optional[float] = Field(None,  example=0.72)
    EXT_SOURCE_3                : Optional[float] = Field(None,  example=0.68)
    DAYS_LAST_PHONE_CHANGE      : Optional[float] = Field(None,  example=-500)
    AMT_REQ_CREDIT_BUREAU_MON   : Optional[float] = Field(None,  example=0)
    AMT_REQ_CREDIT_BUREAU_YEAR  : Optional[float] = Field(None,  example=1)

    # Flag fields
    FLAG_EMP_PHONE              : Optional[int]   = Field(0, example=1)
    FLAG_WORK_PHONE             : Optional[int]   = Field(0, example=0)
    FLAG_PHONE                  : Optional[int]   = Field(0, example=1)
    FLAG_EMAIL                  : Optional[int]   = Field(0, example=0)
    FLAG_DOCUMENT_3             : Optional[int]   = Field(0, example=1)
    FLAG_DOCUMENT_6             : Optional[int]   = Field(0, example=0)
    FLAG_DOCUMENT_13            : Optional[int]   = Field(0, example=0)
    FLAG_DOCUMENT_14            : Optional[int]   = Field(0, example=0)
    FLAG_DOCUMENT_16            : Optional[int]   = Field(0, example=0)

    # Region/city mismatch flags
    REG_REGION_NOT_LIVE_REGION  : Optional[int]   = Field(0, example=0)
    REG_REGION_NOT_WORK_REGION  : Optional[int]   = Field(0, example=0)
    LIVE_REGION_NOT_WORK_REGION : Optional[int]   = Field(0, example=0)
    REG_CITY_NOT_LIVE_CITY      : Optional[int]   = Field(0, example=0)
    REG_CITY_NOT_WORK_CITY      : Optional[int]   = Field(0, example=0)
    LIVE_CITY_NOT_WORK_CITY     : Optional[int]   = Field(0, example=0)

    # Social circle
    OBS_30_CNT_SOCIAL_CIRCLE    : Optional[float] = Field(0, example=2)
    DEF_30_CNT_SOCIAL_CIRCLE    : Optional[float] = Field(0, example=0)
    OBS_60_CNT_SOCIAL_CIRCLE    : Optional[float] = Field(0, example=2)
    DEF_60_CNT_SOCIAL_CIRCLE    : Optional[float] = Field(0, example=0)

    # Bureau features (0 if first-time borrower)
    BUREAU_LOAN_COUNT           : Optional[float] = Field(0, example=3)
    BUREAU_ACTIVE_COUNT         : Optional[float] = Field(0, example=1)
    BUREAU_DAYS_CREDIT_MEAN     : Optional[float] = Field(0, example=-800)
    BUREAU_DAYS_ENDDATE_MEAN    : Optional[float] = Field(0, example=200)
    BUREAU_AMT_CREDIT_MEAN      : Optional[float] = Field(0, example=150000)

    # Misc
    CNT_FAM_MEMBERS             : Optional[float] = Field(2, example=2)
    REGION_RATING_CLIENT        : Optional[int]   = Field(2, example=2)
    AMT_REQ_CREDIT_BUREAU_HOUR  : Optional[float] = Field(0, example=0)
    AMT_REQ_CREDIT_BUREAU_DAY   : Optional[float] = Field(0, example=0)
    AMT_REQ_CREDIT_BUREAU_WEEK  : Optional[float] = Field(0, example=0)
    AMT_REQ_CREDIT_BUREAU_QRT   : Optional[float] = Field(0, example=0)


#RESPONSE SCHEMA

class SHAPFeature(BaseModel):
    feature   : str
    raw_value : float
    shap      : float
    direction : str

class PredictionResponse(BaseModel):
    probability      : float
    risk_category    : str
    action           : str
    shap_explanation : list[SHAPFeature]