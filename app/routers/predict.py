# pyrefly: ignore [missing-import]
from fastapi import HTTPException, APIRouter
from app.services.preprocess import predict
from app.schemas.applicant import ApplicantData, PredictionResponse

router = APIRouter()

@router.get("/")
def root():
    return {
        "message"  : "Credit Risk Prediction API",
        "version"  : "1.0.0",
        "endpoints": {
            "POST /predict" : "Predict default risk for a single applicant",
            "GET /health"   : "Health check",
            "GET /docs"     : "Interactive API documentation"
        }
    }

@router.get("/health")
def health():
    return {"status": "healthy"}

@router.post("/predict", response_model=PredictionResponse)
def predict_endpoint(applicant: ApplicantData):
    try:
        raw = applicant.model_dump()
        result = predict(raw)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))