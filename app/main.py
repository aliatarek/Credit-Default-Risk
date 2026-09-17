import sys
import os
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# pyrefly: ignore [missing-import]
from fastapi import FastAPI
from app.routers.predict import router          # ← add this
from app.services.preprocess import predict

app = FastAPI(
    title="Credit Risk Prediction API",
    description="Predicts default probability, risk category, and SHAP explanations for loan applicants.",
    version="1.0.0"
)

app.include_router(router)                      # ← and this