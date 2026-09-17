"""
train_final.py

Production training script for the final chosen model (LightGBM).

Unlike experiments.py (which reproduces the full model comparison for
learning/tracking purposes), this script does ONE thing: train the winning
configuration, log it to MLflow, and REGISTER it in the Model Registry as
a named, versioned model.

Run with:
    python train_final.py
"""

from pathlib import Path

import joblib
# pyrefly: ignore [missing-import]
import mlflow
import pandas as pd
# pyrefly: ignore [missing-import]
from scipy.stats import randint, uniform
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV
# pyrefly: ignore [missing-import]
import lightgbm as lgb
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt

# ============================================================
# Config
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "Dataset"

MLFLOW_EXPERIMENT_NAME = "credit-default-risk"
REGISTERED_MODEL_NAME = "credit-default-risk-lightgbm"
RANDOM_STATE = 912
CLASSIFICATION_THRESHOLD = 0.29


# ============================================================
# Data loading — uses the SAME existing scaler as experiments.py,
# not a freshly fit one (same reasoning as before: reproducing a
# known-good pipeline, not re-deciding the preprocessing).
# ============================================================
def load_and_scale_data():
    X_train = pd.read_csv(DATASET_PATH / "X_train.csv")
    X_test = pd.read_csv(DATASET_PATH / "X_test.csv")
    y_train = pd.read_csv(DATASET_PATH / "y_train.csv").squeeze()
    y_test = pd.read_csv(DATASET_PATH / "y_test.csv").squeeze()

    scaler = joblib.load(DATASET_PATH / "robust_scaler.pkl")
    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    return X_train_scaled, X_test_scaled, y_train, y_test


def log_classification_metrics(y_true, y_pred, y_proba):
    mlflow.log_metric("roc_auc", roc_auc_score(y_true, y_proba))
    mlflow.log_metric("precision", precision_score(y_true, y_pred))
    mlflow.log_metric("recall", recall_score(y_true, y_pred))
    mlflow.log_metric("f1", f1_score(y_true, y_pred))

    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax)
    mlflow.log_figure(fig, "confusion_matrix.png")
    plt.close(fig)


# ============================================================
# Train + register the final model
# ============================================================
def train_and_register():
    X_train, X_test, y_train, y_test = load_and_scale_data()

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name="lightgbm_final_production"):
        mlflow.log_param("model_type", "LightGBM")

        param_dist = {
            "n_estimators": randint(200, 1000),
            "learning_rate": uniform(0.01, 0.09),      # 0.01 to 0.10
            "max_depth": randint(3, 10),
            "num_leaves": randint(15, 63),
            "min_child_samples": randint(20, 200),
            "subsample": uniform(0.6, 0.4),             # 0.6 to 1.0
            "colsample_bytree": uniform(0.6, 0.4),      # 0.6 to 1.0
            "reg_alpha": uniform(0, 0.5),                # L1
            "reg_lambda": uniform(0, 0.5),                # L2
            "scale_pos_weight": [1, 1.25, 1.5, 2, 3, 5],  # imbalance handling options
        }

        base_model = lgb.LGBMClassifier(random_state=RANDOM_STATE)
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_dist,
            n_iter=10,
            random_state=RANDOM_STATE,
        )
        search.fit(X_train, y_train)

        mlflow.log_params(search.best_params_)

        best_model = search.best_estimator_
        proba = best_model.predict_proba(X_test)[:, 1]

        mlflow.log_param("classification_threshold", CLASSIFICATION_THRESHOLD)
        preds = (proba >= CLASSIFICATION_THRESHOLD).astype(int)
        log_classification_metrics(y_test, preds, proba)

        # Log AND register in one step — this is the new part vs experiments.py.
        # registered_model_name creates (or adds a new version to) an entry
        # in the Model Registry, separate from just logging the run artifact.
        mlflow.lightgbm.log_model(
            best_model,
            "model",
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        print(f"Model registered as '{REGISTERED_MODEL_NAME}'.")
        print(f"ROC-AUC: {roc_auc_score(y_test, proba):.4f}")

        return best_model


if __name__ == "__main__":
    train_and_register()