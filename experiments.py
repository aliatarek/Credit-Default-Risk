"""
experiments.py

Reproduces the model comparison originally done in models.ipynb, with every
run logged to MLflow instead of manually read off notebook cell outputs.

This is a LEARNING/BACKTRACKING exercise: the winning model (LightGBM) is
already known from the original notebook work. The goal here is to practice
the MLflow experiment-tracking workflow on real data, not to re-decide
which model to use.

Run with:
    mlflow ui          (in one terminal, to view results afterward)
    python experiments.py
"""


from sklearn.model_selection import StratifiedKFold
# pyrefly: ignore [missing-import]
from scipy.stats import randint
# pyrefly: ignore [missing-import]
from scipy.stats import uniform
import json
from pathlib import Path

import joblib
# pyrefly: ignore [missing-import]
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeClassifier
# pyrefly: ignore [missing-import]
from imblearn.over_sampling import SMOTE
# pyrefly: ignore [missing-import]
import lightgbm as lgb
# pyrefly: ignore [missing-import]
import xgboost as xgb
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt

# ============================================================
# Config / paths
# ============================================================
# Relative to this script's location, NOT a hardcoded absolute path
# (see the earlier D:/ path discussion — same fix applies here)
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "Dataset"
ARTIFACT_PATH = BASE_DIR / "artifacts"
MLFLOW_EXPERIMENT_NAME = "credit-default-risk"


# ============================================================
# Data loading + preprocessing (shared by every model below)
# ============================================================
def load_and_scale_data():
    X_train = pd.read_csv(DATASET_PATH / "X_train.csv")
    X_test = pd.read_csv(DATASET_PATH / "X_test.csv")
    y_train = pd.read_csv(DATASET_PATH / "y_train.csv").squeeze()
    y_test = pd.read_csv(DATASET_PATH / "y_test.csv").squeeze()

    # Load the SAME scaler already used for the deployed model / original
    # notebook experiments — this script reproduces past results, so it
    # should use the exact artifact those results depended on, not a
    # freshly re-fit one that could subtly differ.
    scaler = joblib.load(ARTIFACT_PATH / "robust_scaler.pkl")
    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def log_classification_metrics(y_true, y_pred, y_proba):
    """Logs the standard metric set + confusion matrix artifact for one run."""
    mlflow.log_metric("roc_auc", roc_auc_score(y_true, y_proba))
    mlflow.log_metric("precision", precision_score(y_true, y_pred))
    mlflow.log_metric("recall", recall_score(y_true, y_pred))
    mlflow.log_metric("f1", f1_score(y_true, y_pred))

    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax)
    mlflow.log_figure(fig, "confusion_matrix.png")
    plt.close(fig)


# ============================================================
# Logistic Regression: baseline, class-weighted, SMOTE
# ============================================================
def run_logistic_regression_experiments(X_train, X_test, y_train, y_test):
    # --- Baseline ---
    with mlflow.start_run(run_name="logreg_baseline"):
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("variant", "baseline")

        model = LogisticRegression(max_iter=1000, random_state=912)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        log_classification_metrics(y_test, preds, proba)

    # --- Class-weighted ---
    with mlflow.start_run(run_name="logreg_class_weighted"):
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("variant", "class_weight_balanced")
        mlflow.log_param("class_weight", "balanced")

        model = LogisticRegression(max_iter=1000, random_state=912, class_weight={0: 0.08, 1: 0.92})
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        log_classification_metrics(y_test, preds, proba)

    # --- SMOTE ---
    with mlflow.start_run(run_name="logreg_smote"):
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("variant", "smote")

        smote = SMOTE(random_state=912)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        mlflow.log_param("resampled_train_size", len(X_res))

        model = LogisticRegression(max_iter=1000)
        model.fit(X_res, y_res)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        log_classification_metrics(y_test, preds, proba)


# ============================================================
# Decision Tree: sweep over max_depth (one run per depth)
# ============================================================
def run_decision_tree_experiments(X_train, X_test, y_train, y_test):
    depths = [3, 5, 7, 10, 15]
    for depth in depths:
        with mlflow.start_run(run_name=f"decision_tree_depth_{depth}"):
            mlflow.log_param("model_type", "DecisionTree")
            mlflow.log_param("max_depth", depth)

            model = DecisionTreeClassifier(max_depth=depth, random_state=912)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            proba = model.predict_proba(X_test)[:, 1]
            log_classification_metrics(y_test, preds, proba)


# ============================================================
# Random Forest: vanilla (expected to be a poor result) + constrained
# ============================================================
def run_random_forest_experiments(X_train, X_test, y_train, y_test):
    # --- Vanilla (kept intentionally, since it's a useful "what NOT to do"
    # comparison point in the MLflow dashboard) ---
    with mlflow.start_run(run_name="random_forest_vanilla"):
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("variant", "vanilla_unconstrained")
        mlflow.log_param("n_estimators", 200)

        model = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=912,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        log_classification_metrics(y_test, preds, proba)

    # --- Constrained ---
    with mlflow.start_run(run_name="random_forest_constrained"):
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("variant", "constrained")
        mlflow.log_param("max_depth", 10)          # match your actual chosen constraints
        mlflow.log_param("min_samples_leaf", 100)  # match your actual chosen constraints
        mlflow.log_param("n_estimators", 250)

        model = RandomForestClassifier(
            n_estimators=250,
            max_depth=10,
            min_samples_leaf=100,    # each leaf must have at least 50 samples
            class_weight="balanced",
            random_state= 912,
            n_jobs=-1
        )  
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        log_classification_metrics(y_test, preds, proba)


# ============================================================
# XGBoost + top-20 feature importance artifact
# ============================================================
def run_xgboost_experiment(X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name="xgboost"):
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_param("n_estimators", 800)
        mlflow.log_param("learning_rate", 0.04)
        mlflow.log_param("max_depth", 4)
        mlflow.log_param("early_stopping_rounds", 30)

        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        print(f"scale_pos_weight: {scale_pos_weight:.2f}")

        model = xgb.XGBClassifier(
            n_estimators=800,
            learning_rate=0.04,
            max_depth=4,
            scale_pos_weight=scale_pos_weight,
            random_state=912,
            n_jobs=-1,
            eval_metric="auc",
            early_stopping_rounds=30,  # stop if no improvement for 50 rounds
        )

        model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        log_classification_metrics(y_test, preds, proba)

        # Feature importance plot — this IS a legitimate MLflow artifact,
        # same category as the confusion matrix image
        importances = pd.Series(model.feature_importances_, index=X_train.columns)
        top20 = importances.sort_values(ascending=False).head(20)

        fig, ax = plt.subplots(figsize=(8, 6))
        top20[::-1].plot(kind="barh", ax=ax)
        ax.set_title("Top 20 Feature Importances (XGBoost)")
        mlflow.log_figure(fig, "feature_importance.png")
        plt.close(fig)


# ============================================================
# LightGBM: randomized search + threshold tuning
# ============================================================
def run_lightgbm_experiment(X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name="lightgbm_randomized_search"):
        mlflow.log_param("model_type", "LightGBM")
    
        param_dist = {
            "n_estimators"      : randint(200, 1000),
            "learning_rate"     : uniform(0.01, 0.09),      # 0.01 to 0.10
            "max_depth"         : randint(3, 10),
            "num_leaves"        : randint(15, 63),
            "min_child_samples" : randint(20, 200),
            "subsample"         : uniform(0.6, 0.4),         # 0.6 to 1.0
            "colsample_bytree"  : uniform(0.6, 0.4),         # 0.6 to 1.0
            "reg_alpha"         : uniform(0, 0.5),           # L1
            "reg_lambda"        : uniform(0, 0.5),           # L2
            "scale_pos_weight"  : [1, 1.25, 1.5, 2,  3, 5],         # imbalance handling options
}

        base_model = lgb.LGBMClassifier(random_state=912)

        search=RandomizedSearchCV(estimator=base_model, param_distributions=param_dist, n_iter=10, random_state=912)
    
        search.fit(X_train, y_train)

        # Log the winning hyperparameters found by the search
        mlflow.log_params(search.best_params_)

        best_model = search.best_estimator_
        proba = best_model.predict_proba(X_test)[:, 1]

        # --- Threshold tuning ---
        chosen_threshold = 0.29  # replace with your actual chosen threshold
        mlflow.log_param("classification_threshold", chosen_threshold)

        preds = (proba >= chosen_threshold).astype(int)
        log_classification_metrics(y_test, preds, proba)

        # Log the model itself — this is what lets you reload/compare
        # actual model artifacts later, not just their metrics
        mlflow.lightgbm.log_model(best_model, "model")

        return best_model  # returned so main() can optionally save it as the final artifact


# ============================================================
# Main
# ============================================================
def main():
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test, _ = load_and_scale_data()

    print("Running Logistic Regression experiments...")
    run_logistic_regression_experiments(X_train, X_test, y_train, y_test)

    print("Running Decision Tree experiments...")
    run_decision_tree_experiments(X_train, X_test, y_train, y_test)

    print("Running Random Forest experiments...")
    run_random_forest_experiments(X_train, X_test, y_train, y_test)

    print("Running XGBoost experiment...")
    run_xgboost_experiment(X_train, X_test, y_train, y_test)

    print("Running LightGBM experiment (randomized search)...")
    run_lightgbm_experiment(X_train, X_test, y_train, y_test)


    print("\nAll experiments logged. Run `mlflow ui` to view the comparison dashboard.")


if __name__ == "__main__":
    main()