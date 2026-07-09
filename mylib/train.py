import os
import mlflow
import mlflow.sklearn
import xgboost as xgb
import matplotlib.pyplot as plt
import optuna
import joblib
import json

import sys
print(sys.executable)

import shap
import numpy as np
import pandas as pd
import time
from dotenv import load_dotenv
from sklearn.metrics import (
    average_precision_score, 
    precision_recall_curve, 
    accuracy_score, 
    confusion_matrix, 
    f1_score,
    ConfusionMatrixDisplay
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from mylib.data_preprocess import get_processed_data

load_dotenv()

# Configuración
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "Telco_Churn_Model")
DATA_PATH = os.getenv("DATA_PATH", "archive/WA_Fn-UseC_-Telco-Customer-Churn.csv")
OUTPUT_DIR = "api/models_local"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def objective(trial, X_train, X_val, y_train, y_val):
    """
    METRIC 1: Threshold INDEPENDENT (PR-AUC)
    Used purely to find the best Hyperparameters.
    """
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'booster': 'gbtree',
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 10.0),
    }

    with mlflow.start_run(nested=True):
        model = xgb.XGBClassifier(**params)#, use_label_encoder=False)
        model.fit(X_train, y_train)
        
        y_proba = model.predict_proba(X_val)[:, 1]
        score = average_precision_score(y_val, y_proba)
        
        mlflow.log_params(params)
        mlflow.log_metric("pr_auc_optimization", score)
        
        return score

def find_best_threshold(y_true, y_proba):
    """
    Finds the threshold that maximizes F1-Score using Precision-Recall Curve.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    ix = np.argmax(f1_scores)
    best_thresh = thresholds[ix]
    best_f1 = f1_scores[ix]
    
    return best_thresh, best_f1

def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # 1. Load Data
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names = get_processed_data(DATA_PATH)
    
    with mlflow.start_run(run_name="Production_Full_Retrain") as run:
        
        # ---------------------------------------------------------
        # PHASE A: HYPERPARAMETER OPTIMIZATION (Train vs Val)
        # ---------------------------------------------------------
        print("🔍 Optimizing Hyperparameters...")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, X_train, X_val, y_train, y_val), n_trials=2)
        
        print("🏆 Best Params:", study.best_params)
        mlflow.log_params(study.best_params)
        
        # ---------------------------------------------------------
        # PHASE B: THRESHOLD SELECTION (Train vs Val)
        # ---------------------------------------------------------
        # We need a temp model on X_train to find the threshold on X_val
        # before we mix them together.
        print("⚙️ calculating optimal threshold...")
        temp_model = xgb.XGBClassifier(**study.best_params)#, use_label_encoder=False)
        temp_model.fit(X_train, y_train)
        
        y_proba_val = temp_model.predict_proba(X_val)[:, 1]
        best_threshold, val_f1 = find_best_threshold(y_val, y_proba_val)
        
        print(f"🎯 Optimal Threshold (on Val): {best_threshold:.4f}")
        mlflow.log_param("optimal_threshold", best_threshold)
        
        # ---------------------------------------------------------
        # PHASE C: FULL RETRAINING & CALIBRATION (Train + Val)
        # ---------------------------------------------------------
        print("🚀 Retraining on FULL Dataset (Train + Val) with Calibration...")
        
        # 1. Combine Data
        X_full = pd.concat([X_train, X_val])
        y_full = pd.concat([y_train, y_val])
        
        # 2. Define Base Model
        base_model = xgb.XGBClassifier(**study.best_params)#, use_label_encoder=False)
        
        # 3. Wrap in CalibratedClassifierCV
        # cv=5 means it trains 5 models on different folds of X_full and averages probabilities
        calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv=5)
        
        start_time = time.time()
        calibrated_model.fit(X_full, y_full)
        training_time = time.time() - start_time
        
        print(f"⏱️ Retraining Time: {training_time:.4f}s")
        mlflow.log_metric("training_time_seconds", training_time)

        # ---------------------------------------------------------
        # PHASE D: FINAL EVALUATION (On X_test)
        # ---------------------------------------------------------
        # Now we check how good the Calibrated Model is on strictly unseen data
        print("🧪 Evaluating on Held-Out Test Set...")
        
        y_test_proba = calibrated_model.predict_proba(X_test)[:, 1]
        y_test_pred = (y_test_proba >= best_threshold).astype(int)
        
        test_acc = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        
        print(f"✅ Test Set Accuracy: {test_acc:.4f}")
        print(f"✅ Test Set F1 Score: {test_f1:.4f}")
        
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_f1_score", test_f1)

        # Save Metrics JSON
        metrics_data = {
            "accuracy": float(test_acc),
            "f1_score": float(test_f1),
            "training_time_sec": float(training_time)
        }
        
        metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics_data, f)

        # ---------------------------------------------------------
        # PHASE E: PLOTS & ARTIFACTS
        # ---------------------------------------------------------
        
        # 1. Confusion Matrix (Test Set)
        cm = confusion_matrix(y_test, y_test_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
        disp.plot(cmap="Blues")
        plt.title(f"Confusion Matrix (Test Set)")
        plt.savefig("confusion_matrix.png")
        plt.close()
        mlflow.log_artifact("confusion_matrix.png", artifact_path="plots")

        # 2. Calibration Curve (Test Set)
        prob_true, prob_pred = calibration_curve(y_test, y_test_proba, n_bins=10)
        plt.figure(figsize=(8, 6))
        plt.plot(prob_pred, prob_true, marker='o', label='Calibrated Model')
        plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curve (Test Set)')
        plt.legend()
        plt.grid(True)
        plt.savefig("calibration_curve.png", dpi=300)
        plt.close()
        mlflow.log_artifact("calibration_curve.png", artifact_path="plots")

        # 3. Save Artifacts
        # Note: We save the CalibratedClassifierCV object. 
        # It behaves just like a model (has predict/predict_proba).
        mlflow.sklearn.log_model(calibrated_model, name="model", serialization_format="pickle")
        joblib.dump(calibrated_model, os.path.join(OUTPUT_DIR, "model.joblib"))
        
        joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.joblib"))
        joblib.dump(feature_names, os.path.join(OUTPUT_DIR, "feature_names.joblib"))
        joblib.dump(best_threshold, os.path.join(OUTPUT_DIR, "threshold.joblib"))
        
        mlflow.log_artifact(os.path.join(OUTPUT_DIR, "scaler.joblib"), artifact_path="preprocessing")
        mlflow.log_artifact(os.path.join(OUTPUT_DIR, "feature_names.joblib"), artifact_path="preprocessing")
        mlflow.log_artifact(os.path.join(OUTPUT_DIR, "threshold.joblib"), artifact_path="preprocessing")

        # 4. SHAP (Special Handling)
        # CalibratedClassifierCV doesn't expose the tree structure easily for SHAP.
        # We fit a temporary standalone XGBoost on Full Data to generate the interpretation.
        print("📊 Generating SHAP plot (using proxy model)...")
        shap_model = xgb.XGBClassifier(**study.best_params)#, use_label_encoder=False)
        shap_model.fit(X_full, y_full)
        
        explainer = shap.Explainer(shap_model, X_full)
        # We use a sample of Test set for SHAP to speed it up
        shap_values = explainer(X_test)
        
        plt.figure(figsize=(10, 8))
        shap.plots.beeswarm(shap_values, max_display=12, show=False)
        plt.title("SHAP Feature Importance (Full Data)")
        plt.tight_layout()
        plt.savefig("shap_summary.png", bbox_inches='tight', dpi=300)
        plt.close()
        mlflow.log_artifact("shap_summary.png", artifact_path="plots")

        # Cleanup
        for f in ["shap_summary.png", "confusion_matrix.png", "calibration_curve.png"]:
            if os.path.exists(f):
                os.remove(f)

        print("✅ Full Pipeline Completed.")

if __name__ == "__main__":
    main()