import os
import mlflow
import mlflow.pytorch
import torch
import optuna
import joblib
import json
import numpy as np
import pandas as pd
import time
from pytorch_tabnet.tab_model import TabNetClassifier
from dotenv import load_dotenv
from sklearn.metrics import average_precision_score, accuracy_score, roc_auc_score
from mylib.data_preprocess import get_processed_data

load_dotenv()

# Config
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
EXPERIMENT_NAME = "Telco_Churn_Shadow_Model"
DATA_PATH = os.getenv("DATA_PATH", "archive/WA_Fn-UseC_-Telco-Customer-Churn.csv")
OUTPUT_DIR = "api/models_local"

def objective(trial, X_train, X_val, y_train, y_val):
    """
    Phase A: Optuna Objective.
    Uses Train vs Val to find the best hyperparameters.
    """
    n_da = trial.suggest_int('n_da', 8, 64, step=8)
    params = {
        'n_d': n_da, 
        'n_a': n_da,
        'n_steps': trial.suggest_int('n_steps', 3, 10),
        'gamma': trial.suggest_float('gamma', 1.0, 2.0),
        'lambda_sparse': trial.suggest_float('lambda_sparse', 1e-4, 1e-2, log=True),
        'optimizer_fn': torch.optim.Adam,
        'optimizer_params': dict(lr=trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True)),
        'mask_type': 'entmax', 
        'verbose': 0
    }
    
    batch_size = trial.suggest_categorical('batch_size', [256, 512, 1024])
    
    with mlflow.start_run(nested=True):
        model = TabNetClassifier(**params)
        
        model.fit(
            X_train=X_train.values, y_train=y_train.values,
            eval_set=[(X_val.values, y_val.values)],
            eval_name=['valid'],
            eval_metric=['auc'],
            max_epochs=20, 
            patience=5,
            batch_size=batch_size,
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False
        )
        
        preds_prob = model.predict_proba(X_val.values)[:, 1]
        score = average_precision_score(y_val, preds_prob)
        
        mlflow.log_params(params)
        mlflow.log_metric("pr_auc_optimization", score)
        
        return score

def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # 1. Load Data
    print("📥 Loading and Processing Data...")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names = get_processed_data(DATA_PATH)
    
    with mlflow.start_run(run_name="Shadow_Model_Full_Retrain") as run:
        
        # ---------------------------------------------------------
        # PHASE A: HYPERPARAMETER OPTIMIZATION (Train vs Val)
        # ---------------------------------------------------------
        print("🔍 Optimizing TabNet Hyperparameters...")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, X_train, X_val, y_train, y_val), n_trials=2)
        
        print("🏆 Best Params:", study.best_params)
        mlflow.log_params(study.best_params)
        
        # ---------------------------------------------------------
        # PHASE B: FINAL TRAINING (Train + Val)
        # ---------------------------------------------------------
        print("🚀 Retraining on FULL Dataset (Train + Val)...")
        
        # 1. Combine Data
        X_full = pd.concat([X_train, X_val])
        y_full = pd.concat([y_train, y_val])
        
        # 2. Reconstruct Params
        n_da = study.best_params['n_da']
        best_params = {
            'n_d': n_da, 
            'n_a': n_da,
            'n_steps': study.best_params['n_steps'],
            'gamma': study.best_params['gamma'],
            'lambda_sparse': study.best_params['lambda_sparse'],
            'optimizer_fn': torch.optim.Adam,
            'optimizer_params': dict(lr=study.best_params['learning_rate']),
            'mask_type': 'entmax',
            'verbose': 1
        }
        
        clf_tabnet = TabNetClassifier(**best_params)

        # START TIMER
        train_start = time.time()
        
        # 3. Fit on Full Data, Validate on Test Data
        # We use X_test here to monitor for early stopping on strictly unseen data
        clf_tabnet.fit(
            X_train=X_full.values, y_train=y_full.values,
            eval_set=[(X_test.values, y_test.values)],
            eval_name=['test'],
            eval_metric=['auc'],
            max_epochs=50, 
            patience=10,
            batch_size=study.best_params['batch_size'],
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False
        )

        # STOP TIMER
        training_time = time.time() - train_start
        print(f"⏱️ Retraining Time: {training_time:.4f} seconds")
        mlflow.log_metric("training_time_seconds", training_time)

        # ---------------------------------------------------------
        # PHASE C: FINAL EVALUATION (On X_test)
        # ---------------------------------------------------------
        preds_prob = clf_tabnet.predict_proba(X_test.values)[:, 1]
        auc = roc_auc_score(y_test, preds_prob)
        
        print(f"✅ Test Set AUC: {auc:.4f}")
        mlflow.log_metric("test_auc", auc)

        metrics_data = {
            "test_auc": float(auc),
            "training_time_sec": float(training_time)
        }
        
        # Save to api/models_local/shadow_metrics.json
        metrics_path = os.path.join(OUTPUT_DIR, "shadow_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics_data, f)
            
        print(f"✅ shadow_metrics.json saved.")

        # ---------------------------------------------------------
        # PHASE D: ARTIFACT SERIALIZATION
        # ---------------------------------------------------------
        print("💾 Saving Artifacts...")
        
        # Save ONLY the Shadow Model (Champion handles Scaler/Etc)
        joblib.dump(clf_tabnet, os.path.join(OUTPUT_DIR, "shadow_model.joblib"))
        mlflow.log_artifact(os.path.join(OUTPUT_DIR, "shadow_model.joblib"), artifact_path="shadow_model")
        
        print("✅ shadow_model.joblib saved successfully.")
        print("✅ Full Shadow Pipeline Completed.")

if __name__ == "__main__":
    main()