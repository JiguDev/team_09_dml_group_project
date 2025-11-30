import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# MLflow for experiment tracking
import mlflow
import mlflow.sklearn

# --- Configuration ---
# Set the experiment name for MLflow
MLFLOW_EXPERIMENT_NAME = "Air Quality AQI Regression"
MLFLOW_REGISTERED_MODEL_NAME = "AQI_Predictor"
TRAIN_DATA_PATH = "data/processed/train.csv"
MODEL_OUTPUT_PATH = "models/model.pkl"

def train_and_log_model():
    # 1. Setup MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000") # Ensure MLflow server is running
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")

        # 2. Load DVC-tracked Data
        print(f"Loading data from {TRAIN_DATA_PATH}...")
        try:
            data = pd.read_csv(TRAIN_DATA_PATH)
        except FileNotFoundError:
            print(f"Error: Data file not found at {TRAIN_DATA_PATH}. Run DVC stage 'data_preparation' first.")
            return

        # 3. Data Preparation for Training (Simple Example)
        # Assuming data_prep.py has handled City, Date, and missing values.
        
        # Select features (Pollutants) and target (AQI)
        target = 'AQI'
        features = [col for col in data.columns if col not in [target, 'City', 'Date', 'AQI_Bucket']]
        
        X = data[features].fillna(data[features].mean()) # Final imputation
        # NEW FIX: Replace infinite values and check again
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(X.mean(), inplace=True) # Fill any new NaNs created by the replace
        y = data[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 4. Define Model and Hyperparameters
        params = {
            "n_estimators": 50,
            "max_depth": 7,
            "min_samples_split": 5,
            "random_state": 42
        }
        
        # Log parameters to MLflow
        mlflow.log_params(params)
        print("Model Parameters Logged to MLflow.")

        # 5. Train Model
        print("Starting model training...")
        model = RandomForestRegressor(**params, n_jobs=4)
        model.fit(X_train, y_train)

        # 6. Evaluate Model
        predictions = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        metrics = {
            "mae": mae,
            "r2_score": r2
        }

        # Log metrics to MLflow
        mlflow.log_metrics(metrics)
        print(f"Model Metrics Logged: MAE={mae:.2f}, R2={r2:.2f}")

        # 7. Save Model (Local Disk & MLflow Artifacts)
        
        # A. Save model locally for DVC tracking and API deployment
        os.makedirs(os.path.dirname(MODEL_OUTPUT_PATH), exist_ok=True)
        joblib.dump(model, MODEL_OUTPUT_PATH)
        print(f"Model saved locally for DVC tracking: {MODEL_OUTPUT_PATH}")

        # B. Log model as an artifact to MLflow and register it
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model", # Directory within MLflow artifact storage
            registered_model_name=MLFLOW_REGISTERED_MODEL_NAME,
            # Add environment information
            signature=mlflow.models.infer_signature(X_train, predictions)
        )
        print("Model logged to MLflow and registered.")

if __name__ == "__main__":
    train_and_log_model()