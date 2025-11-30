import pandas as pd
import os
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset
# CRITICAL FIX 1: Explicitly import load_model and EXPECTED_FEATURES 
# from the prediction logic module (or redefine the load_model function).
# We'll redefine load_model here for self-containment, but ideally, 
# this logic should be imported from src/predict.py
import joblib 

# Define the expected features based on data_prep.py (must be consistent)
EXPECTED_FEATURES = [
    'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 
    'Year', 'Month', 'DayOfWeek', 'DayOfYear'
]
MODEL_PATH = "models/model.pkl"
REFERENCE_DATA_PATH = "data/processed/train.csv"
CURRENT_DATA_PATH = "data/processed/current_batch.csv"
REPORT_OUTPUT_DIR = "reports"
DATA_DRIFT_REPORT_NAME = "data_drift_report.html"
MODEL_PERFORMANCE_REPORT_NAME = "model_performance_report.html"


def load_model():
    """Load the trained model artifact."""
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}. Ensure the training run completed successfully.")
        return None

def generate_data_drift_report(ref_data: pd.DataFrame, curr_data: pd.DataFrame):
    """Generates and saves the Evidently AI Data Drift report."""
    
    print("Generating Data Drift Report...")
    # Define columns to ignore (metadata, target)
    ignore_cols = ['City', 'AQI']
    
    # Ensure only features used in training are passed for drift checking
    ref_features = ref_data[EXPECTED_FEATURES] 
    curr_features = curr_data[EXPECTED_FEATURES]

    data_drift_report = Report(metrics=[
        DataDriftPreset(),
    ])

    data_drift_report.run(
        reference_data=ref_features, 
        current_data=curr_features, 
        column_mapping=None
    )

    report_path = os.path.join(REPORT_OUTPUT_DIR, DATA_DRIFT_REPORT_NAME)
    os.makedirs(REPORT_OUTPUT_DIR, exist_ok=True)
    data_drift_report.save_html(report_path)
    print(f"Data Drift Report saved to {report_path}")


def generate_model_performance_report(curr_data: pd.DataFrame):
    """
    Generates and saves the Evidently AI Model Performance report.
    Note: This assumes the 'current' data batch has the ground truth ('AQI').
    """
    
    model = load_model()
    if model is None:
        return

    print("Generating Model Performance Report...")
    
    # Prepare features for prediction
    feature_cols = [col for col in curr_data.columns if col in EXPECTED_FEATURES]

    # CRITICAL FIX 2: Ensure the columns used for prediction exactly match EXPECTED_FEATURES
    # This prevents errors if columns are out of order or missing.
    curr_data['prediction'] = model.predict(curr_data[feature_cols])

    # Create and run the Regression Report
    regression_report = Report(metrics=[
        RegressionPreset(
            target_name='AQI',          # Actual ground truth
            prediction_name='prediction' # Model's output
        )
    ])
    
    # Use the current batch as both reference and current for metrics calculation
    regression_report.run(
        reference_data=curr_data, 
        current_data=curr_data, 
        column_mapping=None
    )
    
    report_path = os.path.join(REPORT_OUTPUT_DIR, MODEL_PERFORMANCE_REPORT_NAME)
    regression_report.save_html(report_path)
    print(f"Model Performance Report saved to {report_path}")


def run_monitoring():
    """Loads data and runs both drift and performance reports."""
    
    try:
        ref_df = pd.read_csv(REFERENCE_DATA_PATH)
        curr_df = pd.read_csv(CURRENT_DATA_PATH)
    except FileNotFoundError as e:
        # This is the most likely failure point if the DVC stage didn't run
        print(f"FATAL ERROR: Monitoring data not found at {e.filename}.")
        raise RuntimeError("Data files are missing. Ensure data_prep.py ran successfully.")

    print("\n--- Running Monitoring Pipeline ---")
    generate_data_drift_report(ref_df, curr_df)
    # Only curr_df is needed for performance, but we pass it as a positional argument.
    generate_model_performance_report(curr_df) 

if __name__ == "__main__":
    run_monitoring()