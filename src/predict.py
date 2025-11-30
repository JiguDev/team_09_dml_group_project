import pandas as pd
import joblib
import os
import numpy as np

# Define the expected features based on the data_prep.py script
EXPECTED_FEATURES = [
    'PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 
    'Year', 'Month', 'DayOfWeek', 'DayOfYear'
]

MODEL_PATH = "models/model.pkl"

def load_model():
    """Load the trained model from the DVC-tracked path."""
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except FileNotFoundError:
        print(f"Error: Model file not found at {MODEL_PATH}. Ensure training flow was run.")
        return None

def preprocess_input(data: dict) -> pd.DataFrame:
    """
    Transforms raw input dictionary (from FastAPI) into the format 
    expected by the model.
    """
    # 1. Convert input to DataFrame
    input_df = pd.DataFrame([data])
    
    # 2. Re-order columns and fill missing (should be handled by the caller/API validation)
    # The FastAPI model (AirQualityInput) should ensure all pollutant features are present.
    
    # --- Feature Engineering Simulation ---
    # Since the input is instantaneous, we can't extract Year/Month/DayOfWeek 
    # easily without the 'Date'. For this local deployment, we assume the client 
    # sends these features based on the current date, OR they were engineered
    # in the data_prep.py script.
    
    # CRITICAL: Ensure the feature list matches the training input (e.g., fill with 0/mean if missing)
    
    # Simple check to ensure all expected columns are present.
    missing_cols = set(EXPECTED_FEATURES) - set(input_df.columns)
    if missing_cols:
        # In a real API, you would handle this more gracefully (e.g., raise HTTP error)
        raise ValueError(f"Input is missing required features: {missing_cols}")
        
    return input_df[EXPECTED_FEATURES]

def make_prediction(model, data: dict) -> float:
    """
    Takes the loaded model and raw data, preprocesses it, and returns the prediction.
    """
    if model is None:
        return np.nan # Or raise an exception

    try:
        processed_data = preprocess_input(data)
        prediction = model.predict(processed_data)[0]
        return float(prediction)
    except ValueError as e:
        print(f"Prediction failed due to data issue: {e}")
        return np.nan

if __name__ == "__main__":
    # Example usage for testing the script locally
    model = load_model()
    if model:
        sample_input = {
            'PM2.5': 50.0, 'PM10': 100.0, 'NO': 15.0, 'NO2': 40.0, 
            'NOx': 55.0, 'NH3': 10.0, 'CO': 1.0, 'SO2': 20.0, 'O3': 35.0,
            'Year': 2023, 'Month': 11, 'DayOfWeek': 5, 'DayOfYear': 300 
        }
        
        predicted_aqi = make_prediction(model, sample_input)
        print(f"Test Input: {sample_input}")
        print(f"Predicted AQI: {predicted_aqi:.2f}")