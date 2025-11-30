# src/models/forecast.py

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import joblib
import os

DATA_PATH = "data/processed/city_day_processed.csv"
MODEL_OUT = "forecast_arima.pkl"


def train_forecaster():
    """Train ARIMA model on AQI time series using real date index + cleaned AQI."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Processed data not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # ------------------------------------
    # 1. Validate required columns
    # ------------------------------------
    required = {"day", "month", "year"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Processed dataset must contain {required}, but found {df.columns.tolist()}"
        )

    # ------------------------------------
    # 2. Build proper datetime column
    # ------------------------------------
    df["__date__"] = pd.to_datetime(
        df["day"].astype(str) + "-" +
        df["month"].astype(str) + "-" +
        df["year"].astype(str),
        format="%d-%m-%Y",
        errors="coerce"
    )

    # Remove invalid dates
    df = df.dropna(subset=["__date__"])

    if df.empty:
        raise ValueError("No valid rows found with a proper date.")

    # Sort chronologically
    df = df.sort_values("__date__").set_index("__date__")

    # ------------------------------------
    # 3. Clean AQI values (MOST IMPORTANT FIX)
    # ------------------------------------
    if "AQI" not in df.columns:
        raise ValueError("AQI column missing in processed dataset")

    df["AQI"] = pd.to_numeric(df["AQI"], errors="coerce")

    # Fill missing AQI using forward + backward fill
    df["AQI"] = df["AQI"].fillna(method="ffill").fillna(method="bfill")

    # If still missing after fill → fail instead of feeding ARIMA NaN
    if df["AQI"].isna().any():
        raise ValueError("AQI still contains NaN after ffill/bfill")

    # Final forecast series
    series = df["AQI"].astype(float)

    # ------------------------------------
    # 4. Train ARIMA model
    # ------------------------------------
    print("Training ARIMA model...")
    model = ARIMA(series, order=(5, 1, 2))
    model_fit = model.fit()

    # ------------------------------------
    # 5. Save model
    # ------------------------------------
    joblib.dump(model_fit, MODEL_OUT)
    print(f"Forecast model saved at: {MODEL_OUT}")


if __name__ == "__main__":
    train_forecaster()