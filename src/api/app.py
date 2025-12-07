# src/api/app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
import os

# -----------------------------
# Model Paths
# -----------------------------
CLASSIFIER_PATH = "model.joblib"
FORECASTER_PATH = "forecast_arima.pkl"

model_classifier = None
model_forecaster = None

# Maximum allowed future days for ARIMA forecasting
MAX_FUTURE_DAYS = 120

# ------------------------------------------------------
# FastAPI App
# ------------------------------------------------------
app = FastAPI(
    title="AQI Prediction API",
    description="Predict AQI Bucket class from pollutant concentrations using trained ML model.",
    version="1.0.0"
)

# ------------------------------------------------------
# Startup Event & Load Model
# ------------------------------------------------------
@app.on_event("startup")
def startup_event():
    global model_classifier, model_forecaster
    print("Using THIS app.py file")
    try:
        # classifier
        if os.path.exists(CLASSIFIER_PATH):
            model_classifier = joblib.load(CLASSIFIER_PATH)
            print("Classifier Loaded.")

        # forecaster
        if os.path.exists(FORECASTER_PATH):
            model_forecaster = joblib.load(FORECASTER_PATH)
            print("Forecaster Loaded.")
    except Exception as e:
        print("Failed to load model:", e)

# ------------------------------------------------------
# Request Schemas
# ------------------------------------------------------
class AQIInput(BaseModel):
    PM2_5: float = Field(..., example=110)
    PM10: float = Field(..., example=180)
    NO2: float = Field(..., example=25)
    SO2: float = Field(..., example=8)
    CO: float = Field(..., example=1.2)
    O3: float = Field(..., example=30)

    year: int = Field(..., example=2023)
    month: int = Field(..., example=11)
    day: int = Field(..., example=15)
    weekday: int = Field(..., example=3)
    season: int = Field(..., example=3)

    City_Amaravati: int = Field(..., example=0)
    City_Amritsar: int = Field(..., example=0)
    City_Bengaluru: int = Field(..., example=0)
    City_Chennai: int = Field(..., example=0)
    City_Delhi: int = Field(..., example=1)
    City_Gurugram: int = Field(..., example=0)
    City_Hyderabad: int = Field(..., example=0)
    City_Jaipur: int = Field(..., example=0)
    City_Jorapokhar: int = Field(..., example=0)
    City_Lucknow: int = Field(..., example=0)
    City_Mumbai: int = Field(..., example=0)
    City_Other: int = Field(..., example=0)
    City_Patna: int = Field(..., example=0)
    City_Thiruvananthapuram: int = Field(..., example=0)
    City_Visakhapatnam: int = Field(..., example=0)

class ForecastInput(BaseModel):
    year: int = Field(..., example=2020)
    month: int = Field(..., example=10)
    day: int = Field(..., example=15)
    city: str = Field(..., example="Hyderabad")


# ------------------------------------------------------
# AQI Label Mapping
# ------------------------------------------------------
AQI_BUCKET_MAP = {
    0: "Good",
    1: "Moderate",
    2: "Satisfactory",
    3: "Poor",
    4: "Very Poor",
    5: "Severe"
}

# ------------------------------------------------------
# Health Check
# ------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# ------------------------------------------------------
# Root endpoint
# ------------------------------------------------------
@app.get("/")
def home():
    return {
        "message": "AQI Prediction API is running",
        "docs": "/docs",
        "redoc": "/redoc"
    }

# -----------------------------
# AQI Label Mapping
# -----------------------------
def aqi_to_bucket(aqi):
    if aqi <= 50: return "Good"
    if aqi <= 100: return "Moderate"
    if aqi <= 200: return "Satisfactory"
    if aqi <= 300: return "Poor"
    if aqi <= 400: return "Very Poor"
    return "Severe"

# -----------------------------
# FORECAST Endpoint (simple horizon forecast)
# -----------------------------
@app.get("/forecast")
def forecast(days: int = 7):
    if model_forecaster is None:
        raise HTTPException(status_code=500, detail="Forecast model not loaded")

    if days < 1 or days > 30:
        raise HTTPException(status_code=400, detail="days must be between 1 and 30")

    # Load processed data once for fallback
    df = pd.read_csv("data/processed/city_day_processed.csv")

    try:
        # Primary: ARIMA forecast
        result = model_forecaster.forecast(steps=days)
        values = [float(v) for v in result]
    except Exception as e:
        # Fallback: repeat last observed AQI
        print(f"[WARN] ARIMA forecast failed in /forecast ({e}), using naive last-value forecast.")
        last_aqi = float(df["AQI"].dropna().iloc[-1])
        values = [last_aqi for _ in range(days)]

    buckets = [aqi_to_bucket(v) for v in values]

    return {
        "requested_days": days,
        "forecast_values": values,
        "forecast_buckets": buckets,
    }

# ===== Required feature order from processed dataset =====
REQUIRED_FEATURE_ORDER = [
    "PM2.5", "PM10", "NO2", "SO2", "CO", "O3",
    "year", "month", "day", "weekday", "season",
    "city_Amaravati", "city_Amritsar", "city_Bengaluru", "city_Chennai",
    "city_Delhi", "city_Gurugram", "city_Hyderabad", "city_Jaipur",
    "city_Jorapokhar", "city_Lucknow", "city_Mumbai", "city_Other",
    "city_Patna", "city_Thiruvananthapuram", "city_Visakhapatnam",
]

# API → model column rename mapping
RENAME_MAP = {
    "PM2_5": "PM2.5"   # convert user input → model feature
}

# ------------------------------------------------------
# CLASSIFICATION Endpoint
# ------------------------------------------------------
@app.post("/classify")
def predict(input_data: AQIInput):
    if model_classifier is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    try:
        data = input_data.dict()

        # Map API field names -> model feature names
        rename_map = {
            "PM2_5": "PM2.5",
            "City_Amaravati": "city_Amaravati",
            "City_Amritsar": "city_Amritsar",
            "City_Bengaluru": "city_Bengaluru",
            "City_Chennai": "city_Chennai",
            "City_Delhi": "city_Delhi",
            "City_Gurugram": "city_Gurugram",
            "City_Hyderabad": "city_Hyderabad",
            "City_Jaipur": "city_Jaipur",
            "City_Jorapokhar": "city_Jorapokhar",
            "City_Lucknow": "city_Lucknow",
            "City_Mumbai": "city_Mumbai",
            "City_Other": "city_Other",
            "City_Patna": "city_Patna",
            "City_Thiruvananthapuram": "city_Thiruvananthapuram",
            "City_Visakhapatnam": "city_Visakhapatnam",
        }
        
        for old, new in rename_map.items():
            # all fields are present in the Pydantic model, so .pop is safe
            data[new] = data.pop(old)
        df = pd.DataFrame([data])
        for col in REQUIRED_FEATURE_ORDER:
            if col not in df.columns:
                df[col] = 0
        df = df[REQUIRED_FEATURE_ORDER]
        pred = model_classifier.predict(df)[0]
        proba = model_classifier.predict_proba(df)[0].tolist() if hasattr(model_classifier, "predict_proba") else None
        return {
            "predicted_label": int(pred),
            "predicted_bucket": AQI_BUCKET_MAP.get(int(pred), "Unknown"),
            "probabilities": proba,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
CITY_COLUMNS = [
    "City_Amaravati","City_Amritsar","City_Bengaluru","City_Chennai",
    "City_Delhi","City_Gurugram","City_Hyderabad","City_Jaipur",
    "City_Jorapokhar","City_Lucknow","City_Mumbai","City_Other",
    "City_Patna","City_Thiruvananthapuram","City_Visakhapatnam"
]

def city_to_onehot(city_name: str):
    city_name = city_name.strip().lower()
    onehot = {col: 0 for col in CITY_COLUMNS}

    matched = False
    for col in CITY_COLUMNS:
        cname = col.replace("City_", "").lower()
        if cname == city_name:
            onehot[col] = 1
            matched = True
            break

    if not matched:
        onehot["City_Other"] = 1

    return onehot

from datetime import datetime

@app.post("/forecast_date")
def forecast_date(request: ForecastInput):
    if model_forecaster is None:
        raise HTTPException(status_code=500, detail="Forecast model not loaded")

    # Parse date from request
    try:
        future_date = datetime(request.year, request.month, request.day)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid date")

    # -------------------------
    # LOAD DATA
    # -------------------------
    df = pd.read_csv("data/processed/city_day_processed.csv")

    # Build proper datetime column from year / month / day
    df["__date__"] = pd.to_datetime(
        df["day"].astype(int).astype(str)
        + "-"
        + df["month"].astype(int).astype(str)
        + "-"
        + df["year"].astype(int).astype(str),
        format="%d-%m-%Y",
        errors="coerce",
    )

    df = df.dropna(subset=["__date__"])

    if df.empty:
        raise HTTPException(status_code=500, detail="No valid dates found in dataset")

    last_date = df["__date__"].max()

    # Date must be strictly after dataset end
    if future_date <= last_date:
        raise HTTPException(
            status_code=400,
            detail=f"Date must be after dataset end ({last_date.date()})",
        )

    # Days ahead horizon
    days_ahead = (future_date - last_date).days

    if days_ahead < 1:
        raise HTTPException(
            status_code=400,
            detail=f"Computed days_ahead={days_ahead}, must be >= 1",
        )

    if days_ahead > MAX_FUTURE_DAYS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Max {MAX_FUTURE_DAYS} future days allowed from "
                f"{last_date.date()} (you requested {days_ahead})"
            ),
        )

    # -------------------------
    # FORECAST AQI with fallback
    # -------------------------
    try:
        forecast_values = model_forecaster.forecast(steps=days_ahead)
        aqi_value = float(forecast_values[-1])
    except Exception as e:
        # Fallback: naive last observed AQI
        print(f"[WARN] ARIMA forecast failed in /forecast_date ({e}), using naive last-value forecast.")
        aqi_value = float(df["AQI"].dropna().iloc[-1])

    aqi_value = max(0.0, aqi_value)
    bucket = aqi_to_bucket(aqi_value)

    # -------------------------
    # Build classification payload (API schema)
    # -------------------------
    payload = {
        "PM2_5": 0.0,
        "PM10": 0.0,
        "NO2": 0.0,
        "SO2": 0.0,
        "CO": 0.0,
        "O3": 0.0,
        "year": request.year,
        "month": request.month,
        "day": request.day,
        "weekday": future_date.weekday(),
        "season": request.month % 12 // 3,
    }

    payload.update(city_to_onehot(request.city))

    return {
        "input_date": future_date.strftime("%Y-%m-%d"),
        "dataset_last_date": last_date.strftime("%Y-%m-%d"),
        "days_ahead": days_ahead,
        "forecast_aqi": aqi_value,
        "forecast_bucket": bucket,
        "classify_ready_payload": payload,
    }


# ------------------------------------------------------
# Local Server Entry Point
# ------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)