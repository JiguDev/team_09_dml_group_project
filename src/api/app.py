# src/api/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

MODEL_PATH = "model.joblib"  # CPU-friendly model saved by train.py; if using xgboost/lgbm files change accordingly
app = FastAPI(title="City Day AQI Prediction API")

class Payload(BaseModel):
    data: dict

def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    # fallback: check other model files
    if os.path.exists("model.xgb") or os.path.exists("model.txt"):
        raise FileNotFoundError("GPU model found (xgb/txt); convert to a CPU pickled model or set up proper loader.")
    raise FileNotFoundError("No model found. Train first.")

model = None

@app.on_event("startup")
def startup():
    global model
    model = load_model()

@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/predict")
def predict(payload: Payload):
    try:
        df = pd.DataFrame([payload.data])
        preds = model.predict(df)
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df).tolist()
        return {"prediction": int(preds[0]), "probabilities": proba}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
