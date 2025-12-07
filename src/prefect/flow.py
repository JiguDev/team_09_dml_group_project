# src/prefect/flow.py

from prefect import flow, task
import subprocess, sys, os
from datetime import datetime

def run_cmd(cmd: list):
    """Utility to run subprocess commands safely."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"Command failed: {cmd}")
    return result

# -------------------------------
# TASK 1: DVC Pull
# -------------------------------
@task
def dvc_pull():
    """
    Best-effort DVC pull.

    - If raw CSV already exists locally, we skip 'dvc pull' entirely.
    - If it's missing, we try 'dvc pull' once. On failure, we log a warning
      and continue; user is expected to place raw data manually.
    """
    print(" Pulling latest data from DVC (optional)...")

    raw_path = os.path.join("data", "raw", "city_day.csv")

    if os.path.exists(raw_path):
        print(f" [INFO] Raw data already present at {raw_path}. Skipping 'dvc pull'.")
        return

    try:
        run_cmd(["dvc", "pull"])
    except RuntimeError as e:
        print(f"[WARN] DVC pull failed ({e}). Continuing with existing local data.")

# -------------------------------
# TASK 2: Preprocessing
# -------------------------------
@task
def preprocess():
    print(" Running preprocessing...")
    run_cmd([sys.executable, "src/data/preprocess.py"])
    return "data/processed/city_day_processed.csv"

# -------------------------------
# TASK 3: Training
# -------------------------------
@task
def train():
    print(" Training model...")
    run_cmd([sys.executable, "src/models/train.py"])
    return "model.joblib"

# -------------------------------
# TASK 4: Evidently Monitoring
# -------------------------------
@task
def generate_monitoring():
    print(" Generating Evidently report...")
    run_cmd([sys.executable, "-m", "src.monitoring.evidently_report"])
    return "reports/aqi_drift_report.html"

# -------------------------------
# MAIN PIPELINE
# -------------------------------
@flow(name="AQI Full MLOps Pipeline")
def pipeline():
    print(" Starting AQI MLOps Pipeline")

    dvc_pull()
    processed = preprocess()
    model = train()
    report = generate_monitoring()

    print(" Pipeline Completed Successfully!")
    print(f"Processed file: {processed}")
    print(f"Model file: {model}")
    print(f"Evidently report: {report}")

if __name__ == "__main__":
    pipeline()
