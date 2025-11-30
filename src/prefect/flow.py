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
    print(" Pulling latest data from DVC...")
    run_cmd(["dvc", "pull"])
    return True

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
    run_cmd([sys.executable, "src/monitoring/evidently_report.py"])
    return "monitoring_report.html"

# -------------------------------
# TASK 5: DVC Add + Git commit
# -------------------------------
@task
def dvc_track():
    print(" Tracking artifacts using DVC...")
    files = [
        "model.joblib",
        "data/processed/city_day_processed.csv",
        "artifacts/classification_report.json"
    ]

    for f in files:
        if os.path.exists(f):
            run_cmd(["dvc", "add", f])
        else:
            print(f"⚠ WARNING: {f} does not exist, skipping!")

    # Git commit
    run_cmd(["git", "add", "."])
    run_cmd(["git", "commit", "-m", f"Pipeline auto commit {datetime.now()}"])

    return True

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
    dvc_status = dvc_track()

    print(" Pipeline Completed Successfully!")
    print(f"Processed file: {processed}")
    print(f"Model file: {model}")
    print(f"Evidently report: {report}")
    print(f"DVC tracking: {dvc_status}")

if __name__ == "__main__":
    pipeline()
