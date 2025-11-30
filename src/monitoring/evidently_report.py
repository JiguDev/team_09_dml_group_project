# src/monitoring/evidently_report.py
import pandas as pd
import os
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently import ColumnMapping

PROCESSED = "data/processed/city_day_processed.csv"
OUT_HTML = "reports/evidently_report.html"

def run_report():
    if not os.path.exists(PROCESSED):
        raise FileNotFoundError(
            f"Processed data missing: {PROCESSED}. Run preprocess first."
        )

    df = pd.read_csv(PROCESSED)

    # -------- Reference vs Current Split --------
    # Use 70/30 split to avoid empty samples
    ref = df.sample(frac=0.7, random_state=42)
    cur = df.drop(ref.index)

    # -------- Column Mapping --------
    mapping = ColumnMapping()
    if "AQI_Bucket_label" in df.columns:
        mapping.target = "AQI_Bucket_label"
    elif "AQI_Bucket" in df.columns:
        mapping.target = "AQI_Bucket"
    else:
        mapping.target = None  # safe fallback

    # -------- Generate Report --------
    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=ref,
        current_data=cur,
        column_mapping=mapping
    )

    # -------- Save Output --------
    os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)
    report.save_html(OUT_HTML)

    print(f"✔ Evidently report saved to {OUT_HTML}")


if __name__ == "__main__":
    run_report()