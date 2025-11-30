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
        raise FileNotFoundError("Processed data missing. Run preprocess first.")
    df = pd.read_csv(PROCESSED)
    # split reference/current for drift checking
    ref = df.sample(frac=0.8, random_state=42)
    cur = df.drop(ref.index)
    mapping = ColumnMapping()
    mapping.target = "AQI_Bucket_label" if 'AQI_Bucket_label' in df.columns else "AQI_Bucket"
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur, column_mapping=mapping)
    os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)
    report.save_html(OUT_HTML)
    print("Evidently report saved to", OUT_HTML)

if __name__ == "__main__":
    run_report()
