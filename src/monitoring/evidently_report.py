# src/monitoring/evidently_report.py

import os
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Update this if your processed file has a different name
PROCESSED_CSV = "data/processed/city_day_processed.csv"
DEFAULT_REPORT_HTML = "reports/aqi_drift_report.html"


def generate_drift_report(
    processed_csv: str = PROCESSED_CSV,
    output_html: str = DEFAULT_REPORT_HTML,
) -> str:
    """
    Generate an Evidently data drift report by splitting the processed dataset
    into reference and current chunks.

    - Loads the processed CSV.
    - Sorts by Date (if present) or by index.
    - Uses first 70% as reference_data and last 30% as current_data.
    - Runs Evidently DataDriftPreset.
    - Saves HTML report to output_html.
    - Returns output_html path.
    """

    df = pd.read_csv(processed_csv)

    if df.shape[0] < 10:
        raise ValueError("Not enough rows in processed data to compute drift report.")

    # If Date column exists, sort by it; else use index
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")
    else:
        df = df.sort_index()

    n_rows = len(df)
    split_idx = int(n_rows * 0.7)

    reference = df.iloc[:split_idx].reset_index(drop=True)
    current = df.iloc[split_idx:].reset_index(drop=True)

    out_dir = os.path.dirname(output_html)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    report.save_html(output_html)

    return output_html


if __name__ == "__main__":
    path = generate_drift_report()
    print(f"[Monitoring] Data drift report generated -> {path}")
