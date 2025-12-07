# tests/test_monitoring.py

import os

from src.data.preprocess import run as preprocess_run
from src.monitoring.evidently_report import generate_drift_report


def test_drift_report_generated(tmp_path):
    """
    End-to-end style monitoring test:

    1. Run preprocessing to ensure processed CSV exists.
    2. Generate drift report into a temporary path.
    3. Assert the HTML file was created.
    """

    # 1) Ensure processed data is available
    preprocess_run()

    # 2) Prepare a temp output HTML path
    report_path = tmp_path / "drift_report.html"

    # 3) Generate report
    out_path = generate_drift_report(
        processed_csv="data/processed/city_day_processed.csv",
        output_html=str(report_path),
    )

    # 4) Assertions
    assert isinstance(out_path, str)
    assert out_path.endswith(".html")
    assert os.path.exists(out_path)
