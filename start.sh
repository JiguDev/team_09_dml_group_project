#!/usr/bin/env bash
set -e
python -m venv .venv || true
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "Virtualenv ready. Place data/raw/city_day.csv, then run dvc init && dvc add data/raw/city_day.csv"
echo "To run full pipeline: dvc repro OR python src/prefect/flow.py"
