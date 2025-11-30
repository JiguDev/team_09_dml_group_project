# 🌟 **AQI Prediction – Full End-to-End MLOps Pipeline**  
### **FastAPI • MLflow • DVC • Prefect • Docker • GitHub Actions • Evidently AI**

---

## 📌 **Project Overview**

This project implements a **complete end-to-end MLOps pipeline** using **only local, open-source tools**, as required for the Mini Project.

It includes:

### ✔️ **AQI Category Prediction**  
Machine Learning classifier (Random Forest)

### ✔️ **AQI Forecast for Future Dates**  
Time-Series forecasting using ARIMA

---

## 🧩 **Tech Stack**

| Component | Tool |
|----------|------|
| Backend API | FastAPI |
| Workflow Orchestration | Prefect |
| Experiment Tracking | MLflow |
| Data Versioning | DVC |
| CI/CD Automation | GitHub Actions |
| Monitoring | Evidently AI |
| Containerization | Docker |
| Modeling | Scikit-Learn, Statsmodels |

---

# 🚀 **1. Project Structure**

```
📁 india-aqi-mlops
│
├── 📁 .github
│   └── 📁 workflows
│       └── ci.yml
│
├── 📁 artifacts
│   ├── classification_report.json
│   ├── confusion_matrix.png
│
├── 📁 data
│   ├── 📁 raw
│   │   ├── city_day.csv
│   │   └── city_day.csv.dvc
│   └── 📁 processed
│       └── city_day_processed.csv
│
├── 📁 dvc-storage
│   └── (DVC remote data — kept minimal)
│
├── 📁 mlflow
│   └── 📁 mlruns
│       └── (Experiment folders automatically created by MLflow)
│
├── 📁 notebooks
│   └── eda.py
│
├── 📁 reports
│   └── evidently_report.html
│
├── 📁 src
│   ├── 📁 api
│   │   └── app.py
│   ├── 📁 data
│   │   └── preprocess.py
│   ├── 📁 models
│   │   ├── train.py
│   │   └── forecast.py
│   ├── 📁 monitoring
│   │   └── evidently_report.py
│   └── 📁 prefect
│       └── flow.py
│
├── 📁 tests
│   └── test_api.py
│
├── .dvcignore
├── .gitignore
├── dvc.yaml
├── dvc.lock
├── Dockerfile
├── docker-compose.yml
├── forecast_arima.pkl
├── model.joblib
├── params.yaml
├── README.md
├── requirements.txt
└── start.sh
```

---

# 🛠️ **2. Installation Guide (Beginner-Friendly)**

Follow these steps **exactly** even if you’re new to MLOps.

---

## ⭐ Step 1 — Clone the Repository

```sh
git clone <YOUR_REPO_URL>
cd <project-folder>
```

---

## ⭐ Step 2 — Create & Activate Virtual Environment

```sh
python -m venv .venv
```

### Activate:

#### Windows
```sh
.\.venv\Scripts\activate
```

#### Linux/Mac
```sh
source .venv/bin/activate
```

## ✅ How to Remove Virtual Environment (.venv)

If you created your environment using:

```
python -m venv .venv
```

then your virtual environment exists simply as a folder named **`.venv`**.  
Deleting it will completely remove the environment.

---

### 🪟 **Windows (PowerShell / CMD)**

```powershell
Remove-Item -Recurse -Force .\.venv
```

If you get a permission error:

```
Remove-Item -Recurse -Force .\.venv -ErrorAction Ignore
```
🐧 Linux / macOS
```
rm -rf .venv
```
⚠ Before Deleting, Deactivate the Environment

Windows/Linux/macOS:
```
deactivate
```
---

## ⭐ Step 3 — Install Dependencies

```sh
pip install --upgrade pip # if this do not works, try this:
D:\MTech\DML\india-aqi-mlops\.venv\Scripts\python.exe -m pip install --upgrade pip # Replace with your path
pip install -r requirements.txt
```
---

# 🧱 **3. DVC Pipeline Setup**

### Initialize DVC (already configured)

```sh
dvc init
```

### Track raw dataset

```sh
dvc add data/raw/city_day.csv
git add data/raw/city_day.csv.dvc .gitignore
git commit -m "Added raw dataset"
```

### If needed, delete stale Evidently reports:

Windows:
```
del reports/evidently_report.html
```

Linux/Mac:
```
rm reports/evidently_report.html
```

---

# 🔄 **4. Run the Full DVC Pipeline**

```sh
dvc repro
```

This runs:

- `src/data/preprocess.py`
- `src/models/train.py`
- `src/models/forecast.py`
- `src/monitoring/evidently_generate.py` *(if configured)*

---

# 🧪 **5. Train Models Manually (Optional)**

### Preprocess
```sh
python src/data/preprocess.py
```

### Train classifier
```sh
python src/models/train.py
```

### Train ARIMA forecaster
```sh
python src/models/forecast.py
```

---

# 📊 **6. MLflow Tracking Dashboard**

Start MLflow UI:

```sh
mlflow ui --port 5000
```

Open:

👉 http://127.0.0.1:5000

You will see:

- Parameters  
- Metrics  
- Confusion Matrix  
- Classification Report  
- Saved Models  
- Run History  

---

# 🌐 **7. Run FastAPI Server**

```sh
uvicorn src.api.app:app --reload --port 8000
```

Open:

- API Docs → http://127.0.0.1:8000/docs  
- Health Check → http://127.0.0.1:8000/health  

---

## Example Endpoints

### 1️⃣ Health Check
```
GET /health
```

### 2️⃣ AQI Classification
```
POST /classify
```

### 3️⃣ Forecast AQI for a Future Date
```
POST /forecast_date
```

---

# 🧪 **8. Run Unit Tests**

```sh
pytest -q
```

---

# 🐳 **9. Dockerization (Full MLOps Stack)**

---

## ⭐ Dockerfile

```
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## ⭐ Build Docker Image

```sh
docker build -t aqi-app .
```

---

## ⭐ Run Container

```sh
docker run -p 8000:8000 aqi-app
```

Open API:

👉 http://localhost:8000/docs  

---

# 🧩 **10. Docker Compose – Full Stack Deployment**

Includes:

✔ FastAPI  
✔ MLflow Server  
✔ Persisted Volumes  

Run:

```sh
docker compose up --build
```

Services:

- API → http://localhost:8000  
- MLflow → http://localhost:5000  

---

# 📈 **11. Monitoring with Evidently**

Generate drift report:

```sh
python src/monitoring/evidently_generate.py
```

Output saved:

```
reports/evidently_report.html
```

Open manually in browser.

---

# 🤖 **12. Prefect Workflow Orchestration**

Start UI:

```sh
prefect orion start
```

Run flow:

```sh
python prefect/flow.py
```

Prefect UI:

👉 http://127.0.0.1:4200

---

# 🔁 **13. CI/CD with GitHub Actions**

The `ci.yml` pipeline performs:

✔ Install dependencies  
✔ Preprocess data  
✔ Train model  
✔ Run tests  
✔ Upload model artifact  

Trigger:

- Push to `jigyasu-mlops` branch  
- Pull Request → `main`

---

# 📦 **14. Model Artifacts**

| File | Description |
|------|-------------|
| `model.joblib` | RandomForest classifier |
| `forecast_arima.pkl` | ARIMA model |
| `city_day_processed.csv` | Processed dataset |
| `confusion_matrix.png` | Evaluation plot |
| `classification_report.json` | Detailed performance metrics |

---

# 🎉 **Project Complete**

This README is fully detailed and beginner-friendly.  
If you want, I can also generate:

- 📘 Final Report PDF  
- 🎞 Demo Video Script  
- 🖼 Architecture Diagram  
- 📊 Monitoring Dashboard Guide  
- 📝 Submission Format Document  

Just say **"generate report"**, **"generate diagram"**, or **"generate demo script"**.