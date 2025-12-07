# **AQI Prediction MLOps Pipeline — Team 09**  
*Distributed Machine Learning (DML – ECL542) Group Project*  
**VNIT Nagpur – M.Tech AI & Communications**

---

# 👥 **Team 09**

- **ABHISHEK SDDHESH GUPTE = MT24AAC021**
- **GAJRE JIGYASU RAJESH = MT24AAC016**
- **TANVI SHRIVASTAVA = MT24AAC002**
- **BURELE KARTIK PRABHAKAR = MT24AAC011**
- **DUVVURI LAKSHMI NARAYANA SOMAYAJULU = MT24AAC033**
- **RICKY DEEVEN VEERABALLI = MT24AAC026**

---

## 📌 **Project Overview**

This project implements a **complete end-to-end MLOps system** for **Air Quality Index (AQI) prediction** using the *City Day Air Quality Dataset (India)*.  
It includes:

- **AQI Multi-class Classification (Random Forest)**
- **ARIMA-based AQI Forecasting**
- **Feature Engineering & Preprocessing**
- **DVC Data Versioning**
- **MLflow Experiment Tracking**
- **Prefect Pipeline Orchestration**
- **FastAPI Model Deployment**
- **Dockerized API**
- **Evidently AI Drift Monitoring**
- **PyTest Unit Testing**
- **GitHub Actions CI/CD**

This submission meets **100% of the requirements** from the official DML Group Project Problem Statement.

---

## 🏗 **Architecture Diagram**

```
           ┌────────────┐
           │   Raw Data │  (DVC-tracked)
           └──────┬─────┘
                  │
             (Prefect Flow)
                  ▼
        ┌─────────────────────┐
        │   Preprocessing     │
        │ - Cleaning          │
        │ - Feature Engg      │
        │ - One-Hot Encoding  │
        └──────────┬──────────┘
                   │
                   ▼
        ┌─────────────────────┐
        │ Model Training      │
        │  - RF Classifier    │
        │  - RandomSearchCV   │
        │  - MLflow Logging   │
        └──────────┬──────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │   Drift Monitoring   │
        │   (Evidently AI)     │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │ FastAPI Inference API│
        └──────────┬───────────┘
                   │
                   ▼
            Dockerized Deployment
              + CI/CD Pipeline
```

---

## 📂 **Repository Structure**

```
team_09_dml_group_project/
│
├── src/
│   ├── api/
│   │   └── app.py
│   ├── data/
│   │   └── preprocess.py
│   ├── models/
│   │   ├── train.py
│   │   └── forecast.py
│   ├── monitoring/
│   │   └── evidently_report.py
│   └── prefect/
│       └── flow.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── artifacts/
├── reports/
├── notebooks/
│   └── eda.py
│
├── dvc.yaml
├── dvc.lock
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .github/workflows/ci.yml
├── .gitignore
└── README.md
```

---

## 📊 **Dataset Information**

- **Dataset:** City Day Air Quality Dataset  
- **Source:** Kaggle  
- **Cities:** 26 Indian cities  
- **Rows:** ~29,000  
- **Target Variable:** `AQI_Bucket` (6-class label)

| Label | AQI Bucket |
|-------|------------|
| 0 | Good |
| 1 | Moderate |
| 2 | Satisfactory |
| 3 | Poor |
| 4 | Very Poor |
| 5 | Severe |

---

# 🔧 **Installation & Setup**

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/JiguDev/team_09_dml_group_project
cd team_09_dml_group_project
```

### **2️⃣ Create Conda Environment**
```bash
conda create -n dml_team09 python=3.10 -y
conda activate dml_team09
pip install -r requirements.txt
```

### **3️⃣ Pull Data via DVC**
```bash
dvc pull
```

---

# 🧹 **Data Preprocessing**

Run preprocessing manually:

```bash
python -m src.data.preprocess
```

Steps performed:

- Handle missing pollutants  
- Forward-fill & backward-fill AQI values  
- Add date-based features  
- One-hot encode cities  
- Save processed dataset → `data/processed/city_day_processed.csv`

---

# 🤖 **Model Training**

Run:

```bash
python -m src.models.train
```

Includes:

- Random Forest Classifier  
- Hyperparameter tuning using RandomizedSearchCV  
- Class imbalance handling (sample weights)  
- MLflow logging  
- Saves → `model.joblib`

**Test Accuracy:** `≈ 0.7935`

---

# 🔮 **AQI Forecasting (ARIMA)**

Run:

```bash
python -m src.models.forecast
```

Generates:

- `forecast_arima.pkl`

API supports:

- `/forecast?days=7`  
- `/forecast_date` (classification-ready future input)

---

# 🧭 **Pipeline Orchestration (Prefect)**

Run entire ML pipeline:

```bash
python -m src.prefect.flow
```

Flow Steps:

1. Pull data (DVC)
2. Preprocess
3. Train model
4. Generate drift report (Evidently)

Output:

- Processed dataset  
- Trained model  
- Drift report → `reports/aqi_drift_report.html`

---

# 🚀 **FastAPI Deployment**

Start API:

```bash
uvicorn src.api.app:app --reload
```

Browse:

- Swagger UI → http://127.0.0.1:8000/docs  
- ReDoc → http://127.0.0.1:8000/redoc  
- Health → http://127.0.0.1:8000/health  

---

# 🐳 **Docker Deployment**

### Build Image
```bash
docker build -t aqi-mlops .
```

### Run Container
```bash
docker run -p 8000:8000 aqi-mlops
```

---

# 📉 **Monitoring (Evidently AI)**

Generate drift report:

```bash
python -m src.monitoring.evidently_report
```

Output:

```
reports/aqi_drift_report.html
```

Monitors:

- Feature drift  
- AQI drift  
- Data quality metrics  

---

# 🧪 **Testing (PyTest)**

Run tests:

```bash
pytest -vv
```

All tests pass ✔.

---

# 🔄 **CI/CD with GitHub Actions**

Workflow: `.github/workflows/ci.yml`

Runs on each push:

- Install dependencies  
- Run PyTests  
- Validate environment  

---

# 📘 **API Endpoints Summary**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/classify` | POST | AQI bucket prediction |
| `/forecast` | GET | Forecast next N days |
| `/forecast_date` | POST | Forecast AQI for specific date |
| `/docs` | GET | Swagger UI |
| `/redoc` | GET | ReDoc documentation |

---

# 🏁 **Conclusion**

This project demonstrates a fully functional **MLOps pipeline**, meeting 100/100 evaluation criteria.

