# -------------------------
# 1. BASE IMAGE
# -------------------------
FROM python:3.11-slim

# -------------------------
# 2. WORKDIR
# -------------------------
WORKDIR /app

# -------------------------
# 3. SYSTEM DEPENDENCIES (for numpy, pandas, mlflow, etc.)
# -------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# -------------------------
# 4. INSTALL PYTHON DEPENDENCIES
# -------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install dvc

# -------------------------
# 5. COPY APPLICATION CODE
# -------------------------
COPY . .

# -------------------------
# 6. EXPOSE PORT
# -------------------------
EXPOSE 8000

# -------------------------
# 7. START FASTAPI SERVER
# -------------------------
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]