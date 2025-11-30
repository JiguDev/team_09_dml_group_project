# Use a minimal base image
FROM python:3.10-slim

WORKDIR /app

# 1. Install uv
RUN pip install uv

# 2. Copy the minimal requirements for the serving API
# Create a service/requirements.txt that only lists fastapi, uvicorn, scikit-learn, joblib
COPY service/requirements.txt . 

# 3. Install dependencies using uv
RUN uv sync --no-cache --system

# 4. Copy required components
# Copy model artifact and application code
COPY models/model.pkl /app/models/
COPY src/predict.py /app/src/
COPY service/app.py /app/

# 5. Expose the port FastAPI runs on
EXPOSE 8000

# 6. Command to run the FastAPI server using Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]