from prefect import flow, task, get_run_logger
import subprocess
import os
from datetime import date

# --- 1. Utility Task to Run DVC Stages ---

@task(name="Run DVC Stage")
def run_dvc_stage(stage_name: str):
    """Executes a defined DVC stage using dvc repro."""
    logger = get_run_logger()
    logger.info(f"Starting DVC stage: {stage_name}")
    
    try:
        # Use dvc repro -s to run a specific stage
        # --force is often useful in MLOps to force re-run even if dependencies haven't changed
        result = subprocess.run(
            ["dvc", "repro", "-s", stage_name, "--force"], 
            check=True, # Raises CalledProcessError if the command returns a non-zero exit code
            capture_output=True,
            text=True
        )
        logger.info(f"DVC Stage Output:\n{result.stdout}")
        logger.info(f"DVC stage '{stage_name}' completed successfully.")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"DVC stage '{stage_name}' failed!")
        logger.error(f"Stderr: {e.stderr}")
        # Re-raise the error to fail the Prefect task
        raise RuntimeError(f"DVC stage {stage_name} failed: {e.stderr}") from e
    except FileNotFoundError:
        logger.error("DVC command not found. Is DVC installed and in PATH?")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during DVC execution: {e}")
        raise

# --- 2. Monitoring Task ---

@task(name="Run Evidently AI Monitoring")
def monitor_data_drift():
    """
    Executes the Evidently AI monitoring script after the training run.
    This simulates a check immediately after model creation (i.e., a data quality check 
    on the reference data) or a post-deployment batch check.
    """
    logger = get_run_logger()
    logger.info("Running Evidently AI monitoring script (src/monitor.py)...")
    
    try:
        # Execute the python monitoring script
        result = subprocess.run(
            ["python", "src/monitor.py"], 
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"Monitoring Script Output:\n{result.stdout}")
        logger.info("Evidently AI monitoring completed successfully.")

    except subprocess.CalledProcessError as e:
        logger.error("Monitoring script failed!")
        logger.error(f"Stderr: {e.stderr}")
        raise RuntimeError(f"Monitoring failed: {e.stderr}") from e

# --- 3. Main Flow Definition ---

@flow(name="Air Quality MLOps Training Pipeline")
def training_pipeline():
    """
    The main flow orchestrating data preparation, model training, 
    and monitoring checks.
    """
    
    # 1. Data Preparation Stage (DVC)
    # This task prepares the data and creates the DVC-tracked 'train.csv' 
    # and 'current_batch.csv' files.
    data_prep_result = run_dvc_stage.submit(stage_name="data_preparation") 
    
    # 2. Model Training Stage (DVC + MLflow)
    # This task depends on data_preparation to complete successfully.
    # It trains the model, logs it to MLflow, and saves 'model.pkl'.
    train_model_result = run_dvc_stage.submit(
        stage_name="train_model", 
        wait_for=[data_prep_result] # Explicit dependency link
    )
    
    # 3. Monitoring Stage (Evidently AI)
    # This task depends on both data and model artifacts being created.
    # It checks for data drift and model performance.
    monitor_data_drift.submit(
        wait_for=[train_model_result] # Explicit dependency link
    )
    
    logger = get_run_logger()
    logger.info("MLOps Training Pipeline flow submitted successfully.")

# --- Execution ---

if __name__ == "__main__":
    # Ensure all tools (DVC, MLflow UI, Prefect Server) are running before executing
    training_pipeline()