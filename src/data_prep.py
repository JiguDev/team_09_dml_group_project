import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# --- Configuration ---
RAW_DATA_PATH = "data/raw/city_day.csv"
PROCESSED_DATA_PATH = "data/processed/train.csv"
MONITOR_DATA_PATH = "data/processed/current_batch.csv"
TARGET_CITY = "Delhi" # Selected City for modeling
TEST_SIZE = 0.20 # 20% for testing the model training stage
MONITOR_BATCH_SIZE = 0.05 # 5% of the data for the monitoring batch

def data_cleaning_and_preparation():
    print(f"Starting data preparation for city: {TARGET_CITY}")
    
    # 1. Load Data
    try:
        df = pd.read_csv(RAW_DATA_PATH)
    except FileNotFoundError:
        print(f"ERROR: Raw data not found at {RAW_DATA_PATH}. Please ensure 'city_day.csv' is in 'data/raw/'.")
        return

    # 2. Initial Cleaning and Filtering
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['City', 'Date']).reset_index(drop=True)
    
    # Filter for the target city
    city_df = df[df['City'] == TARGET_CITY].copy()
    
    # Drop the AQI_Bucket column as we are performing Regression on AQI
    city_df = city_df.drop(columns=['AQI_Bucket']) 
    
    # 3. Handle Missing Values (Imputation)
    # Air quality data is time-series, so simple mean imputation is generally poor.
    # Linear interpolation is a better choice for small, scattered gaps.
    print(f"Missing values before imputation: \n{city_df.isnull().sum()}")

    # Apply forward fill (ffill) and then backward fill (bfill) to handle gaps
    # For long time gaps, interpolation is better.
    # We apply linear interpolation to all pollutant and target columns
    imputation_cols = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'AQI']
    city_df[imputation_cols] = city_df[imputation_cols].interpolate(
        method='linear', 
        limit_direction='both', # Fills missing values at both ends
        axis=0
    )

    # 4. Feature Engineering
    city_df['Year'] = city_df['Date'].dt.year
    city_df['Month'] = city_df['Date'].dt.month
    city_df['DayOfWeek'] = city_df['Date'].dt.dayofweek
    city_df['DayOfYear'] = city_df['Date'].dt.dayofyear
    
    # Remove the original Date column after extracting features
    city_df = city_df.drop(columns=['Date'])

    # Final check for remaining NaNs (should be zero after linear interpolation)
    if city_df.isnull().sum().any():
        print("WARNING: Some NaNs still remain. Inspect data for very long gaps.")
        
    print("Data cleaning and feature engineering complete.")
    return city_df

def save_processed_data(df):
    """Splits the clean data into train (reference) and monitoring batch."""
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    
    # 1. Separate the small batch for local monitoring simulation
    monitor_batch_size = int(len(df) * MONITOR_BATCH_SIZE)
    
    # The monitoring batch is the latest N days of data to simulate production data
    monitor_df = df.tail(monitor_batch_size)
    training_df = df.iloc[:-monitor_batch_size]

    # 2. Split the main training data into Train and Test for model development
    # (The test set is used inside train_model.py, the final output is just 'train.csv')
    X = training_df.drop(columns=['AQI', 'City'])
    y = training_df['AQI']

    # Splitting to ensure the model training script has a validation set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, shuffle=False, random_state=42)

    # Recombine for saving the full "reference" data for DVC/MLflow/Evidently AI reference
    # For simplicity, we save the full training dataset (including features/target)
    train_reference_df = pd.concat([X_train, X_test]).assign(AQI=pd.concat([y_train, y_test]))
    
    # 3. Save DVC Outputs
    train_reference_df.to_csv(PROCESSED_DATA_PATH, index=False)
    monitor_df.to_csv(MONITOR_DATA_PATH, index=False)

    print(f"Training reference data saved to: {PROCESSED_DATA_PATH} ({len(train_reference_df)} rows)")
    print(f"Monitoring batch data saved to: {MONITOR_DATA_PATH} ({len(monitor_df)} rows)")

if __name__ == "__main__":
    cleaned_df = data_cleaning_and_preparation()
    if cleaned_df is not None:
        save_processed_data(cleaned_df)