# src/data/preprocess.py
import pandas as pd
import numpy as np
import os

RAW = "data/raw/city_day.csv"
OUT = "data/processed/city_day_processed.csv"

def load_raw(path=RAW):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw CSV not found at {path}. Place the dataset there.")
    return pd.read_csv(path)

def standardize_column_names(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df

def find_pollutant_columns(df):
    # common pollutant column name patterns in city_day datasets
    candidates = {}
    for c in df.columns:
        lc = c.lower()
        if 'pm2' in lc and ('25' in lc or '.' in lc):
            candidates['PM2.5'] = c
        elif 'pm10' in lc:
            candidates['PM10'] = c
        elif 'no2' in lc:
            candidates['NO2'] = c
        elif 'so2' in lc:
            candidates['SO2'] = c
        elif lc == 'co' or 'co ' in lc:
            candidates['CO'] = c
        elif 'o3' in lc:
            candidates['O3'] = c
    return candidates

def aqi_bucket_to_label(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()

    mapping = {
        "good": 0,
        "satisfactory": 1,
        "moderate": 2,
        "poor": 3,
        "very poor": 4,
        "very_poor": 4,
        "verypoor": 4,
        "severe": 5
    }

    return mapping.get(s, np.nan)

def run():
    print("Loading raw data...")
    df = load_raw()
    df = standardize_column_names(df)

    # detect pollutant columns
    pol_map = find_pollutant_columns(df)
    print("Detected pollutant columns mapping:", pol_map)

    # normalize pollutant columns to known names
    for target_name, col in pol_map.items():
        df[target_name] = pd.to_numeric(df[col], errors='coerce')

    # AQI numeric column detection
    possible_aqi = None
    for c in df.columns:
        if c.lower() in ['aqi', 'aqi_value', 'aqi value', 'aqi_value']:
            possible_aqi = c
            break
    if possible_aqi is None:
        # search for 'aqi' substring
        for c in df.columns:
            if 'aqi' in c.lower():
                possible_aqi = c
                break

    if possible_aqi is None:
        print("Warning: Could not find numeric AQI column automatically.")
    else:
        df['AQI'] = pd.to_numeric(df[possible_aqi], errors='coerce')

    # Convert AQI_Bucket (string) to numeric label if exists
    bucket_col = None
    for c in df.columns:
        if 'bucket' in c.lower() or 'category' in c.lower() or 'class' in c.lower():
            bucket_col = c
            break

    if bucket_col:
        df['AQI_Bucket_label'] = df[bucket_col].apply(aqi_bucket_to_label)

    # If we don't have bucket label but have AQI, derive per NAQI ranges
    def aqi_to_bucket(aqi):
        if pd.isna(aqi):
            return np.nan
        aqi = float(aqi)
        if aqi <= 50:
            return 0
        elif aqi <= 100:
            return 1
        elif aqi <= 200:
            return 2
        elif aqi <= 300:
            return 3
        elif aqi <= 400:
            return 4
        else:
            return 5

    if 'AQI_Bucket_label' not in df.columns or df['AQI_Bucket_label'].isnull().all():
        if 'AQI' in df.columns:
            df['AQI_Bucket_label'] = df['AQI'].apply(aqi_to_bucket)
        else:
            raise ValueError("No AQI numeric column found and no bucket column found. Please inspect dataset.")

    # Select features: pollutant numeric columns and optional city/state/time features
    features = [col for col in ['PM2.5','PM10','NO2','SO2','CO','O3'] if col in df.columns]
    print("Using features:", features)

    # optional: extract date features if any date column exists
    date_col = None
    for c in df.columns:
        if 'date' in c.lower() or 'time' in c.lower():
            date_col = c
            break
    if date_col:
        df['__date__'] = pd.to_datetime(df[date_col], errors='coerce')
        df['year'] = df['__date__'].dt.year
        df['month'] = df['__date__'].dt.month
        df['day'] = df['__date__'].dt.day
        features += ['year','month','day']

        df['weekday'] = df['__date__'].dt.weekday
        df['season'] = df['month'] % 12 // 3
        features += ['weekday', 'season']

    # handle City/State categorical columns (keep top N)
    for c in df.columns:
        if c.lower() in ['city','station','location']:
            top = df[c].value_counts().nlargest(15).index.tolist()
            df[c] = df[c].apply(lambda x: x if x in top else 'Other')
            dummies = pd.get_dummies(df[c], prefix=c, drop_first=True)
            dummies = dummies.astype(int)
            df = pd.concat([df, dummies], axis=1)
            features += list(dummies.columns)
            df = df.drop(columns=[c])

    # Fill missing numeric values with median
    for f in features:
        if f in df.columns:
            if pd.api.types.is_numeric_dtype(df[f]):
                df[f] = df[f].fillna(df[f].median())
            else:
                # fill categorical na with mode
                df[f] = df[f].fillna(df[f].mode().iloc[0] if not df[f].mode().empty else 0)

    # Final processed df
    processed = df[features + ['AQI_Bucket_label', 'AQI']].copy()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    processed.to_csv(OUT, index=False)
    print(f"Saved processed data to {OUT}. Shape: {processed.shape}")

if __name__ == "__main__":
    run()
