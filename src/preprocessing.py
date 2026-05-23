"""
preprocessing.py — Data loading, cleaning, and feature engineering pipeline.

Steps performed:
1. Load the GlobalWeatherRepository CSV
2. Parse and engineer datetime features
3. Handle missing values (median/mode imputation)
4. Detect and cap outliers with IQR
5. Encode categorical columns
6. Normalise numeric features
7. Add continent and season columns
8. Save the cleaned dataset for downstream modules
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

from utils import (
    DATA_FILE, OUTPUT_DIR, get_logger, timer,
    month_to_season, get_continent,
)

logger = get_logger("preprocessing")


# ─────────────────────────────────────────────
# Column groups
# ─────────────────────────────────────────────
NUMERIC_COLS = [
    "temperature_celsius", "feels_like_celsius",
    "humidity", "wind_kph", "wind_degree",
    "pressure_mb", "precip_mm", "visibility_km",
    "uv_index", "gust_kph", "cloud", "dewpoint_celsius",
    "air_quality_PM2.5", "air_quality_PM10",
    "air_quality_CO", "air_quality_NO2",
    "air_quality_O3", "air_quality_SO2",
]

CATEGORICAL_COLS = ["condition_text", "wind_direction", "country"]


# ─────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────
@timer
def load_data(filepath: Path = DATA_FILE) -> pd.DataFrame:
    """Load raw CSV and do a quick sanity-check."""
    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset not found at {filepath}.\n"
            "Please download GlobalWeatherRepository.csv from Kaggle and place it in data/."
        )
    df = pd.read_csv(filepath, low_memory=False)
    logger.info(f"Loaded {len(df):,} rows × {df.shape[1]} columns from {filepath.name}")
    return df


# ─────────────────────────────────────────────
# Datetime engineering
# ─────────────────────────────────────────────
def parse_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the 'last_updated' column and extract time-based features."""
    df = df.copy()
    date_col = "last_updated" if "last_updated" in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    df["datetime"]    = df[date_col]
    df["year"]        = df["datetime"].dt.year
    df["month"]       = df["datetime"].dt.month
    df["day"]         = df["datetime"].dt.day
    df["hour"]        = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["day_of_year"] = df["datetime"].dt.dayofyear
    df["week"]        = df["datetime"].dt.isocalendar().week.astype(int)
    df["quarter"]     = df["datetime"].dt.quarter

    df["season"]      = df["month"].apply(month_to_season)
    df["continent"]   = df["country"].apply(get_continent) if "country" in df.columns else "Unknown"

    # Cyclical encoding for hour & month
    df["hour_sin"]    = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]    = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"]   = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]   = np.cos(2 * np.pi * df["month"] / 12)

    logger.info("Datetime features engineered.")
    return df


# ─────────────────────────────────────────────
# Missing value handling
# ─────────────────────────────────────────────
def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values with median (numeric) or mode (categorical)."""
    df = df.copy()
    num_cols  = [c for c in NUMERIC_COLS if c in df.columns]
    cat_cols  = [c for c in CATEGORICAL_COLS if c in df.columns]

    before = df.isnull().sum().sum()
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    after = df.isnull().sum().sum()
    logger.info(f"Missing values: {before} → {after}")
    return df


# ─────────────────────────────────────────────
# Outlier handling (IQR cap)
# ─────────────────────────────────────────────
def handle_outliers(df: pd.DataFrame, factor: float = 3.0) -> pd.DataFrame:
    """Winsorise outliers using IQR × factor."""
    df = df.copy()
    num_cols = [c for c in NUMERIC_COLS if c in df.columns]
    outlier_counts = {}

    for col in num_cols:
        Q1, Q3  = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR     = Q3 - Q1
        lo, hi  = Q1 - factor * IQR, Q3 + factor * IQR
        n_out   = ((df[col] < lo) | (df[col] > hi)).sum()
        if n_out:
            outlier_counts[col] = n_out
            df[col] = df[col].clip(lo, hi)

    if outlier_counts:
        logger.info(f"Outliers capped: { {k: v for k, v in sorted(outlier_counts.items(), key=lambda x: -x[1])[:5]} } …")
    return df


# ─────────────────────────────────────────────
# Encoding
# ─────────────────────────────────────────────
def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Label-encode categorical columns; return df and encoder map."""
    df = df.copy()
    encoders = {}
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    logger.info(f"Encoded columns: {list(encoders.keys())}")
    return df, encoders


# ─────────────────────────────────────────────
# Scaling
# ─────────────────────────────────────────────
def scale_features(df: pd.DataFrame) -> tuple[pd.DataFrame, MinMaxScaler]:
    """Min-Max scale numeric columns; return df and scaler."""
    df = df.copy()
    num_cols = [c for c in NUMERIC_COLS if c in df.columns]
    scaler   = MinMaxScaler()
    scaled   = scaler.fit_transform(df[num_cols])
    scaled_df = pd.DataFrame(scaled, columns=[f"{c}_scaled" for c in num_cols], index=df.index)
    df = pd.concat([df, scaled_df], axis=1)
    logger.info(f"Scaled {len(num_cols)} numeric columns.")
    return df, scaler


# ─────────────────────────────────────────────
# AQI composite
# ─────────────────────────────────────────────
def compute_aqi_composite(df: pd.DataFrame) -> pd.DataFrame:
    """Create a simple composite AQI index from PM2.5 and PM10."""
    df = df.copy()
    pm25 = df.get("air_quality_PM2.5", pd.Series(0, index=df.index))
    pm10 = df.get("air_quality_PM10",  pd.Series(0, index=df.index))
    df["aqi_composite"] = 0.6 * pm25 + 0.4 * pm10

    bins   = [0, 12, 35.4, 55.4, 150.4, 250.4, np.inf]
    labels = ["Good", "Moderate", "Unhealthy (SG)", "Unhealthy", "Very Unhealthy", "Hazardous"]
    df["aqi_category"] = pd.cut(df["aqi_composite"], bins=bins, labels=labels)
    return df


# ─────────────────────────────────────────────
# Master pipeline
# ─────────────────────────────────────────────
@timer
def run_preprocessing(save: bool = True) -> tuple[pd.DataFrame, dict, MinMaxScaler]:
    """
    Full preprocessing pipeline.

    Returns
    -------
    df       : cleaned, feature-engineered DataFrame
    encoders : dict of LabelEncoders
    scaler   : fitted MinMaxScaler
    """
    df = load_data()
    df = parse_datetime(df)
    df = handle_missing(df)
    df = handle_outliers(df)
    df = compute_aqi_composite(df)
    df, encoders = encode_categoricals(df)
    df, scaler   = scale_features(df)

    logger.info(f"Final shape: {df.shape}")

    if save:
        out_path = OUTPUT_DIR / "reports" / "cleaned_data.parquet"
        df.to_parquet(out_path, index=False)
        logger.info(f"Cleaned data saved → {out_path}")

    return df, encoders, scaler


if __name__ == "__main__":
    df, encoders, scaler = run_preprocessing()
    print(df.head(3))
