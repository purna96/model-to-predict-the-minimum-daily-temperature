"""
utils.py — Shared utility functions for the Time Series Capstone Project
Minimum Daily Temperature Forecasting — Melbourne, Australia (1981–1990)
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

DATA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
DATA_FILE = "daily-minimum-temperatures.csv"


def load_data(filepath: str = DATA_FILE) -> pd.DataFrame:
    """Load and clean the Melbourne temperature dataset."""
    if not os.path.exists(filepath):
        import urllib.request
        urllib.request.urlretrieve(DATA_URL, filepath)

    df = pd.read_csv(filepath, parse_dates=["Date"], index_col="Date")
    df.columns = ["Temp"]
    df = df.sort_index()

    # Fill any missing dates
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
    df = df.reindex(full_range).ffill()

    # Handle numeric NaNs
    df["Temp"] = df["Temp"].interpolate(method="time")

    # Winsorize outliers (IQR)
    Q1, Q3 = df["Temp"].quantile(0.25), df["Temp"].quantile(0.75)
    IQR = Q3 - Q1
    df["Temp"] = df["Temp"].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time series features for ML models."""
    d = df.copy()

    # Calendar
    d["day_of_year"] = d.index.dayofyear
    d["day_of_week"] = d.index.dayofweek
    d["month"]       = d.index.month
    d["quarter"]     = d.index.quarter
    d["year"]        = d.index.year

    # Fourier terms
    for k in [1, 2, 3]:
        d[f"sin_{k}"] = np.sin(2 * np.pi * k * d["day_of_year"] / 365.25)
        d[f"cos_{k}"] = np.cos(2 * np.pi * k * d["day_of_year"] / 365.25)

    # Lag features
    for lag in [1, 2, 3, 7, 14, 21, 28, 30, 60, 90, 180, 365]:
        d[f"lag_{lag}"] = d["Temp"].shift(lag)

    # Rolling statistics
    for w in [7, 14, 30, 60, 90]:
        shifted = d["Temp"].shift(1)
        d[f"roll_mean_{w}"] = shifted.rolling(w).mean()
        d[f"roll_std_{w}"]  = shifted.rolling(w).std()
        d[f"roll_min_{w}"]  = shifted.rolling(w).min()
        d[f"roll_max_{w}"]  = shifted.rolling(w).max()

    # EWM
    d["ewm_7"]  = d["Temp"].shift(1).ewm(span=7).mean()
    d["ewm_30"] = d["Temp"].shift(1).ewm(span=30).mean()

    return d.dropna()


def create_sequences(data: np.ndarray, seq_len: int = 30):
    """Create sliding window sequences for DL models."""
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, label: str = "Model") -> dict:
    """Return MAE, RMSE, MAPE, R² metrics."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    r2   = r2_score(y_true, y_pred)
    return {
        "Model":    label,
        "MAE":      round(mae,  4),
        "RMSE":     round(rmse, 4),
        "MAPE(%)":  round(mape, 4),
        "R²":       round(r2,   4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MODEL PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────

def save_pickle(obj, path: str):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_comparison_table(path: str = "models/comparison_ranked.csv") -> pd.DataFrame:
    """Load the pre-computed model comparison table."""
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def ml_predict(model, df_feat: pd.DataFrame, scaler_X: StandardScaler) -> np.ndarray:
    """Run inference with an sklearn / xgboost / lightgbm model."""
    X = df_feat.drop(columns=["Temp"], errors="ignore")
    X_sc = scaler_X.transform(X)
    return model.predict(X_sc)


def dl_predict(model, series_scaled: np.ndarray,
               scaler: MinMaxScaler, seq_len: int = 30) -> np.ndarray:
    """Run inference with a Keras DL model (sequence input)."""
    X_seq, _ = create_sequences(series_scaled, seq_len)
    preds_sc = model.predict(X_seq, verbose=0).ravel()
    return scaler.inverse_transform(preds_sc.reshape(-1, 1)).ravel()
