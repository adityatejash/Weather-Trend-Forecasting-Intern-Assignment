"""
ensemble.py — Stacking ensemble model combining all base learners.

Strategy:
  - Base models: Linear Regression, Random Forest, XGBoost
  - Meta-learner: Ridge Regression trained on OOF predictions
  - Evaluation: RMSE, MAE, MAPE, R²
"""

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from utils import get_logger, timer, evaluate_model, save_figure, MODELS_DIR

logger = get_logger("ensemble")

TARGET = "temperature_celsius"
RANDOM_STATE = 42

FEATURE_COLS = [
    "humidity", "wind_kph", "pressure_mb", "precip_mm",
    "visibility_km", "uv_index", "cloud", "dewpoint_celsius",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "day_of_year", "quarter",
]


def _prepare_Xy(df):
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].copy()
    y = df[TARGET].copy()
    mask = X.notna().all(axis=1) & y.notna()
    return X[mask], y[mask]


@timer
def train_stacking_ensemble(df: pd.DataFrame) -> dict:
    """Train a stacking ensemble and return metrics."""
    logger.info("Training Stacking Ensemble …")
    X, y = _prepare_Xy(df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    base_learners = [
        ("lr",  LinearRegression()),
        ("rf",  RandomForestRegressor(n_estimators=100, max_depth=8,
                                       n_jobs=-1, random_state=RANDOM_STATE)),
        ("xgb", xgb.XGBRegressor(n_estimators=200, learning_rate=0.05,
                                   max_depth=5, verbosity=0,
                                   random_state=RANDOM_STATE, tree_method="hist")),
    ]

    meta_learner = Ridge(alpha=1.0)

    stacking = StackingRegressor(
        estimators=base_learners,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=-1,
        passthrough=False,
    )

    stacking.fit(X_tr, y_tr)
    y_pred = stacking.predict(X_te)

    joblib.dump(stacking, MODELS_DIR / "stacking_ensemble.pkl")
    logger.info(f"Stacking model saved → {MODELS_DIR / 'stacking_ensemble.pkl'}")

    # plot
    _plot_ensemble_result(y_te.values, y_pred)

    return evaluate_model("Stacking Ensemble", y_te.values, y_pred)


def _plot_ensemble_result(y_true, y_pred):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0f1117")
    BG, TC, AC = "#0f1117", "#e0e0e0", "#00d4ff"

    ax = axes[0]
    ax.set_facecolor(BG)
    ax.scatter(y_true[:600], y_pred[:600], alpha=0.4, color=AC, s=10)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1.5)
    ax.set_title("Stacking Ensemble — Actual vs Predicted", color=TC)
    ax.set_xlabel("Actual (°C)", color=TC); ax.set_ylabel("Predicted (°C)", color=TC)
    ax.tick_params(colors=TC)

    ax = axes[1]
    ax.set_facecolor(BG)
    residuals = y_true - y_pred
    ax.hist(residuals, bins=60, color=AC, edgecolor="#222", alpha=0.85)
    ax.axvline(0, color="red", linestyle="--", linewidth=1)
    ax.set_title("Stacking Ensemble — Residual Distribution", color=TC)
    ax.set_xlabel("Residual (°C)", color=TC)
    ax.tick_params(colors=TC)

    plt.tight_layout()
    save_figure(fig, "09_stacking_ensemble.png")


@timer
def run_ensemble(df: pd.DataFrame) -> dict:
    return train_stacking_ensemble(df)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from preprocessing import run_preprocessing
    df, _, _ = run_preprocessing(save=False)
    result = run_ensemble(df)
    print(result)
