"""
feature_importance.py — SHAP-based feature importance + permutation importance.

Outputs:
  - SHAP summary plot
  - SHAP bar plot
  - Permutation importance chart
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

from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from utils import get_logger, timer, save_figure, MODELS_DIR

logger = get_logger("feature_importance")

TARGET = "temperature_celsius"
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
def compute_shap_importance(df: pd.DataFrame) -> Path:
    """Compute SHAP values from the saved XGBoost model."""
    try:
        import shap
    except ImportError:
        logger.warning("SHAP not installed — skipping.")
        return None

    model_path = MODELS_DIR / "xgboost.json"
    if not model_path.exists():
        logger.warning("XGBoost model not found — run forecasting first.")
        return None

    import xgboost as xgb
    model = xgb.XGBRegressor()
    model.load_model(str(model_path))

    X, y = _prepare_Xy(df)
    X_sample = X.sample(min(3000, len(X)), random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Summary plot
    fig, ax = plt.subplots(figsize=(10, 7), facecolor="#0f1117")
    shap.summary_plot(shap_values, X_sample, plot_type="bar",
                      show=False, color="#00d4ff")
    ax = plt.gca()
    ax.set_facecolor("#0f1117")
    plt.gcf().set_facecolor("#0f1117")
    ax.tick_params(colors="#e0e0e0")
    ax.set_xlabel("Mean |SHAP Value|", color="#e0e0e0")
    ax.set_title("SHAP Feature Importance (XGBoost)", color="#e0e0e0", fontsize=14)
    plt.tight_layout()
    p = save_figure(plt.gcf(), "10_shap_importance.png")

    # Beeswarm
    fig2, ax2 = plt.subplots(figsize=(10, 7), facecolor="#0f1117")
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.gcf().set_facecolor("#0f1117")
    plt.gca().tick_params(colors="#e0e0e0")
    plt.gca().set_xlabel("SHAP value", color="#e0e0e0")
    plt.title("SHAP Beeswarm Plot", color="#e0e0e0", fontsize=14)
    plt.tight_layout()
    save_figure(plt.gcf(), "10b_shap_beeswarm.png")

    logger.info("SHAP plots saved.")
    return p


@timer
def compute_permutation_importance(df: pd.DataFrame) -> Path:
    """Permutation importance using the Random Forest model."""
    model_path = MODELS_DIR / "random_forest.pkl"
    if not model_path.exists():
        logger.warning("Random Forest model not found — run forecasting first.")
        return None

    model = joblib.load(model_path)
    X, y = _prepare_Xy(df)
    X_te = X.sample(min(2000, len(X)), random_state=42)
    y_te = y.loc[X_te.index]

    result = permutation_importance(model, X_te, y_te, n_repeats=10,
                                    random_state=42, n_jobs=-1)
    imp = pd.Series(result.importances_mean, index=X_te.columns).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#0f1117")
    ax.set_facecolor("#0f1117")
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(imp)))
    ax.barh(imp.index, imp.values, color=colors)
    ax.set_title("Permutation Feature Importance (Random Forest)", color="#e0e0e0", fontsize=13)
    ax.set_xlabel("Mean decrease in R²", color="#e0e0e0")
    ax.tick_params(colors="#e0e0e0")
    plt.tight_layout()
    return save_figure(fig, "11_permutation_importance.png")


@timer
def run_feature_importance(df: pd.DataFrame) -> None:
    compute_shap_importance(df)
    compute_permutation_importance(df)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from preprocessing import run_preprocessing
    df, _, _ = run_preprocessing(save=False)
    run_feature_importance(df)
