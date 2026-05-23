"""
forecasting.py — Weather temperature forecasting with 5 models.

Models:
  1. Linear Regression  (scikit-learn)
  2. Random Forest      (scikit-learn)
  3. XGBoost
  4. Prophet
  5. LSTM (TensorFlow/Keras)
"""

import warnings
warnings.filterwarnings("ignore")
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import xgboost as xgb

from utils import get_logger, timer, evaluate_model, save_figure, MODELS_DIR

logger = get_logger("forecasting")
TARGET = "temperature_celsius"
RANDOM_STATE = 42
TEST_SIZE = 0.2

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


def _train_test(df):
    X, y = _prepare_Xy(df)
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def _plot_actual_vs_pred(y_true, y_pred, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0f1117")
    for ax, (data_x, data_y, title_) in zip(axes, [
        (y_true[:500], y_pred[:500], "Actual vs Predicted"),
        (y_true, y_true - y_pred, "Residuals"),
    ]):
        ax.set_facecolor("#0f1117")
        if title_ == "Actual vs Predicted":
            ax.scatter(data_x, data_y, alpha=0.4, color="#00d4ff", s=10)
            lims = [min(data_x.min(), data_y.min()), max(data_x.max(), data_y.max())]
            ax.plot(lims, lims, "r--")
            ax.set_xlabel("Actual (°C)", color="#e0e0e0")
            ax.set_ylabel("Predicted (°C)", color="#e0e0e0")
        else:
            ax.hist(data_y, bins=50, color="#00d4ff", edgecolor="#222", alpha=0.85)
            ax.set_xlabel("Residual (°C)", color="#e0e0e0")
        ax.set_title(f"{model_name} — {title_}", color="#e0e0e0")
        ax.tick_params(colors="#e0e0e0")
    plt.tight_layout()
    fname = f"forecast_{model_name.lower().replace(' ','_')}.png"
    return save_figure(fig, fname)


@timer
def train_linear_regression(df):
    logger.info("Training Linear Regression …")
    X_tr, X_te, y_tr, y_te = _train_test(df)
    model = LinearRegression()
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    joblib.dump(model, MODELS_DIR / "linear_regression.pkl")
    _plot_actual_vs_pred(y_te.values, y_pred, "Linear Regression")
    return evaluate_model("Linear Regression", y_te.values, y_pred)


@timer
def train_random_forest(df):
    logger.info("Training Random Forest …")
    X_tr, X_te, y_tr, y_te = _train_test(df)
    model = RandomForestRegressor(n_estimators=200, max_depth=12,
                                   min_samples_leaf=4, n_jobs=-1, random_state=RANDOM_STATE)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    joblib.dump(model, MODELS_DIR / "random_forest.pkl")
    _plot_actual_vs_pred(y_te.values, y_pred, "Random Forest")
    return evaluate_model("Random Forest", y_te.values, y_pred)


@timer
def train_xgboost(df):
    logger.info("Training XGBoost …")
    X_tr, X_te, y_tr, y_te = _train_test(df)
    model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                              subsample=0.8, colsample_bytree=0.8,
                              random_state=RANDOM_STATE, verbosity=0, tree_method="hist")
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    y_pred = model.predict(X_te)
    model.save_model(str(MODELS_DIR / "xgboost.json"))
    _plot_actual_vs_pred(y_te.values, y_pred, "XGBoost")
    return evaluate_model("XGBoost", y_te.values, y_pred)


@timer
def train_prophet(df):
    logger.info("Training Prophet …")
    try:
        from prophet import Prophet
    except ImportError:
        logger.warning("Prophet not installed — skipping.")
        return {"Model": "Prophet", "RMSE": None, "MAE": None, "MAPE": None, "R2": None}

    if "datetime" not in df.columns or TARGET not in df.columns:
        return {"Model": "Prophet", "RMSE": None, "MAE": None, "MAPE": None, "R2": None}

    ts = df[["datetime", TARGET]].dropna().rename(columns={"datetime": "ds", TARGET: "y"})
    ts = ts.set_index("ds").resample("h").mean().reset_index().dropna()
    split = int(len(ts) * 0.8)
    train, test = ts.iloc[:split], ts.iloc[split:]

    model = Prophet(daily_seasonality=True, weekly_seasonality=True,
                    yearly_seasonality=True, changepoint_prior_scale=0.1)
    model.fit(train)
    future = model.make_future_dataframe(periods=len(test), freq="h")
    forecast = model.predict(future)
    y_pred = forecast["yhat"].iloc[split:].values
    y_true = test["y"].values
    n = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[:n], y_pred[:n]
    joblib.dump(model, MODELS_DIR / "prophet_model.pkl")
    _plot_actual_vs_pred(y_true, y_pred, "Prophet")
    return evaluate_model("Prophet", y_true, y_pred)


@timer
def train_lstm(df, look_back=24):
    logger.info("Training LSTM …")
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
        tf.get_logger().setLevel("ERROR")
    except ImportError:
        logger.warning("TensorFlow not installed — skipping LSTM.")
        return {"Model": "LSTM", "RMSE": None, "MAE": None, "MAPE": None, "R2": None}

    if TARGET not in df.columns or "datetime" not in df.columns:
        return {"Model": "LSTM", "RMSE": None, "MAE": None, "MAPE": None, "R2": None}

    ts = (df[["datetime", TARGET]].dropna()
          .set_index("datetime").resample("h").mean().dropna())

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(ts[[TARGET]])

    X, y = [], []
    for i in range(len(scaled) - look_back):
        X.append(scaled[i:i+look_back])
        y.append(scaled[i+look_back])
    X, y = np.array(X), np.array(y)

    split = int(len(X) * 0.8)
    X_tr, X_te, y_tr, y_te = X[:split], X[split:], y[:split], y[split:]

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(look_back, 1)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_tr, y_tr, epochs=30, batch_size=64, validation_split=0.1,
              verbose=0, callbacks=[EarlyStopping(patience=5, restore_best_weights=True)])

    y_pred = scaler.inverse_transform(model.predict(X_te, verbose=0)).flatten()
    y_true = scaler.inverse_transform(y_te).flatten()
    model.save(str(MODELS_DIR / "lstm_model.keras"))
    joblib.dump(scaler, MODELS_DIR / "lstm_scaler.pkl")
    _plot_actual_vs_pred(y_true, y_pred, "LSTM")
    return evaluate_model("LSTM", y_true, y_pred)


def plot_model_comparison(results):
    df_res = pd.DataFrame([r for r in results if r.get("RMSE") is not None])
    if df_res.empty:
        return None
    metrics = ["RMSE", "MAE", "R2"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor="#0f1117")
    fig.suptitle("Model Performance Comparison", color="#e0e0e0", fontsize=16)
    colors = ["#00d4ff", "#ff6b6b", "#ffd93d", "#6bcb77", "#c77dff"]
    for ax, metric in zip(axes, metrics):
        ax.set_facecolor("#0f1117")
        vals = df_res.set_index("Model")[metric].sort_values(ascending=(metric != "R2"))
        bars = ax.bar(vals.index, vals.values, color=colors[:len(vals)])
        for bar, v in zip(bars, vals.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", color="#e0e0e0", fontsize=8)
        ax.set_title(metric, color="#e0e0e0")
        ax.tick_params(colors="#e0e0e0")
        ax.set_xticklabels(vals.index, rotation=30, ha="right", color="#e0e0e0", fontsize=8)
    plt.tight_layout()
    return save_figure(fig, "08_model_comparison.png")


@timer
def run_forecasting(df):
    results = []
    for fn in [train_linear_regression, train_random_forest,
               train_xgboost, train_prophet, train_lstm]:
        try:
            results.append(fn(df))
        except Exception as exc:
            logger.warning(f"{fn.__name__} failed: {exc}")
    plot_model_comparison(results)
    return results


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    from preprocessing import run_preprocessing
    df, _, _ = run_preprocessing(save=False)
    results = run_forecasting(df)
    print(pd.DataFrame(results))
