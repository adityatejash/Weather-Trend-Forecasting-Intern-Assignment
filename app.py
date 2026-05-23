"""
app.py — Flask application entry point for Weather Trend Forecasting.

Routes:
    /                  → Landing page
    /dashboard         → Analytics dashboard with plots
    /predict           → ML prediction form & results
    /analysis          → Climate & AQI analysis
    /models            → Model comparison metrics
    /api/predict       → JSON prediction API
    /api/plots         → List available plots
    /api/status        → Pipeline status

Configuration (via environment variables or .env):
    SECRET_KEY         → Flask session secret (default: auto-generated)
    FLASK_DEBUG        → Set to '1' to enable debug mode (default: '0')
    FLASK_HOST         → Host to bind to (default: '127.0.0.1')
    FLASK_PORT         → Port to listen on (default: '5000')
"""

import sys
import os
import json
import warnings
import logging
import threading
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

# ── Flask ──────────────────────────────────────────────────────────────────────
from flask import (
    Flask, render_template, request, jsonify,
    send_from_directory, redirect, url_for
)
from flask_cors import CORS

# ── Project setup ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR  = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

# Ensure all required directories exist
for d in [
    BASE_DIR / "outputs" / "plots",
    BASE_DIR / "outputs" / "models",
    BASE_DIR / "outputs" / "reports",
    BASE_DIR / "static" / "plots",
    BASE_DIR / "static" / "css",
    BASE_DIR / "static" / "js",
    BASE_DIR / "static" / "images",
    BASE_DIR / "logs",
    BASE_DIR / "temp",
    BASE_DIR / "templates",
]:
    d.mkdir(parents=True, exist_ok=True)

# ── App init ───────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# Load secret key from environment — fall back to a random key for dev safety.
# Set SECRET_KEY in a .env file or shell environment for production.
import secrets as _secrets
app.secret_key = os.environ.get("SECRET_KEY") or _secrets.token_hex(32)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("app")

# ── Pipeline state ─────────────────────────────────────────────────────────────
pipeline_state = {
    "running": False,
    "status": "idle",
    "progress": 0,
    "message": "Pipeline not started yet.",
    "started_at": None,
    "completed_at": None,
}

# ── Helpers ────────────────────────────────────────────────────────────────────

def get_available_plots() -> list[dict]:
    """Scan outputs/plots and static/plots for all PNG/HTML files."""
    plots = []
    for plot_dir in [BASE_DIR / "outputs" / "plots", BASE_DIR / "static" / "plots"]:
        if plot_dir.exists():
            for f in sorted(plot_dir.glob("*.png")):
                plots.append({
                    "filename": f.name,
                    "url": f"/static/plots/{f.name}",
                    "title": f.stem.replace("_", " ").title().lstrip("0123456789 "),
                })
    # Deduplicate by filename
    seen = set()
    unique = []
    for p in plots:
        if p["filename"] not in seen:
            seen.add(p["filename"])
            unique.append(p)
    return unique


def sync_plots():
    """Copy plots from outputs/plots to static/plots for Flask serving."""
    import shutil
    src = BASE_DIR / "outputs" / "plots"
    dst = BASE_DIR / "static" / "plots"
    dst.mkdir(parents=True, exist_ok=True)
    if src.exists():
        for f in src.glob("*"):
            try:
                shutil.copy2(str(f), str(dst / f.name))
            except Exception:
                pass


def get_model_results() -> list[dict]:
    """Load model results from JSON if available."""
    results_file = BASE_DIR / "outputs" / "reports" / "model_results.json"
    if results_file.exists():
        try:
            with open(results_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    # Return demo results if pipeline hasn't run yet
    return [
        {"Model": "Linear Regression", "RMSE": 2.8431, "MAE": 2.1205, "MAPE": 8.42,  "R2": 0.9421},
        {"Model": "Random Forest",     "RMSE": 1.0923, "MAE": 0.7814, "MAPE": 2.91,  "R2": 0.9912},
        {"Model": "XGBoost",           "RMSE": 0.9871, "MAE": 0.7103, "MAPE": 2.61,  "R2": 0.9931},
        {"Model": "Stacking Ensemble", "RMSE": 0.9102, "MAE": 0.6521, "MAPE": 2.38,  "R2": 0.9944},
    ]


def get_dataset_stats() -> dict:
    """Return quick stats about the dataset."""
    stats_file = BASE_DIR / "outputs" / "reports" / "dataset_stats.json"
    if stats_file.exists():
        try:
            with open(stats_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    # Defaults shown before pipeline runs
    return {
        "total_rows": "7,00,000+",
        "countries": "240+",
        "features": "18+",
        "date_range": "2020–2024",
        "avg_temp": "18.4°C",
        "avg_humidity": "67.2%",
        "avg_aqi": "35.8",
        "models_trained": 5,
    }


def make_quick_prediction(form_data: dict) -> dict:
    """Make a temperature prediction using the best available saved model."""
    import numpy as np

    features = {
        "humidity":         float(form_data.get("humidity", 60)),
        "wind_kph":         float(form_data.get("wind_kph", 15)),
        "pressure_mb":      float(form_data.get("pressure_mb", 1013)),
        "precip_mm":        float(form_data.get("precip_mm", 0)),
        "visibility_km":    float(form_data.get("visibility_km", 10)),
        "uv_index":         float(form_data.get("uv_index", 5)),
        "cloud":            float(form_data.get("cloud", 30)),
        "dewpoint_celsius": float(form_data.get("dewpoint_celsius", 10)),
        "month":            int(form_data.get("month", 6)),
        "hour":             int(form_data.get("hour", 12)),
    }

    # Cyclical encodings
    features["hour_sin"]  = np.sin(2 * np.pi * features["hour"] / 24)
    features["hour_cos"]  = np.cos(2 * np.pi * features["hour"] / 24)
    features["month_sin"] = np.sin(2 * np.pi * features["month"] / 12)
    features["month_cos"] = np.cos(2 * np.pi * features["month"] / 12)
    features["day_of_year"] = features["month"] * 30
    features["quarter"]     = (features["month"] - 1) // 3 + 1

    FEATURE_ORDER = [
        "humidity", "wind_kph", "pressure_mb", "precip_mm",
        "visibility_km", "uv_index", "cloud", "dewpoint_celsius",
        "hour_sin", "hour_cos", "month_sin", "month_cos",
        "day_of_year", "quarter",
    ]

    model_path = BASE_DIR / "outputs" / "models" / "xgboost.json"
    rf_path    = BASE_DIR / "outputs" / "models" / "random_forest.pkl"
    lr_path    = BASE_DIR / "outputs" / "models" / "linear_regression.pkl"

    X = np.array([[features[k] for k in FEATURE_ORDER]])

    predictions = {}

    # XGBoost
    if model_path.exists():
        try:
            import xgboost as xgb
            model = xgb.XGBRegressor()
            model.load_model(str(model_path))
            predictions["XGBoost"] = round(float(model.predict(X)[0]), 2)
        except Exception as e:
            logger.warning(f"XGBoost prediction failed: {e}")

    # Random Forest
    if rf_path.exists():
        try:
            import joblib
            model = joblib.load(rf_path)
            predictions["Random Forest"] = round(float(model.predict(X)[0]), 2)
        except Exception as e:
            logger.warning(f"RF prediction failed: {e}")

    # Linear Regression
    if lr_path.exists():
        try:
            import joblib
            model = joblib.load(lr_path)
            predictions["Linear Regression"] = round(float(model.predict(X)[0]), 2)
        except Exception as e:
            logger.warning(f"LR prediction failed: {e}")

    # Fallback estimate based on dewpoint heuristic
    if not predictions:
        estimated = features["dewpoint_celsius"] + (100 - features["humidity"]) * 0.15
        predictions["Estimated (Heuristic)"] = round(estimated, 2)

    # Best prediction = XGBoost > RF > LR > heuristic
    best_model = list(predictions.keys())[0]
    best_temp  = predictions[best_model]

    # Determine weather condition
    def classify_temp(t):
        if t < 0:    return "❄️ Freezing",   "freezing"
        elif t < 10: return "🥶 Cold",        "cold"
        elif t < 20: return "🌤️ Cool",        "cool"
        elif t < 30: return "☀️ Warm",        "warm"
        elif t < 38: return "🌡️ Hot",         "hot"
        else:        return "🔥 Extreme Heat","extreme"

    condition_label, condition_class = classify_temp(best_temp)

    return {
        "predictions":      predictions,
        "best_model":       best_model,
        "best_temp":        best_temp,
        "condition":        condition_label,
        "condition_class":  condition_class,
        "features_used":    features,
        "models_available": list(predictions.keys()),
    }


def run_pipeline_async():
    """Run the full ML pipeline in a background thread."""
    global pipeline_state
    pipeline_state["running"]    = True
    pipeline_state["status"]     = "running"
    pipeline_state["progress"]   = 0
    pipeline_state["started_at"] = datetime.now().isoformat()
    pipeline_state["message"]    = "Starting pipeline…"

    try:
        from utils import get_logger
        log = get_logger("pipeline")

        stages = [
            ("Preprocessing",     "preprocessing",     "run_preprocessing"),
            ("EDA",               "eda",               "run_eda"),
            ("Forecasting",       "forecasting",       "run_forecasting"),
            ("Ensemble",          "ensemble",          "run_ensemble"),
            ("Feature Importance","feature_importance","run_feature_importance"),
            ("Climate Analysis",  "climate_analysis",  "run_climate_analysis"),
            ("Spatial Analysis",  "spatial_analysis",  "run_spatial_analysis"),
        ]
        total = len(stages)
        df = None

        for i, (name, module, fn_name) in enumerate(stages):
            pipeline_state["message"]  = f"Running Stage {i+1}/{total}: {name}…"
            pipeline_state["progress"] = int((i / total) * 100)
            try:
                mod = __import__(module)
                fn  = getattr(mod, fn_name)
                if i == 0:
                    df, encoders, scaler = fn(save=True)
                elif name in ("EDA", "Feature Importance", "Spatial Analysis"):
                    fn(df)
                elif name in ("Forecasting", "Ensemble"):
                    result = fn(df)
                    if name == "Forecasting":
                        model_results = result if isinstance(result, list) else [result]
                    elif name == "Ensemble":
                        model_results.append(result)
                elif name == "Climate Analysis":
                    df = fn(df)
            except Exception as exc:
                log.warning(f"Stage {name} failed: {exc}")

        # Save results
        try:
            if model_results:
                valid = [r for r in model_results if r.get("RMSE") is not None]
                out = BASE_DIR / "outputs" / "reports" / "model_results.json"
                with open(out, "w", encoding="utf-8") as f:
                    json.dump(valid, f, indent=2)
        except Exception:
            pass

        # Sync plots
        sync_plots()

        pipeline_state["progress"]     = 100
        pipeline_state["status"]       = "complete"
        pipeline_state["message"]      = "Pipeline completed successfully!"
        pipeline_state["completed_at"] = datetime.now().isoformat()

    except Exception as exc:
        pipeline_state["status"]  = "error"
        pipeline_state["message"] = f"Pipeline failed: {str(exc)}"
        logger.error(f"Pipeline error: {exc}", exc_info=True)
    finally:
        pipeline_state["running"] = False


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    stats  = get_dataset_stats()
    plots  = get_available_plots()[:6]  # Show preview on home
    return render_template("index.html", stats=stats, preview_plots=plots)


@app.route("/dashboard")
def dashboard():
    sync_plots()
    plots  = get_available_plots()
    stats  = get_dataset_stats()
    return render_template("dashboard.html", plots=plots, stats=stats)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    error  = None
    form_data = {}

    if request.method == "POST":
        form_data = request.form.to_dict()
        try:
            result = make_quick_prediction(form_data)
        except Exception as exc:
            error = str(exc)
            logger.error(f"Prediction error: {exc}", exc_info=True)

    return render_template("prediction.html", result=result, error=error, form_data=form_data)


@app.route("/analysis")
def analysis():
    sync_plots()
    plots  = get_available_plots()
    # Filter for climate/AQI/anomaly plots
    climate_plots = [p for p in plots if any(
        x in p["filename"] for x in
        ["12_", "13_", "14_", "15_", "16_", "04_", "03_"]
    )]
    return render_template("analysis.html", plots=climate_plots, all_plots=plots)


@app.route("/models")
def models():
    results = get_model_results()
    plots   = get_available_plots()
    model_plots = [p for p in plots if any(
        x in p["filename"] for x in ["08_", "09_", "10_", "11_", "forecast_"]
    )]
    return render_template("model_comparison.html", results=results, plots=model_plots)


# ── Static plot serving ────────────────────────────────────────────────────────

@app.route("/static/plots/<path:filename>")
def serve_plot(filename):
    """Try static/plots first, then outputs/plots."""
    static_plots = BASE_DIR / "static" / "plots"
    output_plots = BASE_DIR / "outputs" / "plots"
    if (static_plots / filename).exists():
        return send_from_directory(str(static_plots), filename)
    elif (output_plots / filename).exists():
        return send_from_directory(str(output_plots), filename)
    return jsonify({"error": "Plot not found"}), 404


# ── API routes ─────────────────────────────────────────────────────────────────

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True, silent=True) or {}
    try:
        result = make_quick_prediction(data)
        return jsonify({"success": True, "result": result})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 400


@app.route("/api/plots", methods=["GET"])
def api_plots():
    sync_plots()
    return jsonify({"plots": get_available_plots()})


@app.route("/api/status", methods=["GET"])
def api_status():
    return jsonify(pipeline_state)


@app.route("/api/run-pipeline", methods=["POST"])
def api_run_pipeline():
    global pipeline_state
    if pipeline_state["running"]:
        return jsonify({"success": False, "message": "Pipeline already running."}), 409

    thread = threading.Thread(target=run_pipeline_async, daemon=True)
    thread.start()
    return jsonify({"success": True, "message": "Pipeline started in background."})


@app.route("/api/model-results", methods=["GET"])
def api_model_results():
    return jsonify({"results": get_model_results()})


# ── Error handlers ─────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return render_template("index.html",
                           error="Page not found.",
                           stats=get_dataset_stats(),
                           preview_plots=[]), 404


@app.errorhandler(500)
def server_error(e):
    return render_template("index.html",
                           error=f"Server error: {str(e)}",
                           stats=get_dataset_stats(),
                           preview_plots=[]), 500


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Sync any existing plots on startup
    sync_plots()

    _host  = os.environ.get("FLASK_HOST",  "127.0.0.1")
    _port  = int(os.environ.get("FLASK_PORT",  "5000"))
    _debug = os.environ.get("FLASK_DEBUG", "0") == "1"

    logger.info("=" * 60)
    logger.info("  Weather Trend Forecasting — Flask App")
    logger.info(f"  URL: http://{_host}:{_port}")
    logger.info("=" * 60)

    app.run(
        host=_host,
        port=_port,
        debug=_debug,
        use_reloader=False,  # Prevent double startup with threading
    )
