"""
Weather_Trend_Forecasting.py — Top-level entry point (pure Python).

Runs the complete pipeline:
  1. Preprocessing
  2. EDA
  3. Forecasting (LR / RF / XGBoost / Prophet / LSTM)
  4. Stacking Ensemble
  5. SHAP Feature Importance
  6. Climate Analysis
  7. Spatial Analysis

Usage
-----
  # Full run (requires data/GlobalWeatherRepository.csv)
  python Weather_Trend_Forecasting.py

  # Generate synthetic demo data and run
  python Weather_Trend_Forecasting.py --demo

  # Skip slow stages
  python Weather_Trend_Forecasting.py --demo --skip-models --skip-spatial
"""

import sys
import warnings
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings("ignore")

import sys
import argparse
import time
from pathlib import Path

# ── Resolve project root and put src/ on the path ────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR      = PROJECT_ROOT / "src"
DATA_DIR     = PROJECT_ROOT / "data"
CSV_PATH     = DATA_DIR / "GlobalWeatherRepository.csv"

sys.path.insert(0, str(SRC_DIR))

# Ensure output directories exist
for _d in ["outputs/plots", "outputs/models", "outputs/reports", "data"]:
    (PROJECT_ROOT / _d).mkdir(parents=True, exist_ok=True)


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Weather Trend Forecasting — PM Accelerator Internship"
    )
    p.add_argument("--demo",          action="store_true",
                   help="Generate synthetic data when CSV is missing")
    p.add_argument("--skip-eda",      action="store_true", help="Skip EDA plots")
    p.add_argument("--skip-models",   action="store_true", help="Skip model training")
    p.add_argument("--skip-ensemble", action="store_true", help="Skip ensemble")
    p.add_argument("--skip-shap",     action="store_true", help="Skip SHAP analysis")
    p.add_argument("--skip-climate",  action="store_true", help="Skip climate analysis")
    p.add_argument("--skip-spatial",  action="store_true", help="Skip spatial analysis")
    return p.parse_args()


# ── Synthetic demo data ───────────────────────────────────────────────────────
def generate_demo_data() -> None:
    """Create a realistic synthetic GlobalWeatherRepository.csv."""
    import numpy as np
    import pandas as pd

    print("⚠️  GlobalWeatherRepository.csv not found.")
    print("    Generating synthetic demo dataset (50,000 rows) …\n")

    np.random.seed(42)
    n = 50_000

    countries = [
        "United States", "India", "China", "Brazil", "United Kingdom",
        "Germany", "Japan", "Australia", "Canada", "France",
        "Russia", "South Africa", "Mexico", "Indonesia", "Egypt",
        "Nigeria", "Argentina", "Pakistan", "Bangladesh", "Saudi Arabia",
        "Turkey", "Iran", "Spain", "Italy", "Thailand",
        "Vietnam", "Philippines", "Kenya", "Ethiopia", "Ukraine",
    ]
    conditions = [
        "Sunny", "Partly cloudy", "Cloudy", "Overcast",
        "Light rain", "Heavy rain", "Thunderstorm", "Fog", "Snow", "Mist",
    ]
    wind_dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

    base_temp = {
        "United States": 15, "India": 28, "China": 18, "Brazil": 27,
        "United Kingdom": 11, "Germany": 10, "Japan": 16, "Australia": 22,
        "Canada": 5,  "France": 12, "Russia": 2,  "South Africa": 20,
        "Mexico": 24, "Indonesia": 27, "Egypt": 26, "Nigeria": 28,
        "Argentina": 18, "Pakistan": 25, "Bangladesh": 27, "Saudi Arabia": 32,
        "Turkey": 17, "Iran": 22, "Spain": 19, "Italy": 17, "Thailand": 29,
        "Vietnam": 27, "Philippines": 28, "Kenya": 24, "Ethiopia": 22, "Ukraine": 10,
    }

    dates       = pd.date_range("2022-01-01", "2024-12-31", periods=n)
    country_arr = np.random.choice(countries, n)
    month_arr   = dates.month.to_numpy()

    temp = np.array([
        base_temp.get(c, 20)
        + 10 * np.sin(2 * np.pi * m / 12)
        + np.random.normal(0, 3)
        for c, m in zip(country_arr, month_arr)
    ])

    df = pd.DataFrame({
        "last_updated":          dates.strftime("%Y-%m-%d %H:%M"),
        "country":               country_arr,
        "location_name":         np.random.choice(["Capital", "Metro A", "Metro B", "Port City"], n),
        "latitude":              np.random.uniform(-60, 70, n).round(4),
        "longitude":             np.random.uniform(-180, 180, n).round(4),
        "temperature_celsius":   temp.round(2),
        "feels_like_celsius":    (temp - np.random.uniform(0, 3, n)).round(2),
        "dewpoint_celsius":      (temp - np.random.uniform(5, 15, n)).round(2),
        "humidity":              np.random.uniform(20, 100, n).round(1),
        "wind_kph":              np.abs(np.random.normal(20, 12, n)).round(1),
        "wind_degree":           np.random.randint(0, 360, n),
        "wind_direction":        np.random.choice(wind_dirs, n),
        "pressure_mb":           np.random.normal(1013, 10, n).round(1),
        "precip_mm":             np.abs(np.random.exponential(2, n)).round(2),
        "visibility_km":         np.random.uniform(1, 30, n).round(1),
        "uv_index":              np.random.randint(0, 12, n),
        "gust_kph":              np.abs(np.random.normal(28, 14, n)).round(1),
        "cloud":                 np.random.randint(0, 100, n),
        "condition_text":        np.random.choice(conditions, n),
        "air_quality_PM2.5":     np.abs(np.random.exponential(20, n)).round(2),
        "air_quality_PM10":      np.abs(np.random.exponential(35, n)).round(2),
        "air_quality_CO":        np.abs(np.random.exponential(300, n)).round(2),
        "air_quality_NO2":       np.abs(np.random.exponential(25, n)).round(2),
        "air_quality_O3":        np.abs(np.random.exponential(60, n)).round(2),
        "air_quality_SO2":       np.abs(np.random.exponential(10, n)).round(2),
    })

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV_PATH, index=False)
    print(f"✅ Demo data saved → {CSV_PATH}  ({n:,} rows × {df.shape[1]} columns)\n")


# ── Banner helpers ────────────────────────────────────────────────────────────
def _banner(text: str) -> None:
    width = 62
    print(f"\n{'═' * width}")
    print(f"  {text}")
    print(f"{'═' * width}")


# ── Main pipeline ─────────────────────────────────────────────────────────────
def main() -> None:
    args    = parse_args()
    t_start = time.perf_counter()

    print("""
╔══════════════════════════════════════════════════════════════╗
║   🌦️  WEATHER TREND FORECASTING                              ║
║   PM Accelerator Data Science Internship Project             ║
╚══════════════════════════════════════════════════════════════╝
""")

    # ── Dataset check ─────────────────────────────────────────
    if not CSV_PATH.exists():
        if args.demo:
            generate_demo_data()
        else:
            print(
                "❌  Dataset not found at data/GlobalWeatherRepository.csv\n\n"
                "Options:\n"
                "  1. Download from Kaggle and place it in data/\n"
                "     https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository\n\n"
                "  2. Run with --demo to use synthetic data:\n"
                "     python Weather_Trend_Forecasting.py --demo\n"
            )
            sys.exit(1)

    # ── Stage 1: Preprocessing ────────────────────────────────
    _banner("Stage 1 / 7 — Data Preprocessing")
    from preprocessing import run_preprocessing
    df, encoders, scaler = run_preprocessing(save=True)

    # ── Stage 2: EDA ──────────────────────────────────────────
    if not args.skip_eda:
        _banner("Stage 2 / 7 — Exploratory Data Analysis")
        from eda import run_eda
        run_eda(df)
    else:
        print("\n⏩ Skipping EDA (--skip-eda)")

    # ── Stage 3: Forecasting ──────────────────────────────────
    model_results = []
    if not args.skip_models:
        _banner("Stage 3 / 7 — Model Training  (LR · RF · XGB · Prophet · LSTM)")
        from forecasting import run_forecasting
        model_results = run_forecasting(df)
    else:
        print("\n⏩ Skipping model training (--skip-models)")

    # ── Stage 4: Ensemble ─────────────────────────────────────
    if not args.skip_ensemble:
        _banner("Stage 4 / 7 — Stacking Ensemble")
        from ensemble import run_ensemble
        result = run_ensemble(df)
        model_results.append(result)
    else:
        print("\n⏩ Skipping ensemble (--skip-ensemble)")

    # ── Stage 5: SHAP ─────────────────────────────────────────
    if not args.skip_shap:
        _banner("Stage 5 / 7 — SHAP Feature Importance")
        from feature_importance import run_feature_importance
        run_feature_importance(df)
    else:
        print("\n⏩ Skipping SHAP (--skip-shap)")

    # ── Stage 6: Climate Analysis ─────────────────────────────
    if not args.skip_climate:
        _banner("Stage 6 / 7 — Climate Analysis & Anomaly Detection")
        from climate_analysis import run_climate_analysis
        df = run_climate_analysis(df)
    else:
        print("\n⏩ Skipping climate analysis (--skip-climate)")

    # ── Stage 7: Spatial Analysis ─────────────────────────────
    if not args.skip_spatial:
        _banner("Stage 7 / 7 — Spatial / Geographical Analysis")
        from spatial_analysis import run_spatial_analysis
        run_spatial_analysis(df)
    else:
        print("\n⏩ Skipping spatial analysis (--skip-spatial)")

    # ── Summary ───────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  ✅  Pipeline complete in {elapsed:6.1f}s                        ║
║  📁  Outputs  → outputs/plots/  outputs/models/              ║
║  🌐  Flask app → python app.py  →  http://127.0.0.1:5000     ║
╚══════════════════════════════════════════════════════════════╝
""")

    if model_results:
        import pandas as pd
        valid = [r for r in model_results if r.get("RMSE") is not None]
        if valid:
            print("📊 Model Results:")
            print(pd.DataFrame(valid).to_string(index=False))

            # Persist results for the dashboard
            import json
            out = PROJECT_ROOT / "outputs" / "reports" / "model_results.json"
            with open(out, "w") as f:
                json.dump(valid, f, indent=2)
            print(f"\n💾 Results saved → {out}")


if __name__ == "__main__":
    main()
