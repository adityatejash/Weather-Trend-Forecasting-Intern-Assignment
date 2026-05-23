"""
main.py — Pipeline orchestrator (pure Python, no notebooks).

Usage:
    python src/main.py [--skip-eda] [--skip-models] [--skip-ensemble]
                       [--skip-shap] [--skip-climate] [--skip-spatial]

Stages:
  1. Preprocessing
  2. EDA
  3. Forecasting (LR / RF / XGBoost / Prophet / LSTM)
  4. Stacking Ensemble
  5. SHAP Feature Importance
  6. Climate Analysis
  7. Spatial Analysis
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

# Ensure src/ is on the path when invoked from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import get_logger, timestamp_str, OUTPUT_DIR

logger = get_logger("main")


def parse_args():
    p = argparse.ArgumentParser(description="Weather Trend Forecasting — Full Pipeline")
    p.add_argument("--skip-eda",      action="store_true", help="Skip EDA plots")
    p.add_argument("--skip-models",   action="store_true", help="Skip model training")
    p.add_argument("--skip-ensemble", action="store_true", help="Skip ensemble")
    p.add_argument("--skip-shap",     action="store_true", help="Skip SHAP analysis")
    p.add_argument("--skip-climate",  action="store_true", help="Skip climate analysis")
    p.add_argument("--skip-spatial",  action="store_true", help="Skip spatial analysis")
    return p.parse_args()


def banner(text: str):
    width = 60
    print("\n" + "═" * width)
    print(f"  {text}")
    print("═" * width)


def main():
    args = parse_args()
    t_start = time.perf_counter()

    print("""
╔══════════════════════════════════════════════════════════╗
║   WEATHER TREND FORECASTING — FULL PIPELINE              ║
║   PM Accelerator Data Science Internship Project         ║
╚══════════════════════════════════════════════════════════╝
""")

    # ── Stage 1: Preprocessing ───────────────────────────────
    banner("Stage 1 / 7 — Data Preprocessing")
    from preprocessing import run_preprocessing
    df, encoders, scaler = run_preprocessing(save=True)

    # ── Stage 2: EDA ─────────────────────────────────────────
    if not args.skip_eda:
        banner("Stage 2 / 7 — Exploratory Data Analysis")
        from eda import run_eda
        run_eda(df)

    # ── Stage 3: Forecasting ─────────────────────────────────
    if not args.skip_models:
        banner("Stage 3 / 7 — Model Training (LR / RF / XGB / Prophet / LSTM)")
        from forecasting import run_forecasting
        model_results = run_forecasting(df)
    else:
        model_results = []

    # ── Stage 4: Ensemble ────────────────────────────────────
    if not args.skip_ensemble:
        banner("Stage 4 / 7 — Stacking Ensemble")
        from ensemble import run_ensemble
        ensemble_result = run_ensemble(df)
        model_results.append(ensemble_result)

    # ── Stage 5: SHAP / Feature Importance ───────────────────
    if not args.skip_shap:
        banner("Stage 5 / 7 — SHAP Feature Importance")
        from feature_importance import run_feature_importance
        run_feature_importance(df)

    # ── Stage 6: Climate Analysis ────────────────────────────
    if not args.skip_climate:
        banner("Stage 6 / 7 — Climate Analysis & Anomaly Detection")
        from climate_analysis import run_climate_analysis
        df = run_climate_analysis(df)

    # ── Stage 7: Spatial Analysis ────────────────────────────
    if not args.skip_spatial:
        banner("Stage 7 / 7 — Spatial / Geographical Analysis")
        from spatial_analysis import run_spatial_analysis
        run_spatial_analysis(df)

    # ── Summary ──────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    print(f"""
╔══════════════════════════════════════════════════════════╗
║  ✅ Pipeline complete in {elapsed:6.1f}s                      ║
║  📁 Outputs saved to: outputs/                           ║
╚══════════════════════════════════════════════════════════╝
""")

    if model_results:
        import pandas as pd
        import json
        valid = [r for r in model_results if r.get("RMSE") is not None]
        if valid:
            print("\n📊 Model Results Summary:")
            print(pd.DataFrame(valid).to_string(index=False))
            out = Path(__file__).resolve().parent.parent / "outputs" / "reports" / "model_results.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w") as f:
                json.dump(valid, f, indent=2)
            print(f"\n💾 Results saved → {out}")


if __name__ == "__main__":
    main()
