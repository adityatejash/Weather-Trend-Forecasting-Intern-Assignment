"""
utils.py — Shared utility helpers for the Weather Trend Forecasting project.

Provides logging setup, path resolution, timing decorators, metric formatting,
and other reusable primitives used across all modules.
"""

import os
import time
import logging
import functools
from pathlib import Path
from datetime import datetime

import numpy as np

# ─────────────────────────────────────────────
# Project Paths
# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
OUTPUT_DIR   = PROJECT_ROOT / "outputs"
PLOTS_DIR    = OUTPUT_DIR / "plots"
MODELS_DIR   = OUTPUT_DIR / "models"
REPORTS_DIR  = OUTPUT_DIR / "reports"

for _d in [DATA_DIR, PLOTS_DIR, MODELS_DIR, REPORTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

DATA_FILE = DATA_DIR / "GlobalWeatherRepository.csv"


# ─────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────
def get_logger(name: str = "weather_forecast", level: int = logging.INFO) -> logging.Logger:
    """Return a consistently formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


logger = get_logger()


# ─────────────────────────────────────────────
# Decorators
# ─────────────────────────────────────────────
def timer(func):
    """Decorator – prints execution time of a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        logger.info(f"⏱  {func.__name__} completed in {elapsed:.2f}s")
        return result
    return wrapper


# ─────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Mean Absolute Percentage Error (%)."""
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of Determination (R²)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-10))


def evaluate_model(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return a dict of all four metrics for a model."""
    metrics = {
        "Model": name,
        "RMSE":  round(rmse(y_true, y_pred), 4),
        "MAE":   round(mae(y_true, y_pred), 4),
        "MAPE":  round(mape(y_true, y_pred), 4),
        "R2":    round(r2(y_true, y_pred), 4),
    }
    logger.info(
        f"📊 {name:20s} | RMSE={metrics['RMSE']:.4f} | "
        f"MAE={metrics['MAE']:.4f} | MAPE={metrics['MAPE']:.2f}% | "
        f"R²={metrics['R2']:.4f}"
    )
    return metrics


# ─────────────────────────────────────────────
# File helpers
# ─────────────────────────────────────────────
def save_figure(fig, filename: str, subdir: str = "plots", dpi: int = 150) -> Path:
    """Save a matplotlib figure to outputs/<subdir>/."""
    import matplotlib.pyplot as plt
    dest = OUTPUT_DIR / subdir / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"💾 Figure saved → {dest}")
    return dest


def timestamp_str() -> str:
    """Return a compact UTC timestamp string, e.g. '20240519_1435'."""
    return datetime.utcnow().strftime("%Y%m%d_%H%M")


# ─────────────────────────────────────────────
# Season helper
# ─────────────────────────────────────────────
SEASONS = {12: "Winter", 1: "Winter", 2: "Winter",
           3: "Spring", 4: "Spring", 5: "Spring",
           6: "Summer", 7: "Summer", 8: "Summer",
           9: "Autumn", 10: "Autumn", 11: "Autumn"}


def month_to_season(month: int) -> str:
    return SEASONS.get(month, "Unknown")


# ─────────────────────────────────────────────
# Continent mapping
# ─────────────────────────────────────────────
CONTINENT_MAP = {
    "Afghanistan": "Asia", "Albania": "Europe", "Algeria": "Africa",
    "Angola": "Africa", "Argentina": "South America", "Armenia": "Asia",
    "Australia": "Oceania", "Austria": "Europe", "Azerbaijan": "Asia",
    "Bangladesh": "Asia", "Belgium": "Europe", "Brazil": "South America",
    "Bulgaria": "Europe", "Cambodia": "Asia", "Cameroon": "Africa",
    "Canada": "North America", "Chile": "South America", "China": "Asia",
    "Colombia": "South America", "Croatia": "Europe", "Cuba": "North America",
    "Czech Republic": "Europe", "Denmark": "Europe", "Ecuador": "South America",
    "Egypt": "Africa", "Ethiopia": "Africa", "Finland": "Europe",
    "France": "Europe", "Germany": "Europe", "Ghana": "Africa",
    "Greece": "Europe", "Guatemala": "North America", "Hungary": "Europe",
    "India": "Asia", "Indonesia": "Asia", "Iran": "Asia", "Iraq": "Asia",
    "Ireland": "Europe", "Israel": "Asia", "Italy": "Europe", "Japan": "Asia",
    "Jordan": "Asia", "Kazakhstan": "Asia", "Kenya": "Africa",
    "Kuwait": "Asia", "Malaysia": "Asia", "Mexico": "North America",
    "Morocco": "Africa", "Mozambique": "Africa", "Myanmar": "Asia",
    "Nepal": "Asia", "Netherlands": "Europe", "New Zealand": "Oceania",
    "Nigeria": "Africa", "Norway": "Europe", "Pakistan": "Asia",
    "Peru": "South America", "Philippines": "Asia", "Poland": "Europe",
    "Portugal": "Europe", "Romania": "Europe", "Russia": "Europe",
    "Saudi Arabia": "Asia", "Senegal": "Africa", "Serbia": "Europe",
    "Singapore": "Asia", "South Africa": "Africa", "South Korea": "Asia",
    "Spain": "Europe", "Sri Lanka": "Asia", "Sudan": "Africa",
    "Sweden": "Europe", "Switzerland": "Europe", "Syria": "Asia",
    "Tanzania": "Africa", "Thailand": "Asia", "Tunisia": "Africa",
    "Turkey": "Asia", "Uganda": "Africa", "Ukraine": "Europe",
    "United Arab Emirates": "Asia", "United Kingdom": "Europe",
    "United States": "North America", "Uzbekistan": "Asia",
    "Venezuela": "South America", "Vietnam": "Asia", "Yemen": "Asia",
    "Zambia": "Africa", "Zimbabwe": "Africa",
}


def get_continent(country: str) -> str:
    return CONTINENT_MAP.get(country, "Unknown")
