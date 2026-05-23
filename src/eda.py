"""
eda.py — Exploratory Data Analysis module.

Generates and saves:
  1. Correlation heatmap
  2. Temperature distribution & trends over time
  3. Precipitation analysis
  4. AQI analysis
  5. Seasonal analysis
  6. Country-wise comparison (top N)
  7. Continent-wise comparison
  8. Wind rose / direction charts
  9. Humidity & pressure distributions
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

from utils import get_logger, timer, save_figure, PLOTS_DIR
from preprocessing import NUMERIC_COLS

logger = get_logger("eda")

# ── Colour palette ────────────────────────────────────────────────────────────
PALETTE    = "coolwarm"
BG_COLOR   = "#0f1117"
TEXT_COLOR = "#e0e0e0"
ACCENT     = "#00d4ff"


def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_title(title, color=TEXT_COLOR, fontsize=13, pad=10)
    ax.set_xlabel(xlabel, color=TEXT_COLOR)
    ax.set_ylabel(ylabel, color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")


# ─────────────────────────────────────────────
# 1. Correlation heatmap
# ─────────────────────────────────────────────
@timer
def plot_correlation_heatmap(df: pd.DataFrame) -> Path:
    num_cols = [c for c in NUMERIC_COLS if c in df.columns]
    corr     = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(16, 12), facecolor=BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        annot_kws={"size": 7}, linewidths=0.5, linecolor="#222",
        ax=ax, cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature Correlation Matrix", color=TEXT_COLOR, fontsize=16, pad=14)
    ax.tick_params(colors=TEXT_COLOR)
    plt.tight_layout()
    return save_figure(fig, "01_correlation_heatmap.png")


# ─────────────────────────────────────────────
# 2. Temperature trends
# ─────────────────────────────────────────────
@timer
def plot_temperature_trends(df: pd.DataFrame) -> Path:
    if "temperature_celsius" not in df.columns or "datetime" not in df.columns:
        logger.warning("Temperature or datetime column missing – skipping trend plot.")
        return None

    monthly = (
        df.set_index("datetime")["temperature_celsius"]
        .resample("ME").mean()
        .dropna()
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor=BG_COLOR)
    fig.suptitle("Temperature Analysis", color=TEXT_COLOR, fontsize=18, y=1.01)

    # Monthly average
    ax = axes[0, 0]
    ax.set_facecolor(BG_COLOR)
    ax.plot(monthly.index, monthly.values, color=ACCENT, linewidth=2)
    ax.fill_between(monthly.index, monthly.values, alpha=0.2, color=ACCENT)
    _style_ax(ax, "Monthly Avg Temperature", "Date", "°C")

    # Distribution
    ax = axes[0, 1]
    ax.set_facecolor(BG_COLOR)
    ax.hist(df["temperature_celsius"].dropna(), bins=60, color=ACCENT, edgecolor="#333", alpha=0.85)
    _style_ax(ax, "Temperature Distribution", "°C", "Count")

    # By season
    ax = axes[1, 0]
    ax.set_facecolor(BG_COLOR)
    if "season" in df.columns:
        season_means = df.groupby("season")["temperature_celsius"].mean().sort_values()
        bars = ax.bar(season_means.index, season_means.values,
                      color=[ACCENT, "#ff6b6b", "#ffd93d", "#6bcb77"])
        _style_ax(ax, "Avg Temperature by Season", "Season", "°C")

    # By continent
    ax = axes[1, 1]
    ax.set_facecolor(BG_COLOR)
    if "continent" in df.columns:
        cont_means = df.groupby("continent")["temperature_celsius"].mean().sort_values()
        ax.barh(cont_means.index, cont_means.values,
                color=plt.cm.coolwarm(np.linspace(0.1, 0.9, len(cont_means))))
        _style_ax(ax, "Avg Temperature by Continent", "°C", "Continent")

    plt.tight_layout()
    return save_figure(fig, "02_temperature_trends.png")


# ─────────────────────────────────────────────
# 3. Precipitation analysis
# ─────────────────────────────────────────────
@timer
def plot_precipitation_analysis(df: pd.DataFrame) -> Path:
    if "precip_mm" not in df.columns:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor=BG_COLOR)
    fig.suptitle("Precipitation Analysis", color=TEXT_COLOR, fontsize=16)

    # Distribution (log scale)
    ax = axes[0]
    ax.set_facecolor(BG_COLOR)
    data = df["precip_mm"].dropna()
    data_pos = data[data > 0]
    ax.hist(np.log1p(data_pos), bins=50, color="#6bcb77", edgecolor="#333", alpha=0.85)
    _style_ax(ax, "Precipitation (log scale)", "log(precip_mm+1)", "Count")

    # Monthly average
    ax = axes[1]
    ax.set_facecolor(BG_COLOR)
    if "month" in df.columns:
        monthly_p = df.groupby("month")["precip_mm"].mean()
        ax.bar(monthly_p.index, monthly_p.values, color="#6bcb77", edgecolor="#333")
        ax.set_xticks(range(1, 13))
        _style_ax(ax, "Monthly Avg Precipitation", "Month", "mm")

    # Top 10 rainy countries
    ax = axes[2]
    ax.set_facecolor(BG_COLOR)
    if "country" in df.columns:
        top = df.groupby("country")["precip_mm"].mean().nlargest(10)
        ax.barh(top.index, top.values, color="#6bcb77")
        _style_ax(ax, "Top 10 Rainy Countries", "Avg mm", "")

    plt.tight_layout()
    return save_figure(fig, "03_precipitation_analysis.png")


# ─────────────────────────────────────────────
# 4. AQI analysis
# ─────────────────────────────────────────────
@timer
def plot_aqi_analysis(df: pd.DataFrame) -> Path:
    aqi_cols = [c for c in ["air_quality_PM2.5", "air_quality_PM10",
                             "air_quality_CO", "air_quality_NO2",
                             "air_quality_O3", "air_quality_SO2"] if c in df.columns]
    if not aqi_cols:
        return None

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor=BG_COLOR)
    axes_flat = axes.flatten()
    colors    = [ACCENT, "#ff6b6b", "#ffd93d", "#6bcb77", "#c77dff", "#ff9f43"]

    for i, col in enumerate(aqi_cols[:6]):
        ax = axes_flat[i]
        ax.set_facecolor(BG_COLOR)
        data = df[col].dropna()
        ax.hist(data, bins=50, color=colors[i], edgecolor="#222", alpha=0.85)
        label = col.replace("air_quality_", "")
        _style_ax(ax, f"{label} Distribution", label, "Count")

    for j in range(len(aqi_cols), 6):
        axes_flat[j].set_visible(False)

    fig.suptitle("Air Quality Index (AQI) Analysis", color=TEXT_COLOR, fontsize=16)
    plt.tight_layout()
    return save_figure(fig, "04_aqi_analysis.png")


# ─────────────────────────────────────────────
# 5. Seasonal analysis
# ─────────────────────────────────────────────
@timer
def plot_seasonal_analysis(df: pd.DataFrame) -> Path:
    if "season" not in df.columns or "temperature_celsius" not in df.columns:
        return None

    metrics = [c for c in ["temperature_celsius", "humidity",
                             "precip_mm", "wind_kph"] if c in df.columns]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor=BG_COLOR)
    fig.suptitle("Seasonal Analysis", color=TEXT_COLOR, fontsize=16)
    colors = ["#00d4ff", "#ffd93d", "#ff6b6b", "#6bcb77"]

    for ax, metric in zip(axes.flatten(), metrics):
        ax.set_facecolor(BG_COLOR)
        seasons = ["Spring", "Summer", "Autumn", "Winter"]
        vals    = [df[df["season"] == s][metric].mean() for s in seasons]
        bars    = ax.bar(seasons, vals, color=colors)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", va="bottom", color=TEXT_COLOR, fontsize=9)
        label = metric.replace("_celsius","°C").replace("_"," ").title()
        _style_ax(ax, f"{label} by Season", "Season", label)

    plt.tight_layout()
    return save_figure(fig, "05_seasonal_analysis.png")


# ─────────────────────────────────────────────
# 6. Country-wise comparison
# ─────────────────────────────────────────────
@timer
def plot_country_comparison(df: pd.DataFrame, top_n: int = 15) -> Path:
    if "country" not in df.columns or "temperature_celsius" not in df.columns:
        return None

    country_stats = df.groupby("country").agg(
        avg_temp=("temperature_celsius", "mean"),
        avg_humidity=("humidity", "mean"),
        avg_precip=("precip_mm", "mean"),
    ).sort_values("avg_temp", ascending=False).head(top_n)

    fig, axes = plt.subplots(1, 3, figsize=(20, 7), facecolor=BG_COLOR)
    fig.suptitle(f"Top {top_n} Countries — Weather Comparison", color=TEXT_COLOR, fontsize=16)

    for ax, (col, label, color) in zip(axes, [
        ("avg_temp",     "Avg Temperature (°C)", "#ff6b6b"),
        ("avg_humidity", "Avg Humidity (%)",     ACCENT),
        ("avg_precip",   "Avg Precipitation (mm)", "#6bcb77"),
    ]):
        ax.set_facecolor(BG_COLOR)
        data = country_stats[col].sort_values()
        ax.barh(data.index, data.values, color=color, edgecolor="#222")
        _style_ax(ax, label, label, "Country")

    plt.tight_layout()
    return save_figure(fig, "06_country_comparison.png")


# ─────────────────────────────────────────────
# 7. Continent-wise comparison
# ─────────────────────────────────────────────
@timer
def plot_continent_comparison(df: pd.DataFrame) -> Path:
    if "continent" not in df.columns:
        return None

    metrics = [c for c in ["temperature_celsius", "humidity",
                             "precip_mm", "wind_kph", "uv_index"] if c in df.columns]
    cont_stats = df.groupby("continent")[metrics].mean().dropna(how="all")

    fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 6), facecolor=BG_COLOR)
    fig.suptitle("Continent-wise Weather Comparison", color=TEXT_COLOR, fontsize=16)

    cmap = plt.cm.plasma
    for ax, metric in zip(axes, metrics):
        ax.set_facecolor(BG_COLOR)
        data = cont_stats[metric].sort_values(ascending=False)
        colors_bar = [cmap(i/len(data)) for i in range(len(data))]
        ax.bar(data.index, data.values, color=colors_bar)
        ax.set_xticklabels(data.index, rotation=45, ha="right", color=TEXT_COLOR, fontsize=8)
        label = metric.replace("_celsius","").replace("_"," ").title()
        _style_ax(ax, label, "", label)

    plt.tight_layout()
    return save_figure(fig, "07_continent_comparison.png")


# ─────────────────────────────────────────────
# Master EDA runner
# ─────────────────────────────────────────────
@timer
def run_eda(df: pd.DataFrame) -> list[Path]:
    """Run all EDA plots and return list of saved paths."""
    logger.info("Starting EDA …")
    paths = []
    for fn in [
        plot_correlation_heatmap,
        plot_temperature_trends,
        plot_precipitation_analysis,
        plot_aqi_analysis,
        plot_seasonal_analysis,
        plot_country_comparison,
        plot_continent_comparison,
    ]:
        try:
            p = fn(df)
            if p:
                paths.append(p)
        except Exception as exc:
            logger.warning(f"{fn.__name__} failed: {exc}")
    logger.info(f"EDA complete — {len(paths)} plots saved to {PLOTS_DIR}")
    return paths


if __name__ == "__main__":
    from preprocessing import run_preprocessing
    df, _, _ = run_preprocessing(save=False)
    run_eda(df)
