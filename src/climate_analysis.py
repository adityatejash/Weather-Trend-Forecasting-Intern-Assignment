"""
climate_analysis.py — Long-term climate trend analysis.

Analyses:
  1. Rolling temperature trends (30-day, 90-day)
  2. Year-over-year comparison
  3. AQI vs weather correlation analysis
  4. Anomaly detection (Z-score + IQR based)
  5. Radar charts for multi-metric comparison
  6. Extreme weather event counts
"""

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from pathlib import Path

from utils import get_logger, timer, save_figure

logger = get_logger("climate_analysis")

BG, TC, AC = "#0f1117", "#e0e0e0", "#00d4ff"


def _ax_style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(BG)
    ax.set_title(title, color=TC, fontsize=12, pad=8)
    ax.set_xlabel(xlabel, color=TC)
    ax.set_ylabel(ylabel, color=TC)
    ax.tick_params(colors=TC)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")


# ─────────────────────────────────────────────
# 1. Rolling temperature trend
# ─────────────────────────────────────────────
@timer
def plot_rolling_temperature(df: pd.DataFrame) -> Path:
    if "datetime" not in df.columns or "temperature_celsius" not in df.columns:
        return None

    ts = (df[["datetime", "temperature_celsius"]].dropna()
          .set_index("datetime").resample("D").mean().dropna())
    ts["roll30"] = ts["temperature_celsius"].rolling(30).mean()
    ts["roll90"] = ts["temperature_celsius"].rolling(90).mean()

    fig, ax = plt.subplots(figsize=(16, 5), facecolor=BG)
    _ax_style(ax, "Global Temperature Trends — Rolling Averages", "Date", "°C")
    ax.plot(ts.index, ts["temperature_celsius"], color="#555", alpha=0.4, linewidth=0.7, label="Daily")
    ax.plot(ts.index, ts["roll30"],  color="#ff6b6b", linewidth=1.5, label="30-day MA")
    ax.plot(ts.index, ts["roll90"],  color=AC,        linewidth=2.0, label="90-day MA")
    ax.legend(facecolor="#1e1e2e", labelcolor=TC)
    plt.tight_layout()
    return save_figure(fig, "12_rolling_temperature.png")


# ─────────────────────────────────────────────
# 2. Year-over-year
# ─────────────────────────────────────────────
@timer
def plot_yoy_comparison(df: pd.DataFrame) -> Path:
    if "year" not in df.columns or "temperature_celsius" not in df.columns:
        return None

    yoy = df.groupby(["year", "month"])["temperature_celsius"].mean().reset_index()
    years = sorted(yoy["year"].unique())

    fig, ax = plt.subplots(figsize=(14, 6), facecolor=BG)
    _ax_style(ax, "Year-over-Year Temperature Comparison", "Month", "Avg Temp (°C)")
    cmap = plt.cm.plasma
    for i, yr in enumerate(years):
        sub = yoy[yoy["year"] == yr]
        ax.plot(sub["month"], sub["temperature_celsius"],
                color=cmap(i / max(len(years)-1, 1)),
                linewidth=2, label=str(yr), marker="o", markersize=4)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                         "Jul","Aug","Sep","Oct","Nov","Dec"], color=TC)
    ax.legend(title="Year", facecolor="#1e1e2e", labelcolor=TC, title_fontsize=9,
              loc="upper right", ncol=2)
    plt.tight_layout()
    return save_figure(fig, "13_yoy_comparison.png")


# ─────────────────────────────────────────────
# 3. AQI vs Weather
# ─────────────────────────────────────────────
@timer
def plot_aqi_vs_weather(df: pd.DataFrame) -> Path:
    aqi_col = "aqi_composite" if "aqi_composite" in df.columns else None
    weather_cols = [c for c in ["temperature_celsius", "humidity",
                                  "wind_kph", "precip_mm"] if c in df.columns]
    if not aqi_col or not weather_cols:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor=BG)
    fig.suptitle("AQI vs Weather Metrics", color=TC, fontsize=16)
    colors = [AC, "#ff6b6b", "#ffd93d", "#6bcb77"]

    for ax, col, color in zip(axes.flatten(), weather_cols, colors):
        _ax_style(ax, f"AQI vs {col.replace('_', ' ').title()}", col, "AQI Composite")
        sample = df[[col, aqi_col]].dropna().sample(min(3000, len(df)), random_state=42)
        ax.scatter(sample[col], sample[aqi_col], alpha=0.3, color=color, s=8)
        # Trend line
        m, b, r, _, _ = stats.linregress(sample[col], sample[aqi_col])
        x_range = np.linspace(sample[col].min(), sample[col].max(), 100)
        ax.plot(x_range, m*x_range + b, color="white", linewidth=1.5)
        ax.text(0.95, 0.05, f"r={r:.2f}", transform=ax.transAxes,
                ha="right", color=TC, fontsize=10)

    plt.tight_layout()
    return save_figure(fig, "14_aqi_vs_weather.png")


# ─────────────────────────────────────────────
# 4. Anomaly detection
# ─────────────────────────────────────────────
@timer
def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Flag anomalies in temperature using Z-score (|z| > 3)."""
    if "temperature_celsius" not in df.columns:
        return df

    df = df.copy()
    z = np.abs(stats.zscore(df["temperature_celsius"].fillna(df["temperature_celsius"].median())))
    df["is_anomaly"] = z > 3

    n_anom = df["is_anomaly"].sum()
    logger.info(f"Anomalies detected: {n_anom} ({100*n_anom/len(df):.2f}%)")

    # Plot
    fig, ax = plt.subplots(figsize=(16, 5), facecolor=BG)
    _ax_style(ax, "Temperature Anomaly Detection (|Z-score| > 3)", "Index", "°C")
    ax.scatter(df.index, df["temperature_celsius"], s=2, color="#555", alpha=0.5, label="Normal")
    anomalies = df[df["is_anomaly"]]
    ax.scatter(anomalies.index, anomalies["temperature_celsius"],
               s=20, color="#ff4757", alpha=0.8, label="Anomaly")
    ax.legend(facecolor="#1e1e2e", labelcolor=TC)
    plt.tight_layout()
    save_figure(fig, "15_anomaly_detection.png")
    return df


# ─────────────────────────────────────────────
# 5. Radar chart
# ─────────────────────────────────────────────
@timer
def plot_radar_chart(df: pd.DataFrame) -> Path:
    if "continent" not in df.columns:
        return None

    metrics = [c for c in ["temperature_celsius", "humidity", "wind_kph",
                             "uv_index", "precip_mm"] if c in df.columns]
    if len(metrics) < 3:
        return None

    cont_stats = df.groupby("continent")[metrics].mean().dropna(how="all")
    # Normalise 0-1
    norm = (cont_stats - cont_stats.min()) / (cont_stats.max() - cont_stats.min() + 1e-9)

    continents = norm.index.tolist()
    N = len(metrics)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(10, 8), facecolor=BG)
    ax  = fig.add_subplot(111, polar=True, facecolor=BG)
    ax.set_facecolor(BG)

    colors = plt.cm.tab10(np.linspace(0, 1, len(continents)))
    for cont, color in zip(continents, colors):
        vals  = norm.loc[cont, metrics].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, color=color, linewidth=2, label=cont)
        ax.fill(angles, vals, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace("_celsius","").replace("_"," ").title()
                        for m in metrics], color=TC, fontsize=10)
    ax.set_yticklabels([])
    ax.set_title("Multi-metric Radar Chart by Continent", color=TC, fontsize=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1),
              facecolor="#1e1e2e", labelcolor=TC)
    ax.tick_params(colors=TC)
    ax.spines["polar"].set_color("#333")
    plt.tight_layout()
    return save_figure(fig, "16_radar_chart.png")


@timer
def run_climate_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Run all climate analysis routines."""
    logger.info("Starting climate analysis …")
    plot_rolling_temperature(df)
    plot_yoy_comparison(df)
    plot_aqi_vs_weather(df)
    df = detect_anomalies(df)
    plot_radar_chart(df)
    logger.info("Climate analysis complete.")
    return df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from preprocessing import run_preprocessing
    df, _, _ = run_preprocessing(save=False)
    run_climate_analysis(df)
