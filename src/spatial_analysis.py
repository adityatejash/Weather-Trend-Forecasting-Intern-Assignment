"""
spatial_analysis.py — Geographical / spatial weather analysis.

Outputs:
  1. Choropleth map of avg temperature (Plotly)
  2. Choropleth map of avg AQI
  3. Scatter geo for wind speed
  4. Country heatmap (seaborn)
"""

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from utils import get_logger, timer, save_figure, PLOTS_DIR

logger = get_logger("spatial_analysis")


@timer
def plot_choropleth_temperature(df: pd.DataFrame) -> Path:
    if "country" not in df.columns or "temperature_celsius" not in df.columns:
        return None

    country_temp = df.groupby("country")["temperature_celsius"].mean().reset_index()
    country_temp.columns = ["country", "avg_temp"]

    fig = px.choropleth(
        country_temp, locations="country", locationmode="country names",
        color="avg_temp", hover_name="country",
        color_continuous_scale="RdYlBu_r",
        title="🌍 Average Temperature by Country (°C)",
        labels={"avg_temp": "Avg Temp (°C)"},
        template="plotly_dark",
    )
    fig.update_layout(
        paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
        font_color="#e0e0e0", title_font_size=18,
        geo=dict(bgcolor="#0f1117", lakecolor="#0f1117",
                 showland=True, landcolor="#1e1e2e",
                 showocean=True, oceancolor="#0d0d1a"),
        coloraxis_colorbar=dict(title="°C", tickfont=dict(color="#e0e0e0")),
    )
    out = PLOTS_DIR / "17_choropleth_temperature.html"
    fig.write_html(str(out))
    logger.info(f"Choropleth temperature saved → {out}")

    # Static PNG version
    try:
        fig.write_image(str(PLOTS_DIR / "17_choropleth_temperature.png"))
    except Exception:
        pass
    return out


@timer
def plot_choropleth_aqi(df: pd.DataFrame) -> Path:
    aqi_col = "aqi_composite" if "aqi_composite" in df.columns else None
    if "country" not in df.columns or not aqi_col:
        return None

    country_aqi = df.groupby("country")[aqi_col].mean().reset_index()
    country_aqi.columns = ["country", "avg_aqi"]

    fig = px.choropleth(
        country_aqi, locations="country", locationmode="country names",
        color="avg_aqi", hover_name="country",
        color_continuous_scale="YlOrRd",
        title="🌫️ Average AQI Composite by Country",
        template="plotly_dark",
    )
    fig.update_layout(
        paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
        font_color="#e0e0e0", title_font_size=18,
        geo=dict(bgcolor="#0f1117", landcolor="#1e1e2e",
                 showocean=True, oceancolor="#0d0d1a"),
    )
    out = PLOTS_DIR / "18_choropleth_aqi.html"
    fig.write_html(str(out))
    logger.info(f"Choropleth AQI saved → {out}")
    try:
        fig.write_image(str(PLOTS_DIR / "18_choropleth_aqi.png"))
    except Exception:
        pass
    return out


@timer
def plot_wind_scatter_geo(df: pd.DataFrame) -> Path:
    needed = ["country", "wind_kph"]
    if not all(c in df.columns for c in needed):
        return None

    country_wind = df.groupby("country")["wind_kph"].mean().reset_index()
    country_wind.columns = ["country", "avg_wind"]

    fig = px.choropleth(
        country_wind, locations="country", locationmode="country names",
        color="avg_wind", hover_name="country",
        color_continuous_scale="Blues",
        title="💨 Average Wind Speed by Country (km/h)",
        template="plotly_dark",
    )
    fig.update_layout(
        paper_bgcolor="#0f1117", font_color="#e0e0e0", title_font_size=18,
        geo=dict(bgcolor="#0f1117", landcolor="#1e1e2e",
                 showocean=True, oceancolor="#0d0d1a"),
    )
    out = PLOTS_DIR / "19_wind_geo.html"
    fig.write_html(str(out))
    logger.info(f"Wind geo plot saved → {out}")
    return out


@timer
def plot_country_heatmap(df: pd.DataFrame) -> Path:
    """Seaborn heatmap: countries × months."""
    if "country" not in df.columns or "month" not in df.columns:
        return None

    top_countries = df["country"].value_counts().head(20).index
    pivot = (df[df["country"].isin(top_countries)]
             .groupby(["country", "month"])["temperature_celsius"].mean()
             .unstack(fill_value=np.nan))

    fig, ax = plt.subplots(figsize=(16, 8), facecolor="#0f1117")
    ax.set_facecolor("#0f1117")
    import seaborn as sns
    sns.heatmap(pivot, cmap="coolwarm", annot=True, fmt=".1f",
                linewidths=0.3, linecolor="#222", ax=ax,
                annot_kws={"size": 7},
                cbar_kws={"shrink": 0.8, "label": "Avg Temp (°C)"})
    ax.set_title("Country × Month Temperature Heatmap (Top 20 Countries)",
                 color="#e0e0e0", fontsize=14)
    ax.set_xlabel("Month", color="#e0e0e0")
    ax.set_ylabel("Country", color="#e0e0e0")
    ax.tick_params(colors="#e0e0e0")
    plt.tight_layout()
    return save_figure(fig, "20_country_month_heatmap.png")


@timer
def run_spatial_analysis(df: pd.DataFrame) -> None:
    logger.info("Starting spatial analysis …")
    plot_choropleth_temperature(df)
    plot_choropleth_aqi(df)
    plot_wind_scatter_geo(df)
    plot_country_heatmap(df)
    logger.info("Spatial analysis complete.")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from preprocessing import run_preprocessing
    df, _, _ = run_preprocessing(save=False)
    run_spatial_analysis(df)
