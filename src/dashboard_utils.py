"""
dashboard_utils.py — Shared helpers for the Streamlit dashboard.

Provides:
  - Data loading with caching
  - Plotly figure builders reused across dashboard pages
  - PM Accelerator mission text
"""

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import joblib

from utils import DATA_FILE, MODELS_DIR, PLOTS_DIR, get_logger

logger = get_logger("dashboard_utils")

# ── PM Accelerator Mission Statement ─────────────────────────────────────────
PM_ACCELERATOR_MISSION = """
## 🚀 PM Accelerator Mission

**PM Accelerator** is the #1 Product Management community and education platform,
dedicated to empowering the next generation of world-class Product Managers.

Our mission is to bridge the gap between aspiring PMs and the skills, network,
and real-world experience needed to land top PM roles at leading tech companies.

Through **mentorship**, **hands-on projects**, **community support**, and
**data-driven learning**, PM Accelerator helps individuals transition into
product management or level up their careers with practical, industry-relevant experience.

> *"We don't just teach product management — we help you live it."*

### 🎯 Core Values
- **Data-Driven Decision Making** — Leverage analytics to drive product insights
- **Community First** — Learn and grow alongside industry professionals
- **Continuous Improvement** — Iterate fast, learn faster
- **Real-World Application** — Hands-on projects like this Weather Forecasting System

---
*This project was built as part of the PM Accelerator Data Science Internship Program.*
"""

DARK_TEMPLATE = "plotly_dark"
PAPER_BG      = "#0f1117"
PLOT_BG       = "#1e1e2e"
ACCENT_COLOR  = "#00d4ff"


# ─────────────────────────────────────────────
# Data loader
# ─────────────────────────────────────────────
def load_cleaned_data() -> pd.DataFrame:
    """Load cleaned parquet or fall back to raw CSV."""
    parquet_path = Path(__file__).resolve().parent.parent / "outputs" / "reports" / "cleaned_data.parquet"
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        logger.info(f"Loaded cleaned data from {parquet_path} — {len(df):,} rows")
        return df

    if DATA_FILE.exists():
        df = pd.read_csv(DATA_FILE, low_memory=False)
        logger.info(f"Loaded raw CSV — {len(df):,} rows")
        return df

    raise FileNotFoundError(
        "No dataset found. Please place GlobalWeatherRepository.csv in data/ "
        "or run preprocessing first."
    )


# ─────────────────────────────────────────────
# Chart builders
# ─────────────────────────────────────────────
def make_temperature_gauge(value: float, min_v: float = -30, max_v: float = 60) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Current Temperature (°C)", "font": {"color": "#e0e0e0"}},
        delta={"reference": 20},
        gauge={
            "axis": {"range": [min_v, max_v], "tickcolor": "#e0e0e0"},
            "bar":  {"color": ACCENT_COLOR},
            "steps": [
                {"range": [min_v, 0],  "color": "#1a3a5c"},
                {"range": [0,  20],    "color": "#1a5c3a"},
                {"range": [20, 35],    "color": "#5c4a1a"},
                {"range": [35, max_v], "color": "#5c1a1a"},
            ],
            "threshold": {
                "line": {"color": "#ff4757", "width": 3},
                "thickness": 0.75,
                "value": 35,
            },
        },
    ))
    fig.update_layout(paper_bgcolor=PAPER_BG, font_color="#e0e0e0", height=300)
    return fig


def make_country_bar(df: pd.DataFrame, metric: str, top_n: int = 15,
                     title: str = None, color_scale: str = "viridis") -> go.Figure:
    if "country" not in df.columns or metric not in df.columns:
        return go.Figure()
    data = df.groupby("country")[metric].mean().nlargest(top_n).sort_values()
    fig = go.Figure(go.Bar(
        x=data.values, y=data.index, orientation="h",
        marker=dict(color=data.values, colorscale=color_scale),
    ))
    fig.update_layout(
        title=title or f"Top {top_n} Countries — {metric}",
        template=DARK_TEMPLATE, paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        xaxis_title=metric, height=450,
    )
    return fig


def make_seasonal_box(df: pd.DataFrame, metric: str = "temperature_celsius") -> go.Figure:
    if "season" not in df.columns or metric not in df.columns:
        return go.Figure()
    seasons = ["Spring", "Summer", "Autumn", "Winter"]
    colors  = [ACCENT_COLOR, "#ff6b6b", "#ffd93d", "#6bcb77"]
    fig = go.Figure()
    for season, color in zip(seasons, colors):
        sub = df[df["season"] == season][metric].dropna()
        fig.add_trace(go.Box(y=sub, name=season, marker_color=color))
    label = metric.replace("_celsius", " (°C)").replace("_", " ").title()
    fig.update_layout(
        title=f"Seasonal Distribution — {label}",
        template=DARK_TEMPLATE, paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        yaxis_title=label,
    )
    return fig


def make_choropleth(df: pd.DataFrame, metric: str,
                    title: str = "", color_scale: str = "RdYlBu_r") -> go.Figure:
    if "country" not in df.columns or metric not in df.columns:
        return go.Figure()
    data = df.groupby("country")[metric].mean().reset_index()
    data.columns = ["country", "value"]
    fig = px.choropleth(
        data, locations="country", locationmode="country names",
        color="value", hover_name="country",
        color_continuous_scale=color_scale, title=title,
        template=DARK_TEMPLATE,
    )
    fig.update_layout(
        paper_bgcolor=PAPER_BG, font_color="#e0e0e0",
        geo=dict(bgcolor=PAPER_BG, landcolor=PLOT_BG,
                 showocean=True, oceancolor="#0d0d1a"),
    )
    return fig


def make_time_series(df: pd.DataFrame, metric: str = "temperature_celsius",
                     resample_freq: str = "D") -> go.Figure:
    if "datetime" not in df.columns or metric not in df.columns:
        return go.Figure()
    ts = (df[["datetime", metric]].dropna()
          .set_index("datetime").resample(resample_freq).mean().dropna().reset_index())
    label = metric.replace("_celsius","°C").replace("_"," ").title()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts["datetime"], y=ts[metric], mode="lines",
        line=dict(color=ACCENT_COLOR, width=1.5),
        fill="tozeroy", fillcolor="rgba(0,212,255,0.1)",
        name=label,
    ))
    fig.update_layout(
        title=f"{label} Over Time",
        template=DARK_TEMPLATE, paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        xaxis_title="Date", yaxis_title=label,
    )
    return fig


def make_model_comparison_chart(results: list[dict]) -> go.Figure:
    if not results:
        return go.Figure()
    df_r = pd.DataFrame([r for r in results if r.get("RMSE") is not None])
    if df_r.empty:
        return go.Figure()

    fig = make_subplots(rows=1, cols=3, subplot_titles=["RMSE", "MAE", "R²"])
    colors = [ACCENT_COLOR, "#ff6b6b", "#ffd93d", "#6bcb77", "#c77dff"]
    for col_i, metric in enumerate(["RMSE", "MAE", "R2"], 1):
        vals = df_r.set_index("Model")[metric].sort_values(ascending=(metric != "R2"))
        fig.add_trace(
            go.Bar(x=vals.index, y=vals.values,
                   marker_color=colors[:len(vals)], showlegend=False),
            row=1, col=col_i,
        )
    fig.update_layout(
        title="Model Performance Comparison",
        template=DARK_TEMPLATE, paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG,
        height=400,
    )
    return fig
