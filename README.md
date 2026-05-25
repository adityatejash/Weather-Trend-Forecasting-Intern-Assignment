# 🌤️ Weather Trend Forecasting

> **AI-powered global weather analytics platform** using 5 ML models on 700,000+ real-world observations.  
> PM Accelerator Data Science Internship Project — Flask Edition

![Flask](https://img.shields.io/badge/Flask-3.0+-blue?logo=flask)
![Python](https://img.shields.io/badge/Python-3.9+-yellow?logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-✓-green)
![Prophet](https://img.shields.io/badge/Prophet-✓-orange)
![LSTM](https://img.shields.io/badge/LSTM-TensorFlow-red)

---

## 🚀 Quick Start (Flask)

```bash
# 1. Clone or download the project
cd Weather-Trend-Forecasting

# 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Flask app
python app.py

# 5. Open browser
# http://127.0.0.1:5000
```

**Or just double-click `run_project.bat`** (Windows).

---

## 🗺️ Project Structure

```
Weather-Trend-Forecasting/
├── app.py                        ← Flask application (main entry point)
├── src/
│   ├── preprocessing.py          ← Data cleaning & feature engineering
│   ├── eda.py                    ← 20+ EDA plots
│   ├── forecasting.py            ← LR / RF / XGBoost / Prophet / LSTM
│   ├── ensemble.py               ← Stacking ensemble (Ridge meta-learner)
│   ├── feature_importance.py     ← SHAP + permutation importance
│   ├── climate_analysis.py       ← Anomaly detection, rolling trends
│   ├── spatial_analysis.py       ← Choropleth maps
│   ├── utils.py                  ← Shared helpers, paths, metrics
│   └── main.py                   ← CLI pipeline runner
├── templates/
│   ├── base.html                 ← Shared layout
│   ├── index.html                ← Landing page
│   ├── dashboard.html            ← Plot gallery
│   ├── prediction.html           ← ML prediction form
│   ├── analysis.html             ← Climate & AQI analysis
│   └── model_comparison.html     ← Model benchmarks
├── static/
│   ├── css/style.css
│   ├── js/main.js
│   └── plots/                    ← Synced plot images
├── outputs/
│   ├── plots/                    ← Generated matplotlib/plotly plots
│   ├── models/                   ← Saved ML models (.pkl / .json)
│   └── reports/                  ← JSON results, parquet data
├── data/
│   └── GlobalWeatherRepository.csv
├── requirements.txt
├── run_project.bat
└── README.md
```

---

## 🌐 Flask Routes

| Route | Description |
|---|---|
| `/` | Landing page with hero, stats, feature cards |
| `/dashboard` | Full plot gallery with filter buttons |
| `/predict` | ML prediction form (POST) with Chart.js results |
| `/analysis` | Climate & AQI analysis with insight cards |
| `/models` | Model comparison leaderboard & metrics table |
| `/api/predict` | JSON prediction API (POST) |
| `/api/plots` | List all available plots (GET) |
| `/api/status` | Pipeline run status (GET) |
| `/api/run-pipeline` | Start full ML pipeline (POST) |
| `/api/model-results` | Model metrics JSON (GET) |

---

## 🤖 ML Models

| Model | Type | Best For |
|---|---|---|
| Linear Regression | Baseline | Interpretability |
| Random Forest | Ensemble | Tabular data |
| **XGBoost** | Gradient Boost | Best accuracy |
| Prophet | Time Series | Seasonality |
| LSTM | Deep Learning | Temporal patterns |
| **Stacking Ensemble** | Meta-learner | Overall best |

---

## ✨ Features

- 🌍 **700,000+ records** from GlobalWeatherRepository (Kaggle)
- 📊 **20+ auto-generated plots** saved to `outputs/plots/`
- 🤖 **5 ML models** trained end-to-end
- 🔍 **SHAP explainability** — feature importance visualization
- 🗺️ **Interactive choropleth maps** via Plotly
- 🌡️ **Climate anomaly detection** using Z-score (|Z| > 3σ)
- 📈 **Real-time predictions** via Flask web form
- 🔗 **REST API** for programmatic access
- 🎨 **Modern dark glassmorphism UI** with Bootstrap 5 + Chart.js

---

## 📦 Dataset

**GlobalWeatherRepository.csv** (~35 MB, 700K+ rows)  
Source: [Kaggle — Global Weather Repository](https://www.kaggle.com/datasets/nelgiriyewithana/global-weather-repository)

Place the CSV in `data/GlobalWeatherRepository.csv` before running the pipeline.

---

## 🔧 Running the Full ML Pipeline

Option 1 — Via the web UI:
- Go to `http://127.0.0.1:5000`
- Click **Run Pipeline** button in the navbar
- All 7 stages run in the background

Option 2 — Via CLI:
```bash
python src/main.py
# With flags:
python src/main.py --skip-eda --skip-lstm
```

Option 3 — Via API:
```bash
curl -X POST http://127.0.0.1:5000/api/run-pipeline
```

---

## 📸 Screenshots

| Page | Description |
|---|---|
| Home | Hero section with animated weather globe |
| Dashboard | Plot gallery with filter buttons |
| Predict | Input form + live prediction results |
| Analysis | Climate & AQI insight cards |
| Models | Podium leaderboard + metrics table |

---

## 🏗️ Tech Stack

- **Backend**: Python 3.9+, Flask 3.0, Flask-CORS
- **ML**: scikit-learn, XGBoost, Prophet, TensorFlow/Keras, SHAP
- **Data**: pandas, numpy, scipy, statsmodels
- **Viz**: matplotlib, seaborn, plotly
- **Frontend**: Bootstrap 5.3, Chart.js, Inter font, vanilla CSS/JS
- **Dataset**: GlobalWeatherRepository.csv (Kaggle)

---

## 📄 License

MIT License — PM Accelerator Data Science Internship 2026
