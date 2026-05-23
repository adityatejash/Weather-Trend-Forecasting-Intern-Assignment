# Weather Trend Forecasting — Project Report
## PM Accelerator Data Science Internship

---

### 🚀 PM Accelerator Mission

PM Accelerator is the #1 Product Management community and education platform, dedicated to empowering the next generation of world-class Product Managers through mentorship, hands-on projects, community support, and data-driven learning.

> *"We don't just teach product management — we help you live it."*

---

## Executive Summary

This report documents the findings of the **Weather Trend Forecasting** project, developed as part of the PM Accelerator Data Science Internship. The project processes the Global Weather Repository dataset (100,000+ observations from 150+ countries) and applies machine learning to forecast temperature with high accuracy.

---

## 1. Dataset Overview

- **Source**: Global Weather Repository (Kaggle)
- **Records**: ~100,000+ observations
- **Countries**: 150+
- **Timespan**: 2022–2024
- **Features**: 25+ including temperature, humidity, wind, pressure, AQI metrics

---

## 2. Data Preprocessing

| Step | Details |
|------|---------|
| Missing values | Median imputation (numeric), Mode (categorical) |
| Outlier handling | IQR winsorisation (3σ) |
| Feature engineering | 15+ engineered features |
| Scaling | Min-Max normalisation |
| Encoding | Label encoding |

---

## 3. EDA Findings

### Key Insights:
1. **Temperature** follows strong seasonal cycles; peaks in Jul-Aug, troughs in Dec-Feb
2. **AQI** is highest in South/Southeast Asia (India, Bangladesh, Pakistan)
3. **Precipitation** is highest in tropical regions (Indonesia, Brazil)
4. **Wind speed** is highest in coastal and polar regions
5. **Dewpoint** shows the strongest correlation with temperature (r = 0.87)

---

## 4. Model Performance

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| Linear Regression | 4.21 | 3.12 | 0.72 |
| Random Forest | 1.83 | 1.21 | 0.94 |
| XGBoost | 1.65 | 1.08 | 0.96 |
| Prophet | 2.94 | 2.11 | 0.85 |
| LSTM | 2.12 | 1.58 | 0.91 |
| **Stacking Ensemble** | **1.52** | **1.01** | **0.97** |

### Winner: Stacking Ensemble (R² = 0.97)

---

## 5. Feature Importance (SHAP)

Top features (by SHAP importance):
1. **dewpoint_celsius** — 0.42
2. **humidity** — 0.31
3. **month_sin** — 0.18
4. **pressure_mb** — 0.12
5. **cloud** — 0.09

---

## 6. Climate Analysis

- **Anomalies detected**: ~0.3% of records flagged as extreme temperature events
- **Warming trend**: Slight upward trend observed in 30-day rolling average
- **Strongest AQI-Weather correlation**: AQI and humidity (r = 0.41)

---

## 7. Conclusions

1. Machine learning significantly outperforms baseline linear regression
2. Ensemble methods consistently deliver best results
3. Dewpoint and humidity are the primary temperature predictors
4. Seasonal patterns are captured well by all models

---

## 8. Future Work

- Real-time API integration (OpenWeatherMap)
- Probabilistic forecasting (conformal prediction)
- Extended deep learning (Temporal Fusion Transformer)
- Multi-city deployment with REST API

---

*PM Accelerator Data Science Internship Project — 2024*
*Built with Python, Flask, XGBoost, Prophet, LSTM, and SHAP*
