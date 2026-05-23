@echo off
title Weather Trend Forecasting — Flask App
color 0B

echo.
echo  ╔══════════════════════════════════════════════════════════╗
echo  ║   WEATHER TREND FORECASTING — FLASK WEB APP              ║
echo  ║   PM Accelerator Data Science Internship Project         ║
echo  ╚══════════════════════════════════════════════════════════╝
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.9+
    pause
    exit /b 1
)

:: Check/create venv
if not exist "venv\Scripts\activate.bat" (
    echo [SETUP] Creating virtual environment...
    python -m venv venv
)

:: Activate venv
call venv\Scripts\activate.bat

:: Install dependencies
echo [SETUP] Installing dependencies...
pip install flask flask-cors pandas numpy matplotlib seaborn plotly scikit-learn scipy xgboost prophet shap statsmodels joblib pyarrow openpyxl tqdm --quiet

:: Create required directories
if not exist "outputs\plots" mkdir "outputs\plots"
if not exist "outputs\models" mkdir "outputs\models"
if not exist "outputs\reports" mkdir "outputs\reports"
if not exist "static\plots" mkdir "static\plots"
if not exist "static\css" mkdir "static\css"
if not exist "static\js" mkdir "static\js"
if not exist "logs" mkdir "logs"
if not exist "temp" mkdir "temp"

echo.
echo  Starting Flask app...
echo  Open your browser at: http://127.0.0.1:5000
echo.

python app.py

pause
