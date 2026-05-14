@echo off
title Machine Learning Driven Smart Intrusion Detection System
color 0A

echo.
echo ============================================================
echo     MACHINE LEARNING DRIVEN SMART INTRUSION DETECTION SYSTEM
echo     ML Models + Rule-Based Detection
echo ============================================================
echo.
echo Starting controlled system...
echo.
echo This will launch:
echo     Continuous Capture
echo     Auto Processor (5 ML Models)
echo     Streamlit Dashboard
echo.
echo The dashboard will open automatically in your browser
echo.
echo Press Ctrl+C in the Python window to stop everything
echo.

cd /d "C:\Users\ataha\IoT_Project"

echo.
echo [INFO] Working directory: %CD%
echo.

:: Check if model files exist
echo [CHECK] Verifying model files...
if exist "model\rf_model.pkl" (echo     Random Forest) else (echo     Random Forest - missing)
if exist "model\xgb_model.json" (echo     XGBoost) else (echo     XGBoost - missing)
if exist "model\et_model.pkl" (echo     Extra Trees) else (echo     Extra Trees - missing)
if exist "model\dt_model.pkl" (echo     Decision Tree) else (echo     Decision Tree - missing)
if exist "model\gb_model.joblib" (echo     Gradient Boosting) else (echo     Gradient Boosting - missing)
if exist "model\label_mapping.json" (echo     Label Mapping) else (echo     Label Mapping - missing)
echo.

echo [START] Launching controller...
echo.

python controller.py

echo.
echo [INFO] Controller has exited
pause