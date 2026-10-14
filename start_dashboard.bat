@echo off
title Groundwater AI Dashboard
echo ====================================================
echo Starting Groundwater AI Dashboard on http://localhost:8501 ...
echo ====================================================
cd /d "%~dp0"
python dashboard.py
pause
