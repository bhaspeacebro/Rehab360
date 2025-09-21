@echo off
echo =====================================
echo    REHAB360 PRO - Motion Analysis
echo =====================================
echo.
cd /d "c:\Users\bhask\Projects\Rehab360"

echo Checking dependencies...
python -c "import cv2, mediapipe, numpy" 2>NUL
if %errorlevel% neq 0 (
    echo.
    echo Installing required packages...
    pip install opencv-python mediapipe numpy pyttsx3 pyyaml
    echo.
)

echo Starting application...
echo.
py -3.11 src\main.py
if %errorlevel% neq 0 (
    echo.
    echo Trying legacy version...
    py -3.11 src\mainApp.py
)
pause
