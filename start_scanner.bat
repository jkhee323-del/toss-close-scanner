@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo [1/2] 필요한 패키지를 확인합니다...
python -m pip install -r requirements.txt --disable-pip-version-check -q
if errorlevel 1 (
  echo 패키지 설치에 실패했습니다. 인터넷 연결을 확인해주세요.
  pause
  exit /b 1
)
echo [2/2] 스캐너를 시작합니다...
python -m streamlit run app.py
pause
