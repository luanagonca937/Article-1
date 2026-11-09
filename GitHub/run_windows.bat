@echo off
cd /d "%~dp0"
python -m venv .venv
call .venv\Scripts\activate
pip install -r requirements-tested.txt || pip install -r requirements.txt
python reproduce_article1.py %*
pause
