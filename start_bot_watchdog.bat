@echo off
:: ============================================================
:: GPT1C OtchetBot — Watchdog STABLE
:: ============================================================

cd /d C:\Users\User\Documents\GPT1C_Processor_analitic

:: путь к python venv
set PYTHON=C:\Users\User\Documents\GPT1C_Processor_analitic\.venv\Scripts\python.exe

:: ВАЖНО — путь к боту
set SCRIPT=bot\send_reports.py

set LOG=logs\watchdog.log
set RESTART_COUNT=0

echo [Watchdog] Запуск %DATE% %TIME% >> %LOG%

:LOOP
set /A RESTART_COUNT+=1

echo. >> %LOG%
echo ============================================================ >> %LOG%
echo [%DATE% %TIME%] Запуск #%RESTART_COUNT% >> %LOG%
echo ============================================================ >> %LOG%

:: запуск бота
%PYTHON% %SCRIPT%

set EXIT_CODE=%ERRORLEVEL%
echo [%DATE% %TIME%] Бот завершился с кодом %EXIT_CODE% >> %LOG%

:: если код 0 — штатная остановка
if %EXIT_CODE% EQU 0 (
    echo [%DATE% %TIME%] Чистый выход. Watchdog остановлен. >> %LOG%
    echo Бот остановлен штатно.
    pause
    exit /b 0
)

echo [%DATE% %TIME%] Ошибка. Перезапуск через 30 сек... >> %LOG%
echo Бот упал. Перезапуск через 30 сек...

timeout /t 30 /nobreak > nul
goto LOOP