@echo off
:: ============================================================
:: GPT1C OtchetBot — Watchdog STABLE
:: Пути относительные — работает и на E:\ и на C:\
:: ============================================================

cd /d "%~dp0"

:: Путь к python venv (относительно папки скрипта)
set PYTHON=%~dp0.venv\Scripts\python.exe

:: Скрипт и лог
set SCRIPT=bot\send_reports.py
set LOG=logs\watchdog.log
set STOP_FILE=logs\bot.stop
set RESTART_COUNT=0

echo [Watchdog] Старт %DATE% %TIME% >> %LOG%

:LOOP
if exist "%STOP_FILE%" (
    echo [%DATE% %TIME%] Stop file found: %STOP_FILE%. Watchdog stopped. >> %LOG%
    exit /b 0
)

set /A RESTART_COUNT+=1

echo. >> %LOG%
echo ============================================================ >> %LOG%
echo [%DATE% %TIME%] Запуск #%RESTART_COUNT% >> %LOG%
echo ============================================================ >> %LOG%

:: Запуск бота
"%PYTHON%" %SCRIPT%

set EXIT_CODE=%ERRORLEVEL%
echo [%DATE% %TIME%] Бот завершился с кодом %EXIT_CODE% >> %LOG%

:: Код 0 — штатное завершение, перезапускаем через 5 сек (не блокируем)
if %EXIT_CODE% EQU 0 (
    echo [%DATE% %TIME%] Штатный выход. Перезапуск через 5 сек... >> %LOG%
    timeout /t 5 /nobreak > nul
    goto LOOP
)

echo [%DATE% %TIME%] Ошибка. Перезапуск через 30 сек... >> %LOG%
timeout /t 30 /nobreak > nul
goto LOOP
