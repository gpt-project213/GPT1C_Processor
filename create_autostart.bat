@echo off
:: Создаёт ярлык автозапуска watchdog в папке Startup Windows
:: Запускать один раз на каждой машине

set STARTUP=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup
set BAT_PATH=%~dp0start_bot_watchdog.bat
set SHORTCUT=%STARTUP%\GPT1C_Bot_Watchdog.lnk

powershell -Command "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut('%SHORTCUT%'); $s.TargetPath = '%BAT_PATH%'; $s.WorkingDirectory = '%~dp0'; $s.WindowStyle = 7; $s.Description = 'GPT1C Bot Watchdog'; $s.Save()"

if exist "%SHORTCUT%" (
    echo Автозапуск настроен: %SHORTCUT%
) else (
    echo ОШИБКА: ярлык не создан
)
pause
