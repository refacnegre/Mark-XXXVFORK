@echo off
REM Script'in bulunduğu klasörü bul
cd /d "%~dp0"

REM Git güncellemesi yap
git pull origin main

echo.
echo Repository guncellendi!
pause
