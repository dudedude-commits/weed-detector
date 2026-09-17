@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
title Сорняки на фото с дрона - запуск
cd /d "%~dp0"

echo ============================================
echo   Сорняки на фото с дрона - запуск
echo ============================================
echo.
echo Ищу подходящую версию Python (нужна 3.9 или новее)...
echo.

set "PYEXE="
set "PYCMD_ARGS="

REM --- 1) пробуем launcher "py" с конкретными версиями, от новых к старым:
REM        так мы находим современный Python, даже если "python" на PATH
REM        указывает на старую версию (частый случай, если Python
REM        ставился давно для другой программы) ---
for %%V in (3.13 3.12 3.11 3.10 3.9) do (
    if not defined PYEXE (
        py -%%V -c "exit()" >nul 2>nul
        if not errorlevel 1 (
            set "PYEXE=py"
            set "PYCMD_ARGS=-%%V"
        )
    )
)

REM --- 2) если launcher "py" без конкретной версии умеет сам выбрать
REM        current/newest >=3.9 ---
if not defined PYEXE (
    py -3 -c "import sys; exit(0 if sys.version_info>=(3,9) else 1)" >nul 2>nul
    if not errorlevel 1 (
        set "PYEXE=py"
        set "PYCMD_ARGS=-3"
    )
)

REM --- 3) в последнюю очередь - просто "python" из PATH, но проверяем
REM        его версию, а не берём вслепую ---
if not defined PYEXE (
    python -c "import sys; exit(0 if sys.version_info>=(3,9) else 1)" >nul 2>nul
    if not errorlevel 1 (
        set "PYEXE=python"
        set "PYCMD_ARGS="
    )
)

if not defined PYEXE (
    echo [ОШИБКА] Не нашёл на компьютере Python версии 3.9 или новее.
    echo.
    where python >nul 2>nul
    if not errorlevel 1 (
        echo Python на PATH есть, но это слишком старая версия:
        python --version 2>nul
        echo Нужна версия 3.9+, лучше всего 3.10-3.12.
    ) else (
        echo Python на компьютере не найден вообще.
    )
    echo.
    echo Установите Python 3.10-3.12 с сайта https://www.python.org/downloads/
    echo При установке обязательно отметьте галочку "Add python.exe to PATH".
    echo После установки запустите start.bat ещё раз.
    echo.
    pause
    exit /b 1
)

set "PY=%PYEXE% %PYCMD_ARGS%"
for /f "delims=" %%I in ('%PY% --version') do echo Использую: %%I (%PY%)
echo.

REM --- если .venv уже существует, но создан старым Python (например,
REM     после прошлого неудачного запуска) - пересоздаём его заново ---
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -c "import sys; exit(0 if sys.version_info>=(3,9) else 1)" >nul 2>nul
    if errorlevel 1 (
        echo Найдено старое виртуальное окружение .venv на устаревшем Python - пересоздаю...
        rmdir /s /q ".venv"
    )
)

REM --- создаём виртуальное окружение, если его ещё нет ---
if not exist ".venv\Scripts\activate.bat" (
    echo Создаю виртуальное окружение ^(.venv^)...
    %PY% -m venv .venv
    if not exist ".venv\Scripts\activate.bat" (
        echo.
        echo [ОШИБКА] Не удалось создать виртуальное окружение .venv
        echo.
        pause
        exit /b 1
    )
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 (
    echo.
    echo [ОШИБКА] Не удалось активировать виртуальное окружение.
    echo.
    pause
    exit /b 1
)

echo Устанавливаю/проверяю зависимости - это может занять пару минут...
python -m pip install --upgrade pip >nul
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [ОШИБКА] Не удалось установить зависимости из requirements.txt
    echo Проверьте подключение к интернету и повторите запуск.
    echo Если ошибка про версии numpy/pandas/streamlit - значит venv
    echo всё равно создался не на той версии Python: удалите папку
    echo .venv и запустите start.bat заново.
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================
echo   Готово! Открываю веб-интерфейс...
echo   Адрес: http://localhost:8501
echo   Чтобы остановить сервер - закройте это окно
echo   или нажмите Ctrl+C.
echo ============================================
echo.

streamlit run app.py

echo.
echo Сервер остановлен.
pause
