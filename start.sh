#!/usr/bin/env bash
# Стартовый файл для Linux/macOS — одна команда вместо ручной установки.
#   ./start.sh
set -e
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  echo "▸ Создаю виртуальное окружение (.venv)…"
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "▸ Устанавливаю/проверяю зависимости…"
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo "▸ Запускаю веб-интерфейс — откроется http://localhost:8501"
streamlit run app.py
