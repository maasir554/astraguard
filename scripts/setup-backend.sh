#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
PYTHON="${ASTRAGUARD_PYTHON:-python3.11}"
if [ ! -x backend/.venv/bin/python ]; then "$PYTHON" -m venv backend/.venv; fi
backend/.venv/bin/python -m pip install --no-cache-dir -r backend/requirements.txt
backend/.venv/bin/python backend/scripts/prepare_weights.py
