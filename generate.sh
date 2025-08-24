#!/usr/bin/env bash
set -euo pipefail
# optional: einmalig Modelle ziehen (idempotent)
python download_weights.py

# API starten
exec uvicorn app:app --host 0.0.0.0 --port 8000
