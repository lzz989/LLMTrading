#!/usr/bin/env bash
set -euo pipefail

# One-command smoke check:
# - syntax/bytecode compile
# - minimal unit tests (no network)
PY="${PYTHON:-.venv/bin/python}"

if [[ ! -x "${PY}" ]]; then
  echo "找不到 Python：${PY}" >&2
  echo "可选：PYTHON=/path/to/python bash scripts/smoke.sh" >&2
  exit 2
fi

"${PY}" -m compileall -q llm_trading
"${PY}" -m unittest discover -s tests -p "test_*.py"

