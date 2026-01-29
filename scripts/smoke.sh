#!/usr/bin/env bash
set -euo pipefail

# One-command smoke check:
# - syntax/bytecode compile
# - minimal unit tests (no network)
PY_IN="${PYTHON:-.venv/bin/python}"
PY="$(command -v "${PY_IN}" || true)"

# In CI we often pass PYTHON=python (a command, not a path). Resolve it via PATH.
if [[ -z "${PY}" ]]; then
  echo "找不到 Python：${PY_IN}" >&2
  echo "可选：PYTHON=/path/to/python bash scripts/smoke.sh" >&2
  exit 2
fi

"${PY}" -m compileall -q llm_trading
"${PY}" -m unittest discover -s tests -p "test_*.py"

# CLI parser smoke (no network)
"${PY}" -m llm_trading skill five_schools --help >/dev/null
"${PY}" -m llm_trading skill hotlines --help >/dev/null
