#!/usr/bin/env bash
set -euo pipefail

# Coverage runner:
# - compileall
# - unittest with coverage.py
PY_IN="${PYTHON:-.venv/bin/python}"
PY="$(command -v "${PY_IN}" || true)"

# In CI we often pass PYTHON=python (a command, not a path). Resolve it via PATH.
if [[ -z "${PY}" ]]; then
  echo "找不到 Python：${PY_IN}" >&2
  echo "可选：PYTHON=/path/to/python bash scripts/coverage.sh" >&2
  exit 2
fi

"${PY}" -m compileall -q llm_trading

# 默认门槛：60%（Phase5 验收线）；想只看报告可设 COVERAGE_FAIL_UNDER=0
"${PY}" scripts/run_coverage.py --fail-under "${COVERAGE_FAIL_UNDER:-60}"
