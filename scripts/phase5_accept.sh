#!/usr/bin/env bash
set -euo pipefail

# Phase5 验收一键检查（覆盖率优先 + 吞错治理）：
# - smoke：compileall + unittest
# - coverage：核心模块（见 .coveragerc include）总覆盖率 >= 80%
# - except Exception：数量相对当前仓库基线(86)下降 >=50% => <=43
#
# 用法：
#   bash scripts/phase5_accept.sh
#   PYTHON=/path/to/python bash scripts/phase5_accept.sh

PY="${PYTHON:-.venv/bin/python}"
if [[ ! -x "${PY}" ]]; then
  echo "找不到 Python：${PY}" >&2
  exit 2
fi

PYTHON="${PY}" bash scripts/smoke.sh
PYTHON="${PY}" COVERAGE_FAIL_UNDER=80 bash scripts/coverage.sh

EXCEPT_EXCEPTION_COUNT="$(rg -n 'except Exception' llm_trading | wc -l | tr -d '[:space:]')"
THRESHOLD=43
echo "[phase5] except Exception count=${EXCEPT_EXCEPTION_COUNT} threshold=${THRESHOLD}"
if [[ "${EXCEPT_EXCEPTION_COUNT}" -gt "${THRESHOLD}" ]]; then
  echo "[phase5] FAIL: except Exception count too high" >&2
  exit 3
fi

echo "[phase5] OK"

