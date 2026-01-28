# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import sys
import unittest
from pathlib import Path


def _patch_sqlite3_if_broken() -> None:
    """
    coverage.py 依赖 sqlite3；但某些 Conda/Python 发行版的 stdlib sqlite3 可能坏掉。
    这里用 pysqlite3-binary 做兜底（只在必要时启用）。
    """
    try:
        import sqlite3  # noqa: F401
        return
    except Exception:  # noqa: BLE001
        pass

    try:
        import pysqlite3.dbapi2 as sqlite3_mod
    except ModuleNotFoundError as exc:  # noqa: BLE001
        raise SystemExit(
            "当前 Python 的 stdlib sqlite3 模块不可用，且未安装 pysqlite3-binary。\n"
            "解决：pip install -r requirements-dev.txt"
        ) from exc

    sys.modules["sqlite3"] = sqlite3_mod
    sys.modules["sqlite3.dbapi2"] = sqlite3_mod


def main(argv: list[str] | None = None) -> int:
    # 以脚本方式运行时，sys.path[0] 会变成 scripts/，导致 import llm_trading 失败。
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    ap = argparse.ArgumentParser()
    ap.add_argument("--fail-under", type=float, default=float(os.getenv("COVERAGE_FAIL_UNDER", "0") or "0"))
    ap.add_argument("--rcfile", type=str, default=os.getenv("COVERAGE_RCFILE", ".coveragerc"))
    args = ap.parse_args(argv)

    _patch_sqlite3_if_broken()

    try:
        import coverage
    except ModuleNotFoundError as exc:  # noqa: BLE001
        raise SystemExit("未安装 coverage：pip install -r requirements-dev.txt") from exc

    cov = coverage.Coverage(config_file=str(args.rcfile) if args.rcfile else True)
    cov.start()

    suite = unittest.defaultTestLoader.discover("tests", pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)

    cov.stop()
    cov.save()

    total = float(cov.report())
    print(f"[coverage] total={total:.2f}% fail_under={float(args.fail_under):.2f}%")

    if not result.wasSuccessful():
        return 1
    if float(args.fail_under) > 0 and total < float(args.fail_under):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
