# -*- coding: utf-8 -*-
"""
统一日志（Phase5）。

目标：
- CLI/研究工具默认能看见关键信息（stdout）
- 需要时可落文件（debug 追问题）
- 避免重复添加 handler（不然日志会刷屏）
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any


def setup_logger(name: str, *, log_file: Path | None = None, level: str | None = None) -> logging.Logger:
    """
    统一 logger：
    - 默认 INFO；可用环境变量 LLM_TRADING_LOG_LEVEL 覆盖
    - log_file 不为空则追加 FileHandler（DEBUG 级别）
    """
    lv = (level or os.getenv("LLM_TRADING_LOG_LEVEL", "INFO")).strip().upper() or "INFO"
    if lv not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}:
        lv = "INFO"

    logger = logging.getLogger(str(name or "llm_trading"))
    logger.setLevel(getattr(logging, lv, logging.INFO))

    # 避免重复添加 handler（多次 import/setup 时会刷屏）
    if getattr(logger, "_llm_trading_configured", False):
        return logger

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(getattr(logging, lv, logging.INFO))
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file is not None:
        try:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(str(log_file), encoding="utf-8")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        except (AttributeError):  # noqa: BLE001
            # 文件 handler 失败就别硬炸主流程（研究工具优先可用）
            pass

    logger.propagate = False
    setattr(logger, "_llm_trading_configured", True)
    return logger


def get_logger(name: str = "llm_trading") -> logging.Logger:
    return setup_logger(name)


def log_exception(logger: logging.Logger, msg: str, exc: BaseException) -> None:
    """
    统一 exception 输出：默认带 traceback（debug 用）。
    """
    try:
        logger.exception(msg)
    except (AttributeError):  # noqa: BLE001
        try:
            logger.error("%s: %s", msg, exc)
        except (AttributeError):  # noqa: BLE001
            pass

