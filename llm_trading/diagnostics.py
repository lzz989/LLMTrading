# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .logger import get_logger
from .pipeline import write_json

_LOG = get_logger(__name__)


@dataclass
class Diagnostics:
    """
    P0 吞错治理：把“降级但不中断主流程”的异常变成可见证据。

    - warnings：面向人看的短句（去重）
    - errors：结构化错误（去重；用于排查/聚合）
    """

    warnings: list[str] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)

    max_items: int = 200

    _warn_seen: set[str] = field(default_factory=set, init=False, repr=False)
    _err_seen: set[str] = field(default_factory=set, init=False, repr=False)

    def warn(self, msg: str, *, dedupe_key: str | None = None) -> None:
        m = str(msg or "").strip()
        if not m:
            return
        k = str(dedupe_key or m)
        if k in self._warn_seen:
            return
        self._warn_seen.add(k)
        if len(self.warnings) < int(self.max_items):
            self.warnings.append(m)
        try:
            _LOG.warning("%s", m)
        except (TypeError, ValueError, OverflowError, AttributeError, RuntimeError):  # noqa: BLE001
            # logger 也炸了就算了，别反手把主流程炸死
            return

    def record(self, stage: str, exc: BaseException, *, note: str | None = None, dedupe_key: str | None = None) -> None:
        k = str(dedupe_key or stage)
        if k in self._err_seen:
            return
        self._err_seen.add(k)
        if len(self.errors) < int(self.max_items):
            self.errors.append(
                {
                    "ts": datetime.now().isoformat(),
                    "stage": str(stage),
                    "type": exc.__class__.__name__,
                    "error": str(exc),
                    "note": (str(note) if note else None),
                }
            )
        self.warn((note or f"{stage} failed: {exc}"), dedupe_key=k)

    def write(self, out_dir: Path, *, cmd: str) -> None:
        try:
            payload = {
                "schema": "llm_trading.diagnostics.v1",
                "generated_at": datetime.now().isoformat(),
                "cmd": str(cmd),
                "warnings": list(self.warnings[: self.max_items]),
                "errors": list(self.errors[: self.max_items]),
            }
            # 兼容历史：默认仍写 diagnostics.json（单命令/测试依赖这个文件名）。
            write_json(Path(out_dir) / "diagnostics.json", payload)

            # 额外写一份“带命令名”的文件，避免 run 这类编排命令把子命令的 diagnostics.json 覆盖掉。
            # 例：diagnostics_holdings-user.json / diagnostics_rebalance-user.json
            safe = str(cmd or "").strip()
            safe = safe.replace("/", "_").replace("\\", "_").replace(" ", "_")
            safe = "".join(ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in safe)
            if safe and safe != "diagnostics":
                write_json(Path(out_dir) / f"diagnostics_{safe}.json", payload)
        except (OSError, TypeError, ValueError, OverflowError, AttributeError, RuntimeError) as exc:  # noqa: BLE001
            try:
                _LOG.warning("写出 diagnostics.json 失败: %s", exc)
            except (TypeError, ValueError, OverflowError, AttributeError, RuntimeError):  # noqa: BLE001
                pass
