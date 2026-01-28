from __future__ import annotations

from datetime import datetime
from typing import Any

REPORT_SCHEMA_V1 = "llm_trading.report.v1"


def build_report_v1(
    *,
    cmd: str,
    run_meta: dict[str, Any] | None,
    run_config: dict[str, Any] | None,
    artifacts: dict[str, Any],
    counts: dict[str, Any] | None = None,
    summary: Any | None = None,
    warnings: list[str] | None = None,
    disclaimer: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    标准化报告（不同命令输出结构一致，细节放 summary/extra）。
    """
    r: dict[str, Any] = {
        "schema": REPORT_SCHEMA_V1,
        "generated_at": datetime.now().isoformat(),
        "cmd": str(cmd),
        "run_meta": run_meta,
        "run_config": run_config,
        "artifacts": artifacts,
        "counts": counts,
        "summary": summary,
        "warnings": warnings or [],
        "disclaimer": disclaimer or "研究工具输出，不构成投资建议；买卖自负。",
    }
    if extra:
        r["extra"] = extra
    return r

