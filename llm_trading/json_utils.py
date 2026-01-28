from __future__ import annotations

import math
from typing import Any


def _is_bad_float(x: float) -> bool:
    try:
        return math.isnan(x) or math.isinf(x)
    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
        return False


def sanitize_for_json(obj: Any) -> Any:
    """
    Starlette/FastAPI 的 JSONResponse 默认不允许 NaN/Inf（严格 JSON）。
    另外浏览器的 JSON.parse 也不认 NaN/Inf（会直接炸）。

    所以：把所有非有限浮点都清成 None（=> null），别让前端/接口跪了。
    """
    if obj is None:
        return None

    # 常见标量
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        return None if _is_bad_float(obj) else obj
    if isinstance(obj, str):
        return obj

    # 容器
    if isinstance(obj, list):
        return [sanitize_for_json(x) for x in obj]
    if isinstance(obj, tuple):
        return [sanitize_for_json(x) for x in obj]
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}

    # numpy/pandas 标量（可选依赖）
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return sanitize_for_json(obj.item())
    except (AttributeError):  # noqa: BLE001
        pass

    # 兜底：别让 json.dumps 报类型错误
    try:
        return str(obj)
    except (AttributeError):  # noqa: BLE001
        return None

