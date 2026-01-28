from __future__ import annotations


def median(xs: list[float]) -> float:
    """
    轻量 median：
    - 空列表 => 0.0（调用方用它做“无样本兜底”时更省事）
    """
    if not xs:
        return 0.0
    xs2 = sorted(xs)
    n = len(xs2)
    mid = n // 2
    if n % 2 == 1:
        return float(xs2[mid])
    return float((xs2[mid - 1] + xs2[mid]) / 2.0)

