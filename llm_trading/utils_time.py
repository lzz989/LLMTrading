from __future__ import annotations

from datetime import datetime


def parse_date_any(s: str) -> datetime:
    """
    解析日期字符串：
    - YYYYMMDD
    - YYYY-MM-DD

    空字符串直接抛 ValueError，别让上层拿 None 当日期继续跑。
    """
    s2 = str(s or "").strip()
    if not s2:
        raise ValueError("空日期")
    if "-" in s2:
        return datetime.strptime(s2, "%Y-%m-%d")
    return datetime.strptime(s2, "%Y%m%d")


def parse_date_any_opt(s: str | None) -> datetime | None:
    """
    可选日期解析：空值/空白 => None。
    """
    s2 = str(s or "").strip()
    if not s2:
        return None
    return parse_date_any(s2)


def as_yyyymmdd(dt: datetime) -> str:
    return dt.strftime("%Y%m%d")

