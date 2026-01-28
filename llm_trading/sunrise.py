from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SunriseParams:
    """
    “旭日东升/看涨吞没”形态（研究用途）的可量化参数。

    核心：下跌背景 + 两K反转（前阴后阳，阳线实体吞没阴线实体）+ 可选量能确认。
    """

    # 下跌背景：用“前一根K线的收盘”相对 N 日前的收盘，要求跌幅至少 down_ret_min
    trend_lookback_days: int = 10
    down_ret_min: float = 0.05

    # K线实体过滤：避免十字星/小实体误判（body/close 至少该比例）
    min_body_pct: float = 0.003

    # 吞没容错：价格四舍五入/复权差异
    engulf_tol: float = 0.001

    # 吞没力度：阳线实体 >= 阴线实体 * ratio（1.0=不缩放）
    body_engulf_ratio: float = 1.0

    # “接近阶段低位”：前一根收盘 <= rolling_low * (1+tol)
    near_low_days: int = 30
    near_low_tol: float = 0.06

    # 量能确认（可选）：阳线量 >= 阴线量 * ratio
    require_volume_increase: bool = False
    vol_ratio_min: float = 1.10


def compute_sunrise_signal(df_daily, *, params: SunriseParams | None = None):
    """
    生成“旭日东升/看涨吞没”信号序列（收盘后评估，给次日计划用）。
    - 信号发生在第 i 天（需要 i-1 存在）
    - 不做未来函数
    """
    p = params or SunriseParams()
    try:
        import numpy as np
        import pandas as pd
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("缺依赖：先跑 pip install -r \"requirements.txt\"") from exc

    if df_daily is None or getattr(df_daily, "empty", True):
        return pd.Series([], dtype=bool)

    df = df_daily.copy()
    df["date"] = pd.to_datetime(df.get("date"), errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    if df.empty:
        return pd.Series([], dtype=bool)

    n = int(len(df))
    if n < 3:
        return pd.Series([False] * n, index=df.index, dtype=bool)

    open_s = pd.to_numeric(df.get("open", df["close"]), errors="coerce").astype(float)
    close_s = pd.to_numeric(df["close"], errors="coerce").astype(float)
    vol_s = pd.to_numeric(df.get("volume", 0.0), errors="coerce").fillna(0.0).astype(float)

    prev_open = open_s.shift(1)
    prev_close = close_s.shift(1)

    # 形态：前阴后阳
    prev_bear = (prev_close < prev_open).fillna(False)
    cur_bull = (close_s > open_s).fillna(False)

    # 实体过滤：避免 doji
    body = (close_s - open_s).abs()
    prev_body = (prev_close - prev_open).abs()
    body_pct = body / close_s.replace({0.0: np.nan})
    prev_body_pct = prev_body / prev_close.replace({0.0: np.nan})
    ok_body = (body_pct >= float(p.min_body_pct)).fillna(False) & (prev_body_pct >= float(p.min_body_pct)).fillna(False)

    # 吞没（带容错）
    tol = float(p.engulf_tol)
    ok_engulf = (open_s <= prev_close * (1.0 + tol)).fillna(False) & (close_s >= prev_open * (1.0 - tol)).fillna(False)

    ok_body_ratio = (body >= prev_body * float(p.body_engulf_ratio)).fillna(False)

    # 下跌背景：用“前一根收盘”对比 N 日前收盘
    lb = max(2, int(p.trend_lookback_days))
    base = close_s.shift(lb + 1)
    down_ret = (prev_close / base.replace({0.0: np.nan})) - 1.0
    ok_downtrend = (down_ret <= (-abs(float(p.down_ret_min)))).fillna(False)

    # 接近阶段低位：避免在半山腰“假见底”
    near_days = max(5, int(p.near_low_days))
    roll_low = close_s.rolling(window=near_days, min_periods=near_days).min().shift(1)  # 用到前一日为止
    ok_near_low = (prev_close <= roll_low * (1.0 + abs(float(p.near_low_tol)))).fillna(False)

    # 量能确认：默认不开（很多源没 volume/复权差异也大）
    ok_vol = pd.Series([True] * n, index=df.index, dtype=bool)
    if bool(p.require_volume_increase) and float(vol_s.max()) > 0:
        ok_vol = (vol_s >= (vol_s.shift(1) * float(p.vol_ratio_min))).fillna(False)

    sig = prev_bear & cur_bull & ok_body & ok_engulf & ok_body_ratio & ok_downtrend & ok_near_low & ok_vol
    sig = sig.fillna(False)
    sig.iloc[0] = False
    return sig.astype(bool)


def describe_sunrise_setup(df_daily, *, params: SunriseParams | None = None) -> dict[str, Any]:
    """
    解释“最后一根K线”是否触发旭日东升（用于 scan-sunrise 输出）。
    """
    p = params or SunriseParams()
    try:
        import numpy as np
        import pandas as pd
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("缺依赖：先跑 pip install -r \"requirements.txt\"") from exc

    if df_daily is None or getattr(df_daily, "empty", True):
        return {"ok": False, "fails": ["无数据"]}

    df = df_daily.copy()
    df["date"] = pd.to_datetime(df.get("date"), errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    if df.empty or len(df) < 3:
        return {"ok": False, "fails": ["K线太少"]}

    i = int(len(df) - 1)
    last = df.iloc[i]
    prev = df.iloc[i - 1]

    def fnum(v):
        try:
            return None if v is None else float(v)
        except (TypeError, ValueError, OverflowError, KeyError, IndexError, AttributeError):  # noqa: BLE001
            return None

    open0 = fnum(last.get("open")) or fnum(last.get("close")) or 0.0
    close0 = fnum(last.get("close"))
    vol0 = fnum(last.get("volume")) or 0.0

    open1 = fnum(prev.get("open")) or fnum(prev.get("close")) or 0.0
    close1 = fnum(prev.get("close"))
    vol1 = fnum(prev.get("volume")) or 0.0

    last_dt = last.get("date")
    last_date = last_dt.strftime("%Y-%m-%d") if hasattr(last_dt, "strftime") else str(last_dt)

    fails: list[str] = []
    metrics: dict[str, Any] = {}

    if close0 is None or close1 is None or open0 <= 0 or open1 <= 0:
        return {"ok": False, "fails": ["价格字段缺失"], "date": last_date}

    prev_bear = bool(close1 < open1)
    cur_bull = bool(close0 > open0)
    if not prev_bear:
        fails.append("前一根不是阴线")
    if not cur_bull:
        fails.append("当前不是阳线")

    body0 = abs(float(close0) - float(open0))
    body1 = abs(float(close1) - float(open1))
    body_pct0 = (body0 / float(close0)) if float(close0) > 0 else None
    body_pct1 = (body1 / float(close1)) if float(close1) > 0 else None
    metrics["body_pct"] = {"today": body_pct0, "prev": body_pct1}
    if body_pct0 is None or body_pct1 is None or body_pct0 < float(p.min_body_pct) or body_pct1 < float(p.min_body_pct):
        fails.append("实体太小(疑似十字星)")

    tol = float(p.engulf_tol)
    ok_engulf = (open0 <= float(close1) * (1.0 + tol)) and (float(close0) >= float(open1) * (1.0 - tol))
    metrics["engulf"] = {"tol": tol, "ok": ok_engulf}
    if not ok_engulf:
        fails.append("未满足吞没关系")

    ratio = float(p.body_engulf_ratio)
    ok_body_ratio = (body0 >= body1 * ratio) if body1 > 0 else False
    metrics["body_ratio"] = {"ratio": ratio, "today_over_prev": (body0 / body1) if body1 > 0 else None, "ok": ok_body_ratio}
    if not ok_body_ratio:
        fails.append("阳线力度不足")

    # 下跌背景
    close_s = pd.to_numeric(df["close"], errors="coerce").astype(float)
    lb = max(2, int(p.trend_lookback_days))
    base = close_s.shift(lb + 1)
    down_ret = None
    try:
        down_ret = float((close_s.shift(1) / base.replace({0.0: np.nan}) - 1.0).iloc[i])
    except (TypeError, ValueError, OverflowError, KeyError, IndexError, AttributeError):  # noqa: BLE001
        down_ret = None
    metrics["downtrend"] = {"lookback_days": int(lb), "down_ret": down_ret, "min": float(p.down_ret_min)}
    if down_ret is None or down_ret > (-abs(float(p.down_ret_min))):
        fails.append("下跌背景不足")

    # 接近阶段低位
    near_days = max(5, int(p.near_low_days))
    roll_low_prev = close_s.rolling(window=near_days, min_periods=near_days).min().shift(1)
    roll_low_v = None
    ok_near_low = False
    try:
        roll_low_v = float(roll_low_prev.iloc[i])
        ok_near_low = float(close1) <= roll_low_v * (1.0 + abs(float(p.near_low_tol)))
    except (TypeError, ValueError, OverflowError, KeyError, IndexError, AttributeError):  # noqa: BLE001
        roll_low_v = None
        ok_near_low = False
    metrics["near_low"] = {"days": int(near_days), "rolling_low": roll_low_v, "tol": float(p.near_low_tol), "ok": ok_near_low}
    if not ok_near_low:
        fails.append("不在阶段低位附近")

    # 量能确认（可选）
    ok_vol = True
    vol_ratio = None
    if bool(p.require_volume_increase) and (vol0 > 0 and vol1 > 0):
        vol_ratio = float(vol0 / vol1) if vol1 > 0 else None
        ok_vol = bool(vol_ratio is not None and vol_ratio >= float(p.vol_ratio_min))
        if not ok_vol:
            fails.append("量能未确认")
    metrics["volume"] = {"require": bool(p.require_volume_increase), "ratio": vol_ratio, "min": float(p.vol_ratio_min), "ok": ok_vol}

    ok = not bool(fails)
    why = "通过：下跌背景 + 前阴后阳 + 阳线吞没阴线（可选量能确认）" if ok else ("差：" + " / ".join(fails[:4]) + (" …" if len(fails) > 4 else ""))
    return {
        "ok": ok,
        "fails": fails,
        "why": why,
        "date": last_date,
        "close": float(close0),
        "metrics": metrics,
    }

