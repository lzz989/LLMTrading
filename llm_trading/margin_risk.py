# -*- coding: utf-8 -*-
"""
两融（融资融券）= 市场杠杆温度计（研究用途）。

目标：
- 给 CashSignal / MarketRegime 提供一个“过热/去杠杆”的可计算证据。
- 必须缓存 + 可降级：数据源挂了也不能拖垮主流程。

注意：
- 这是“环境风控/风险加权”，不是买卖按钮。
- 数据源来自 AkShare（Jin10 汇总口径），可能缺失/延迟；遇到异常请以 ok=false 降级。
"""

from __future__ import annotations

import math
from datetime import date, datetime
from pathlib import Path
from typing import Any


class MarketMarginRiskError(RuntimeError):
    pass


def _safe_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        x = float(v)
    except (TypeError, ValueError, OverflowError):  # noqa: BLE001
        return None
    if not math.isfinite(x):
        return None
    return float(x)


def _median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    ys = sorted(float(x) for x in xs)
    n = len(ys)
    mid = n // 2
    if n % 2 == 1:
        return float(ys[mid])
    return float(0.5 * (ys[mid - 1] + ys[mid]))


def _mad(xs: list[float], *, center: float | None = None) -> float:
    if not xs:
        return 0.0
    c = float(center) if center is not None else _median(xs)
    dev = [abs(float(x) - c) for x in xs]
    return float(_median(dev))


def robust_zscore(x: float | None, history: list[float]) -> float | None:
    x2 = _safe_float(x)
    if x2 is None:
        return None
    hs = [float(v) for v in history if _safe_float(v) is not None]
    if len(hs) < 8:
        return None
    med = _median(hs)
    mad = _mad(hs, center=med)
    if mad <= 1e-12:
        return 0.0
    return float((x2 - med) / (1.4826 * mad))


def z_to_score01(z: float | None, *, clip: float = 3.0) -> float | None:
    z2 = _safe_float(z)
    if z2 is None:
        return None
    c = max(0.5, float(clip))
    if z2 >= c:
        return 1.0
    if z2 <= -c:
        return 0.0
    return float((z2 + c) / (2.0 * c))


def _read_csv_if_fresh(path: Path, *, ttl_hours: float):
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise MarketMarginRiskError("缺依赖：pandas 未安装") from exc

    if (not path.exists()) or float(ttl_hours) <= 0:
        return None
    try:
        age = (datetime.now().timestamp() - path.stat().st_mtime) / 3600.0
        if age > float(ttl_hours):
            return None
    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
        return None
    try:
        return pd.read_csv(path, encoding="utf-8")
    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
        return None


def _write_csv_silent(df, path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except (OSError, AttributeError):  # noqa: BLE001
        return
    try:
        df.to_csv(path, index=False, encoding="utf-8")
    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
        pass


def _pct_change(xs: list[float], n: int) -> float | None:
    if len(xs) <= int(n):
        return None
    v0 = xs[-1 - int(n)]
    v1 = xs[-1]
    if (not math.isfinite(v0)) or (not math.isfinite(v1)) or v0 <= 0:
        return None
    return float(v1 / v0 - 1.0)


def compute_market_margin_risk(
    *,
    as_of: date,
    cache_dir: Path,
    ttl_hours: float = 6.0,
    lookback_days: int = 180,
) -> dict[str, Any]:
    """
    计算“全市场两融余额”杠杆温度计（研究用途）。

    指标：
    - total_margin_balance_yuan：上证+深证 融资融券余额合计（元）
    - z/score01：对 total_margin_balance 做 robust z-score（历史窗口不含当前点）
    - overheat：杠杆偏热（score01 很高 且近20日未明显去杠杆）
    - deleveraging：去杠杆压力（近5/20日下降明显）
    """
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise MarketMarginRiskError("缺依赖：pandas 未安装") from exc

    try:
        import akshare as ak
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise MarketMarginRiskError("缺依赖：akshare 未安装") from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    path_sh = cache_dir / "market_margin_sh.csv"
    path_sz = cache_dir / "market_margin_sz.csv"

    df_sh = _read_csv_if_fresh(path_sh, ttl_hours=float(ttl_hours))
    if df_sh is None or getattr(df_sh, "empty", True):
        try:
            df_sh = ak.macro_china_market_margin_sh()
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"AkShare macro_china_market_margin_sh 调用失败：{exc}"}
        if df_sh is None or getattr(df_sh, "empty", True):
            return {"ok": False, "error": "AkShare macro_china_market_margin_sh 返回空"}
        _write_csv_silent(df_sh, path_sh)

    df_sz = _read_csv_if_fresh(path_sz, ttl_hours=float(ttl_hours))
    if df_sz is None or getattr(df_sz, "empty", True):
        try:
            df_sz = ak.macro_china_market_margin_sz()
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": f"AkShare macro_china_market_margin_sz 调用失败：{exc}"}
        if df_sz is None or getattr(df_sz, "empty", True):
            return {"ok": False, "error": "AkShare macro_china_market_margin_sz 返回空"}
        _write_csv_silent(df_sz, path_sz)

    need_cols = {"日期", "融资融券余额"}
    if not need_cols.issubset(set(getattr(df_sh, "columns", []))):
        return {"ok": False, "error": f"SH 两融数据缺列：{sorted(need_cols)}"}
    if not need_cols.issubset(set(getattr(df_sz, "columns", []))):
        return {"ok": False, "error": f"SZ 两融数据缺列：{sorted(need_cols)}"}

    def _norm(df0):
        dfx = df0.copy()
        dfx["date"] = pd.to_datetime(dfx["日期"], errors="coerce").dt.date
        dfx["margin_balance_yuan"] = pd.to_numeric(dfx["融资融券余额"], errors="coerce")
        dfx = dfx.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        dfx = dfx[dfx["date"] <= as_of].reset_index(drop=True)
        return dfx

    sh = _norm(df_sh)
    sz = _norm(df_sz)
    if sh.empty or sz.empty:
        return {
            "ok": False,
            "error": f"两融数据在 <=as_of={as_of} 范围内为空",
            "cache_paths": {"sh": str(path_sh), "sz": str(path_sz)},
        }

    sh_map = {str(r["date"]): _safe_float(r["margin_balance_yuan"]) for _, r in sh.iterrows() if _safe_float(r["margin_balance_yuan"]) is not None}
    sz_map = {str(r["date"]): _safe_float(r["margin_balance_yuan"]) for _, r in sz.iterrows() if _safe_float(r["margin_balance_yuan"]) is not None}

    # 只用交集日期，避免“某一边缺数据时硬加 0”把序列搞脏。
    dates = sorted(set(sh_map.keys()) & set(sz_map.keys()))
    if not dates:
        return {
            "ok": False,
            "error": "SH/SZ 两融数据日期无交集（数据源异常）",
            "cache_paths": {"sh": str(path_sh), "sz": str(path_sz)},
        }

    vals: list[float] = []
    for d in dates:
        a = sh_map.get(d)
        b = sz_map.get(d)
        if a is None or b is None:
            continue
        vals.append(float(a) + float(b))

    if len(vals) < 40:
        return {
            "ok": False,
            "error": f"两融数据有效样本不足：n={len(vals)}（需要>=40）",
            "cache_paths": {"sh": str(path_sh), "sz": str(path_sz)},
        }

    ref_date = dates[len(vals) - 1]  # vals 与 dates 可能略有偏移，但这里够用
    cur = float(vals[-1])

    win = max(60, min(int(lookback_days), len(vals) - 1))
    hist = [float(x) for x in vals[-1 - win : -1]]
    z = robust_zscore(cur, hist)
    score01 = z_to_score01(z)

    ret_5d = _pct_change(vals, 5)
    ret_20d = _pct_change(vals, 20)

    overheat = bool(score01 is not None and float(score01) >= 0.85 and ((ret_20d is None) or float(ret_20d) >= -0.01))
    deleveraging = bool(((ret_5d is not None) and float(ret_5d) <= -0.02) or ((ret_20d is not None) and float(ret_20d) <= -0.04))

    return {
        "ok": True,
        "as_of": str(as_of),
        "ref_date": str(ref_date),
        "total_margin_balance_yuan": cur,
        "ret_5d": ret_5d,
        "ret_20d": ret_20d,
        "z": z,
        "score01": score01,
        "overheat": bool(overheat),
        "deleveraging": bool(deleveraging),
        "history_days_used": int(len(hist)),
        "cache_paths": {"sh": str(path_sh), "sz": str(path_sz)},
        "source": {"name": "akshare", "func": "macro_china_market_margin_sh/sz"},
        "note": "两融=杠杆温度计；score01 越高代表相对历史更热；仅用于 CashSignal/Regime 风险加权。",
    }

