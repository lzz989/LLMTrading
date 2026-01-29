from __future__ import annotations

"""
筹码/筹码结构（研究用途）

目标：给“入场过滤 + 上方套牢盘压力（阻力位）”提供可复核的量化 proxy。

注意：
- 这里的“筹码”不是交易所级别的真实成本分布，只能用公开 OHLCV/换手率做近似；
- 在 A 股里它通常更适合作为“风险过滤/价位参考”，别当成独立买卖按钮。
"""

import math
from dataclasses import dataclass
from typing import Any


def _safe_float(x: Any) -> float | None:
    try:
        v = None if x is None else float(x)
    except (TypeError, ValueError, OverflowError):
        return None
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return None
    return float(v)


def _clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return float(x)


def _bin_index(price: float, *, pmin: float, pmax: float, bins: int) -> int:
    if bins <= 1:
        return 0
    if not (pmax > pmin > 0):
        return 0
    # 归一化到 [0,1)
    t = (float(price) - float(pmin)) / (float(pmax) - float(pmin))
    t = _clamp(t, 0.0, 0.999999999)
    i = int(t * int(bins))
    return int(max(0, min(int(bins) - 1, i)))


def _value_area_contiguous(vol_by_bin: list[float], *, poc_idx: int, target_pct: float) -> tuple[int, int]:
    """
    简化版 Value Area：从 POC 向左右扩展，优先吃掉更大的相邻 bin，
    直到累计成交量 >= target_pct。

    这样得到的是“连续区间”，更像交易软件里常见的 VA。
    """
    n = len(vol_by_bin)
    if n <= 0:
        return 0, 0
    total = sum(float(x or 0.0) for x in vol_by_bin)
    if total <= 0:
        return int(max(0, min(n - 1, poc_idx))), int(max(0, min(n - 1, poc_idx)))
    tp = float(target_pct or 0.0)
    tp = _clamp(tp, 0.0, 0.99)
    need = total * tp

    left = int(max(0, min(n - 1, poc_idx)))
    right = int(max(0, min(n - 1, poc_idx)))
    acc = float(vol_by_bin[left] or 0.0)
    if acc >= need:
        return left, right

    while (left > 0) or (right < n - 1):
        cand_left = float(vol_by_bin[left - 1] or 0.0) if left > 0 else -1.0
        cand_right = float(vol_by_bin[right + 1] or 0.0) if right < n - 1 else -1.0

        # 选更“厚”的一边（并列时优先扩右边，贴近“上方压力”习惯）
        if cand_right >= cand_left and right < n - 1:
            right += 1
            acc += max(0.0, cand_right)
        elif left > 0:
            left -= 1
            acc += max(0.0, cand_left)

        if acc >= need:
            break

    return int(left), int(right)


def _local_peaks(vol_by_bin: list[float], *, min_rel_to_max: float) -> list[int]:
    """
    找“局部峰”（近似筹码峰/成交密集区）。
    """
    n = len(vol_by_bin)
    if n < 3:
        return []
    vmax = max(float(x or 0.0) for x in vol_by_bin)
    if vmax <= 0:
        return []
    thr = float(min_rel_to_max or 0.0)
    thr = _clamp(thr, 0.0, 1.0)
    out: list[int] = []
    for i in range(1, n - 1):
        v0 = float(vol_by_bin[i] or 0.0)
        if v0 <= 0:
            continue
        if v0 < vmax * thr:
            continue
        if v0 >= float(vol_by_bin[i - 1] or 0.0) and v0 > float(vol_by_bin[i + 1] or 0.0):
            out.append(i)
    return out


@dataclass(frozen=True, slots=True)
class ChipVbpParams:
    window_days: int = 120
    bins: int = 36
    method: str = "typical"  # typical=(H+L+C)/3；close=收盘价
    value_area_pct: float = 0.70
    peak_min_rel_to_max: float = 0.20
    near_pct_1: float = 0.05
    near_pct_2: float = 0.10


def compute_volume_by_price(
    df,
    *,
    params: ChipVbpParams | None = None,
) -> dict[str, Any] | None:
    """
    Volume-by-Price（成交量按价格分布）近似：
    - 用 daily OHLCV 的 typical price 或 close 把 volume 聚到价格 bin
    - 输出：POC/VAH/VAL + 上方套牢盘比例 + 最近阻力/支撑峰（研究用途）
    """
    try:
        import pandas as pd
        import numpy as np
    except ModuleNotFoundError:  # pragma: no cover
        return None

    if df is None or getattr(df, "empty", True):
        return None

    p = params or ChipVbpParams()
    w = max(20, int(p.window_days or 0))
    bins = max(12, min(200, int(p.bins or 0)))
    method = str(p.method or "typical").strip().lower() or "typical"

    cols_need = {"close", "volume"}
    if method == "typical":
        cols_need |= {"high", "low"}
    if any(c not in df.columns for c in cols_need):
        return None

    df2 = df.copy()
    # 只取尾部窗口
    if len(df2) > w:
        df2 = df2.tail(w)

    close_s = pd.to_numeric(df2["close"], errors="coerce").astype(float)
    vol_s = pd.to_numeric(df2["volume"], errors="coerce").astype(float)
    if method == "typical":
        high_s = pd.to_numeric(df2["high"], errors="coerce").astype(float)
        low_s = pd.to_numeric(df2["low"], errors="coerce").astype(float)
        px_s = (high_s + low_s + close_s) / 3.0
        pmin = float(np.nanmin(low_s.to_numpy(dtype=float)))
        pmax = float(np.nanmax(high_s.to_numpy(dtype=float)))
    else:
        px_s = close_s
        pmin = float(np.nanmin(close_s.to_numpy(dtype=float)))
        pmax = float(np.nanmax(close_s.to_numpy(dtype=float)))

    last_close = _safe_float(close_s.iloc[-1]) if len(close_s) else None
    if last_close is None or last_close <= 0:
        return None

    if (not math.isfinite(pmin)) or (not math.isfinite(pmax)) or (pmin <= 0) or (pmax <= pmin):
        return None

    edges = np.linspace(pmin, pmax, int(bins) + 1, dtype=float)
    centers = (edges[:-1] + edges[1:]) / 2.0
    vol_by_bin = np.zeros(int(bins), dtype=float)

    for px, vv in zip(px_s.to_numpy(dtype=float), vol_s.to_numpy(dtype=float), strict=False):
        if not (math.isfinite(px) and math.isfinite(vv)):
            continue
        if vv <= 0:
            continue
        i = _bin_index(float(px), pmin=float(pmin), pmax=float(pmax), bins=int(bins))
        vol_by_bin[i] += float(vv)

    total_vol = float(vol_by_bin.sum())
    if not (total_vol > 0 and math.isfinite(total_vol)):
        return None
    # 归一化分布（用于“集中度”）
    dist = vol_by_bin / total_vol

    poc_idx = int(int(vol_by_bin.argmax()) if len(vol_by_bin) else 0)
    poc_price = _safe_float(float(centers[poc_idx])) if len(centers) else None

    va_left, va_right = _value_area_contiguous(vol_by_bin.tolist(), poc_idx=poc_idx, target_pct=float(p.value_area_pct))
    val = _safe_float(float(edges[va_left]))
    vah = _safe_float(float(edges[va_right + 1])) if (va_right + 1) < len(edges) else _safe_float(float(edges[-1]))

    # 套牢盘压力：上方成交量占比（近似）
    above_mask = centers > float(last_close)
    below_mask = centers < float(last_close)
    overhead_vol = float(vol_by_bin[above_mask].sum())
    support_vol = float(vol_by_bin[below_mask].sum())
    overhead_pct = overhead_vol / total_vol
    support_pct = support_vol / total_vol

    # “集中度”（VBP 版）：成交量分布是否集中在少数价位 bin（启发式）
    # - top1: POC 单一bin占比（越高=越集中）
    # - top3: 前3个bin占比（越高=筹码峰更尖、更“抱团”）
    try:
        top1 = float(dist.max())
        top3 = float(sum(sorted(dist.tolist(), reverse=True)[:3])) if len(dist) >= 3 else float(top1)
    except Exception:  # noqa: BLE001
        top1 = None
        top3 = None

    # 更关注“近距离”的上方压力（太远的价位不影响短周期）
    near1 = max(0.0, float(p.near_pct_1 or 0.0))
    near2 = max(0.0, float(p.near_pct_2 or 0.0))
    near_up_1 = float(last_close) * (1.0 + near1)
    near_up_2 = float(last_close) * (1.0 + near2)
    near_dn_1 = float(last_close) * (1.0 - near1)
    near_dn_2 = float(last_close) * (1.0 - near2)

    overhead_near_1 = float(vol_by_bin[(centers > float(last_close)) & (centers <= near_up_1)].sum()) / total_vol
    overhead_near_2 = float(vol_by_bin[(centers > float(last_close)) & (centers <= near_up_2)].sum()) / total_vol
    support_near_1 = float(vol_by_bin[(centers < float(last_close)) & (centers >= near_dn_1)].sum()) / total_vol
    support_near_2 = float(vol_by_bin[(centers < float(last_close)) & (centers >= near_dn_2)].sum()) / total_vol

    # 识别“筹码峰”：局部峰（只取显著峰，避免噪声）
    peak_idxs = _local_peaks(vol_by_bin.tolist(), min_rel_to_max=float(p.peak_min_rel_to_max))
    resist = [float(centers[i]) for i in peak_idxs if float(centers[i]) > float(last_close)]
    supp = [float(centers[i]) for i in peak_idxs if float(centers[i]) < float(last_close)]
    resist_sorted = sorted(resist)[:3]
    supp_sorted = sorted(supp, reverse=True)[:3]

    resistance_nearest = _safe_float(resist_sorted[0]) if resist_sorted else None
    support_nearest = _safe_float(supp_sorted[0]) if supp_sorted else None
    resistance_dist_pct = None
    support_dist_pct = None
    if resistance_nearest is not None:
        resistance_dist_pct = float(resistance_nearest) / float(last_close) - 1.0
    if support_nearest is not None:
        support_dist_pct = 1.0 - float(support_nearest) / float(last_close)

    return {
        "ok": True,
        "window_days": int(w),
        "bins": int(bins),
        "method": method,
        "price_min": float(pmin),
        "price_max": float(pmax),
        "poc_price": poc_price,
        "value_area_low": val,
        "value_area_high": vah,
        "overhead_supply_pct": float(overhead_pct),
        "support_pct": float(support_pct),
        # 等价代理（方便对齐“获利盘/套牢盘”的常用说法；口径=VBP）
        "profit_proxy_pct": float(support_pct),
        "loss_proxy_pct": float(overhead_pct),
        "concentration_top1": float(top1) if top1 is not None else None,
        "concentration_top3": float(top3) if top3 is not None else None,
        "overhead_near_pct_5": float(overhead_near_1),
        "overhead_near_pct_10": float(overhead_near_2),
        "support_near_pct_5": float(support_near_1),
        "support_near_pct_10": float(support_near_2),
        "resistance_levels": [float(x) for x in resist_sorted] if resist_sorted else [],
        "support_levels": [float(x) for x in supp_sorted] if supp_sorted else [],
        "resistance_nearest": resistance_nearest,
        "support_nearest": support_nearest,
        "resistance_dist_pct": resistance_dist_pct,
        "support_dist_pct": support_dist_pct,
        "note": "VBP=用(typical/close)把成交量聚到价格分桶的近似；用于套牢盘/支撑阻力启发式，不代表真实持仓成本。",
    }


@dataclass(frozen=True, slots=True)
class ChipCostParams:
    window_days: int = 180
    bins: int = 36
    turnover_col: str = "turnover_rate"  # %（换手率）
    method: str = "typical"
    value_area_pct: float = 0.70


def compute_turnover_cost_distribution(
    df,
    *,
    params: ChipCostParams | None = None,
) -> dict[str, Any] | None:
    """
    换手率驱动的“成本分布”近似（更像软件里的筹码峰）：
    - 每日以 turnover_rate% 作为“筹码迁移比例”
    - 用当日 typical/close 近似成交均价，把迁移筹码落到对应价格 bin

    需要 df 里包含 turnover_rate（TuShare daily_basic 可提供）。
    """
    try:
        import pandas as pd
        import numpy as np
    except ModuleNotFoundError:  # pragma: no cover
        return None

    if df is None or getattr(df, "empty", True):
        return None

    p = params or ChipCostParams()
    w = max(60, int(p.window_days or 0))
    bins = max(12, min(200, int(p.bins or 0)))
    turn_col = str(p.turnover_col or "turnover_rate").strip()
    method = str(p.method or "typical").strip().lower() or "typical"

    cols_need = {"close", turn_col}
    if method == "typical":
        cols_need |= {"high", "low"}
    if any(c not in df.columns for c in cols_need):
        return None

    df2 = df.copy()
    if len(df2) > w:
        df2 = df2.tail(w)

    close_s = pd.to_numeric(df2["close"], errors="coerce").astype(float)
    turn_s = pd.to_numeric(df2[turn_col], errors="coerce").astype(float)
    if method == "typical":
        high_s = pd.to_numeric(df2["high"], errors="coerce").astype(float)
        low_s = pd.to_numeric(df2["low"], errors="coerce").astype(float)
        px_s = (high_s + low_s + close_s) / 3.0
        pmin = float(np.nanmin(low_s.to_numpy(dtype=float)))
        pmax = float(np.nanmax(high_s.to_numpy(dtype=float)))
    else:
        px_s = close_s
        pmin = float(np.nanmin(close_s.to_numpy(dtype=float)))
        pmax = float(np.nanmax(close_s.to_numpy(dtype=float)))

    last_close = _safe_float(close_s.iloc[-1]) if len(close_s) else None
    if last_close is None or last_close <= 0:
        return None
    if (not math.isfinite(pmin)) or (not math.isfinite(pmax)) or (pmin <= 0) or (pmax <= pmin):
        return None

    edges = np.linspace(pmin, pmax, int(bins) + 1, dtype=float)
    centers = (edges[:-1] + edges[1:]) / 2.0
    dist = np.zeros(int(bins), dtype=float)

    inited = False
    for px, tr in zip(px_s.to_numpy(dtype=float), turn_s.to_numpy(dtype=float), strict=False):
        if not (math.isfinite(px) and math.isfinite(tr)):
            continue
        # turnover_rate: % => fraction
        frac = _clamp(float(tr) / 100.0, 0.0, 1.0)
        if not inited:
            i0 = _bin_index(float(px), pmin=float(pmin), pmax=float(pmax), bins=int(bins))
            dist[i0] = 1.0
            inited = True
            continue
        if frac <= 0:
            continue
        dist *= (1.0 - frac)
        i = _bin_index(float(px), pmin=float(pmin), pmax=float(pmax), bins=int(bins))
        dist[i] += frac

    total = float(dist.sum())
    if not (total > 0 and math.isfinite(total)):
        return None
    # 数值漂移矫正
    dist = dist / total

    poc_idx = int(int(dist.argmax()) if len(dist) else 0)
    poc_cost = _safe_float(float(centers[poc_idx])) if len(centers) else None

    va_left, va_right = _value_area_contiguous(dist.tolist(), poc_idx=poc_idx, target_pct=float(p.value_area_pct))
    val = _safe_float(float(edges[va_left]))
    vah = _safe_float(float(edges[va_right + 1])) if (va_right + 1) < len(edges) else _safe_float(float(edges[-1]))

    profit_ratio = float(dist[centers < float(last_close)].sum())
    loss_ratio = float(dist[centers > float(last_close)].sum())

    top1 = float(dist.max())
    top3 = float(sum(sorted(dist.tolist(), reverse=True)[:3])) if len(dist) >= 3 else float(top1)

    return {
        "ok": True,
        "window_days": int(w),
        "bins": int(bins),
        "turnover_col": turn_col,
        "method": method,
        "poc_cost": poc_cost,
        "value_area_low": val,
        "value_area_high": vah,
        "profit_ratio": float(profit_ratio),
        "loss_ratio": float(loss_ratio),
        "concentration_top1": float(top1),
        "concentration_top3": float(top3),
        "note": "成本分布=用换手率驱动的筹码迁移近似；适合看洗筹/套牢盘压力，但不是交易所级别真实成本。",
    }
