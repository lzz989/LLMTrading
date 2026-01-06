from __future__ import annotations

from datetime import datetime
from typing import Any


class VsaError(RuntimeError):
    pass


def _to_float(x: Any, *, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:  # noqa: BLE001
        return float(default)


def _classify_rel_volume(x: float) -> str:
    if x < 0.7:
        return "low"
    if x < 1.3:
        return "normal"
    if x < 1.8:
        return "high"
    return "very_high"


def _classify_rel_spread(x: float) -> str:
    if x < 0.7:
        return "narrow"
    if x < 1.3:
        return "normal"
    return "wide"


def add_vsa_features(
    df,
    *,
    vol_window: int = 20,
    spread_window: int = 20,
    prefix: str = "vsa_",
):
    """
    VSA 特征（脚本可算、可复现）：
    - spread/body/wick/clv/close_pos
    - 相对成交量 rel_volume（相对滚动均量）
    - 相对波动 rel_spread（相对滚动均 spread）
    """
    try:
        import numpy as np
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError("没装 pandas/numpy？先跑：pip install -r requirements.txt") from exc

    if "close" not in df.columns or "date" not in df.columns:
        return df

    df2 = df.copy()
    df2["date"] = pd.to_datetime(df2["date"], errors="coerce")
    df2 = df2.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    if df2.empty:
        return df2

    if "open" not in df2.columns:
        df2["open"] = df2["close"]
    if "high" not in df2.columns:
        df2["high"] = df2["close"]
    if "low" not in df2.columns:
        df2["low"] = df2["close"]
    if "volume" not in df2.columns:
        df2["volume"] = 0.0

    open_ = df2["open"].astype(float)
    high = df2["high"].astype(float)
    low = df2["low"].astype(float)
    close = df2["close"].astype(float)
    volume = df2["volume"].fillna(0.0).astype(float)

    spread = (high - low).abs()
    body = (close - open_).abs()
    upper_wick = high - np.maximum(open_, close)
    lower_wick = np.minimum(open_, close) - low

    denom = (high - low).replace(0.0, np.nan)
    clv = ((2 * close - high - low) / denom).fillna(0.0)
    close_pos = ((close - low) / denom).fillna(0.5).clip(lower=0.0, upper=1.0)

    vol_window = int(vol_window) if int(vol_window) > 0 else 20
    spread_window = int(spread_window) if int(spread_window) > 0 else 20

    vol_ma = volume.rolling(vol_window, min_periods=1).mean()
    rel_volume = (volume / vol_ma.replace(0.0, np.nan)).fillna(0.0)

    spread_ma = spread.rolling(spread_window, min_periods=1).mean()
    rel_spread = (spread / spread_ma.replace(0.0, np.nan)).fillna(0.0)

    df2[prefix + "spread"] = spread
    df2[prefix + "body"] = body
    df2[prefix + "upper_wick"] = upper_wick.clip(lower=0.0)
    df2[prefix + "lower_wick"] = lower_wick.clip(lower=0.0)
    df2[prefix + "clv"] = clv
    df2[prefix + "close_pos"] = close_pos
    df2[prefix + "rel_volume"] = rel_volume
    df2[prefix + "rel_spread"] = rel_spread

    return df2


def compute_vsa_report(
    df,
    *,
    vol_window: int = 20,
    spread_window: int = 20,
    lookback_events: int = 120,
    prefix: str = "vsa_",
) -> tuple[Any, dict[str, Any]]:
    """
    产出：
    - df_features：带 vsa_* 列的 df
    - report：结构化 JSON（含最后一根 K 的特征 + 最近若干条事件）
    """
    df2 = add_vsa_features(df, vol_window=vol_window, spread_window=spread_window, prefix=prefix)
    if df2.empty:
        return df2, {"method": "vsa", "params": {"vol_window": vol_window, "spread_window": spread_window}, "summary": {}}

    last = df2.iloc[-1]
    last_date = last.get("date")
    last_close = _to_float(last.get("close"), default=0.0)

    def f(key: str) -> float | None:
        try:
            v = last.get(key)
            return None if v is None else float(v)
        except Exception:  # noqa: BLE001
            return None

    rel_vol = f(prefix + "rel_volume")
    rel_sp = f(prefix + "rel_spread")
    spread = f(prefix + "spread")
    clv = f(prefix + "clv")
    close_pos = f(prefix + "close_pos")

    vol_level = _classify_rel_volume(float(rel_vol or 0.0))
    spread_level = _classify_rel_spread(float(rel_sp or 0.0))

    # 最近 N 根里打标签（非常粗糙，但可复现；别拿它当圣经）
    events: list[dict[str, Any]] = []
    start = max(0, len(df2) - int(lookback_events))
    for i in range(start, len(df2)):
        row = df2.iloc[i]
        dt = row.get("date")
        if not isinstance(dt, datetime):
            continue
        hi = _to_float(row.get("high"))
        lo = _to_float(row.get("low"))
        op = _to_float(row.get("open", row.get("close")))
        cl = _to_float(row.get("close"))
        vol = _to_float(row.get("volume"))
        spr = _to_float(row.get(prefix + "spread"))
        rv = _to_float(row.get(prefix + "rel_volume"))
        rs = _to_float(row.get(prefix + "rel_spread"))
        cp = _to_float(row.get(prefix + "close_pos"), default=0.5)

        v_lvl = _classify_rel_volume(rv)
        s_lvl = _classify_rel_spread(rs)

        labels: list[str] = []
        is_up = cl >= op

        # 这些规则是“憨批也能跑”的极简启发式，目的是给 LLM/人类一个可读的特征锚点
        if v_lvl in {"high", "very_high"} and s_lvl == "wide" and not is_up and cp >= 0.6:
            labels.append("Stopping Volume/Shakeout（下跌放量但收在上部）")
        if v_lvl == "very_high" and s_lvl == "wide" and not is_up and cp <= 0.35:
            labels.append("Selling Climax（恐慌放量收在下部）")
        if v_lvl == "very_high" and s_lvl == "wide" and is_up and cp >= 0.65:
            labels.append("Buying Climax（冲顶放量收在上部）")
        if v_lvl == "low" and s_lvl == "narrow" and is_up and cp < 0.7:
            labels.append("No Demand（上涨但量能不足）")
        if v_lvl == "low" and s_lvl == "narrow" and (not is_up) and cp > 0.3:
            labels.append("No Supply（下跌但抛压不足）")
        if v_lvl in {"high", "very_high"} and s_lvl == "wide" and cp <= 0.25:
            labels.append("Upthrust-like（大波动收在下部，警惕假突破）")

        if labels:
            events.append(
                {
                    "date": dt.strftime("%Y-%m-%d"),
                    "price": float(cl),
                    "open": float(op),
                    "high": float(hi),
                    "low": float(lo),
                    "close": float(cl),
                    "volume": float(vol),
                    "spread": float(spr),
                    "rel_volume": float(rv),
                    "rel_spread": float(rs),
                    "close_pos": float(cp),
                    "labels": labels,
                    "label": " | ".join(labels),
                }
            )

    report: dict[str, Any] = {
        "method": "vsa",
        "params": {"vol_window": int(vol_window), "spread_window": int(spread_window), "lookback_events": int(lookback_events)},
        "summary": {
            "last_date": last_date.strftime("%Y-%m-%d") if isinstance(last_date, datetime) else str(last_date),
            "last_close": float(last_close),
            "vol_level": vol_level,
            "spread_level": spread_level,
            "rel_volume": rel_vol,
            "rel_spread": rel_sp,
            "spread": spread,
            "clv": clv,
            "close_pos": close_pos,
            "events": int(len(events)),
        },
        "last": {
            "date": last_date.strftime("%Y-%m-%d") if isinstance(last_date, datetime) else str(last_date),
            "close": float(last_close),
            "rel_volume": rel_vol,
            "rel_spread": rel_sp,
            "spread": spread,
            "clv": clv,
            "close_pos": close_pos,
            "vol_level": vol_level,
            "spread_level": spread_level,
        },
        "events": events,
    }
    return df2, report

