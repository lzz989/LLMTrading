from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal


SwingKind = Literal["high", "low"]


class DowError(RuntimeError):
    pass


@dataclass(frozen=True)
class SwingPoint:
    index: int
    date: datetime
    kind: SwingKind
    price: float


def _require_columns(df, cols: list[str]):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise DowError(f"缺少字段：{miss}（做 Dow 趋势结构至少要有 date/high/low/close）")


def _to_float(x: Any, *, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:  # noqa: BLE001
        return float(default)


def detect_swings(df, *, lookback: int = 2) -> list[SwingPoint]:
    """
    Dow swing（极简版）：
    - pivot high：当前 high 严格大于左右 lookback 根 high
    - pivot low ：当前 low  严格小于左右 lookback 根 low
    """
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError("没装 pandas？先跑：pip install -r requirements.txt") from exc

    if lookback < 1:
        lookback = 1

    _require_columns(df, ["date", "high", "low", "close"])
    df2 = df.copy()
    df2["date"] = pd.to_datetime(df2["date"], errors="coerce")
    df2 = df2.dropna(subset=["date", "high", "low", "close"]).sort_values("date").reset_index(drop=True)
    if len(df2) < 2 * lookback + 1:
        return []

    highs = df2["high"].astype(float).reset_index(drop=True)
    lows = df2["low"].astype(float).reset_index(drop=True)

    swings: list[SwingPoint] = []
    for i in range(lookback, len(df2) - lookback):
        hi = float(highs.iloc[i])
        lo = float(lows.iloc[i])
        left_hi = float(highs.iloc[i - lookback : i].max())
        right_hi = float(highs.iloc[i + 1 : i + lookback + 1].max())
        left_lo = float(lows.iloc[i - lookback : i].min())
        right_lo = float(lows.iloc[i + 1 : i + lookback + 1].min())

        is_high = hi > left_hi and hi > right_hi
        is_low = lo < left_lo and lo < right_lo
        if is_high and is_low:
            continue
        if is_high:
            swings.append(SwingPoint(index=i, date=df2.iloc[i]["date"].to_pydatetime(), kind="high", price=hi))
        elif is_low:
            swings.append(SwingPoint(index=i, date=df2.iloc[i]["date"].to_pydatetime(), kind="low", price=lo))

    return swings


def filter_swings(swings: list[SwingPoint], *, min_gap: int = 2) -> list[SwingPoint]:
    if min_gap < 1:
        min_gap = 1

    out: list[SwingPoint] = []
    for s in swings:
        if not out:
            out.append(s)
            continue

        last = out[-1]
        if s.index - last.index < min_gap:
            continue

        if s.kind == last.kind:
            if s.kind == "high" and s.price >= last.price:
                out[-1] = s
            elif s.kind == "low" and s.price <= last.price:
                out[-1] = s
            continue

        out.append(s)

    return out


def compute_dow_structure(df, *, lookback: int = 2, min_gap: int = 2) -> dict[str, Any]:
    swings = filter_swings(detect_swings(df, lookback=lookback), min_gap=min_gap)

    last_close = _to_float(df["close"].iloc[-1]) if "close" in df.columns and not df.empty else None
    last_date = df["date"].iloc[-1] if "date" in df.columns and not df.empty else None

    highs = [s for s in swings if s.kind == "high"]
    lows = [s for s in swings if s.kind == "low"]

    high_relation = "unknown"
    low_relation = "unknown"
    if len(highs) >= 2:
        high_relation = "HH" if highs[-1].price > highs[-2].price else ("LH" if highs[-1].price < highs[-2].price else "EQ")
    if len(lows) >= 2:
        low_relation = "HL" if lows[-1].price > lows[-2].price else ("LL" if lows[-1].price < lows[-2].price else "EQ")

    trend = "unknown"
    if high_relation == "HH" and low_relation == "HL":
        trend = "up"
    elif high_relation == "LH" and low_relation == "LL":
        trend = "down"
    elif len(highs) >= 2 and len(lows) >= 2:
        trend = "range"

    bos = "none"
    if last_close is not None:
        if trend == "up" and lows:
            if float(last_close) < float(lows[-1].price):
                bos = "bear_break_below_last_swing_low"
        elif trend == "down" and highs:
            if float(last_close) > float(highs[-1].price):
                bos = "bull_break_above_last_swing_high"
        elif trend == "range":
            if highs and float(last_close) > float(highs[-1].price):
                bos = "range_break_up"
            elif lows and float(last_close) < float(lows[-1].price):
                bos = "range_break_down"

    def sp(s: SwingPoint) -> dict[str, Any]:
        return {"index": int(s.index), "date": s.date.strftime("%Y-%m-%d"), "kind": s.kind, "price": float(s.price)}

    last_high = sp(highs[-1]) if highs else None
    last_low = sp(lows[-1]) if lows else None

    return {
        "params": {"lookback": int(lookback), "min_gap": int(min_gap)},
        "summary": {
            "swings": int(len(swings)),
            "trend": trend,
            "high_relation": high_relation,
            "low_relation": low_relation,
            "bos": bos,
            "last_date": last_date.strftime("%Y-%m-%d") if isinstance(last_date, datetime) else str(last_date),
            "last_close": float(last_close) if last_close is not None else None,
            "last_swing_high": last_high,
            "last_swing_low": last_low,
        },
        "swings": [sp(s) for s in swings],
    }

