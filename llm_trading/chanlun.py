from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal


ChanFractalKind = Literal["top", "bottom"]
ChanStrokeDir = Literal["up", "down"]


class ChanlunError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChanFractal:
    index: int
    date: datetime
    kind: ChanFractalKind
    price: float


@dataclass(frozen=True)
class ChanStroke:
    start: ChanFractal
    end: ChanFractal
    direction: ChanStrokeDir

    @property
    def low(self) -> float:
        return float(min(self.start.price, self.end.price))

    @property
    def high(self) -> float:
        return float(max(self.start.price, self.end.price))


@dataclass(frozen=True)
class ChanCenter:
    start_stroke: int
    end_stroke: int
    start_date: datetime
    end_date: datetime
    low: float
    high: float


def _require_columns(df, cols: list[str]):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ChanlunError(f"缺少字段：{miss}（做缠论至少要有 date/high/low/close）")


def _to_float(x: Any, *, default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except (TypeError, ValueError, OverflowError, AttributeError):  # noqa: BLE001
        return float(default)


def _bar_dict(row: Any) -> dict[str, Any]:
    return {
        "date": row["date"],
        "open": _to_float(row.get("open", row["close"])),
        "high": _to_float(row["high"]),
        "low": _to_float(row["low"]),
        "close": _to_float(row["close"]),
        "volume": _to_float(row.get("volume", 0.0)),
    }


def remove_inclusion(df):
    """
    去除包含（极简版）：
    - 先按日期升序
    - 若出现包含/被包含，按最近方向合并 high/low，避免分型过密
    """
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError("没装 pandas？先跑：pip install -r requirements.txt") from exc

    _require_columns(df, ["date", "high", "low", "close"])

    df2 = df.copy()
    df2["date"] = pd.to_datetime(df2["date"], errors="coerce")
    df2 = df2.dropna(subset=["date", "high", "low", "close"]).sort_values("date").reset_index(drop=True)
    if df2.empty or len(df2) < 3:
        raise ChanlunError("数据太少，至少给我 3 根K 线。")

    bars = [_bar_dict(r) for _, r in df2.iterrows()]

    merged: list[dict[str, Any]] = []
    direction = 0  # +1 up, -1 down, 0 unknown

    for bar in bars:
        if not merged:
            merged.append(bar.copy())
            continue

        last = merged[-1]
        include = (bar["high"] <= last["high"] and bar["low"] >= last["low"]) or (
            bar["high"] >= last["high"] and bar["low"] <= last["low"]
        )

        if include:
            if direction >= 0:
                last["high"] = max(last["high"], bar["high"])
                last["low"] = max(last["low"], bar["low"])
            else:
                last["high"] = min(last["high"], bar["high"])
                last["low"] = min(last["low"], bar["low"])
            last["close"] = bar["close"]
            last["volume"] = _to_float(last.get("volume", 0.0)) + _to_float(bar.get("volume", 0.0))
            last["date"] = bar["date"]
            continue

        if bar["high"] > last["high"] and bar["low"] > last["low"]:
            direction = 1
        elif bar["high"] < last["high"] and bar["low"] < last["low"]:
            direction = -1

        merged.append(bar.copy())

    out = pd.DataFrame(merged)
    out = out.dropna(subset=["date", "high", "low", "close"]).reset_index(drop=True)
    return out


def detect_fractals(df_merged) -> list[ChanFractal]:
    _require_columns(df_merged, ["date", "high", "low"])
    fxs: list[ChanFractal] = []

    for i in range(1, len(df_merged) - 1):
        l = df_merged.iloc[i - 1]
        m = df_merged.iloc[i]
        r = df_merged.iloc[i + 1]

        is_top = _to_float(m["high"]) > _to_float(l["high"]) and _to_float(m["high"]) > _to_float(r["high"])
        is_bottom = _to_float(m["low"]) < _to_float(l["low"]) and _to_float(m["low"]) < _to_float(r["low"])

        if is_top and is_bottom:
            continue
        if is_top:
            fxs.append(ChanFractal(index=i, date=m["date"].to_pydatetime(), kind="top", price=_to_float(m["high"])))
        elif is_bottom:
            fxs.append(ChanFractal(index=i, date=m["date"].to_pydatetime(), kind="bottom", price=_to_float(m["low"])))

    return fxs


def filter_fractals(fractals: list[ChanFractal], *, min_gap: int = 4) -> list[ChanFractal]:
    if min_gap < 1:
        min_gap = 1

    out: list[ChanFractal] = []
    for fx in fractals:
        if not out:
            out.append(fx)
            continue

        last = out[-1]
        if fx.kind == last.kind:
            if fx.kind == "top" and fx.price >= last.price:
                out[-1] = fx
            elif fx.kind == "bottom" and fx.price <= last.price:
                out[-1] = fx
            continue

        if fx.index - last.index < min_gap:
            continue

        out.append(fx)

    return out


def build_strokes(fractals: list[ChanFractal]) -> list[ChanStroke]:
    strokes: list[ChanStroke] = []
    for i in range(len(fractals) - 1):
        a = fractals[i]
        b = fractals[i + 1]
        if a.kind == "bottom" and b.kind == "top":
            direction: ChanStrokeDir = "up"
        elif a.kind == "top" and b.kind == "bottom":
            direction = "down"
        else:
            continue
        strokes.append(ChanStroke(start=a, end=b, direction=direction))
    return strokes


def detect_centers(strokes: list[ChanStroke]) -> list[ChanCenter]:
    centers: list[ChanCenter] = []
    i = 0
    while i <= len(strokes) - 3:
        s1, s2, s3 = strokes[i], strokes[i + 1], strokes[i + 2]
        low = max(s1.low, s2.low, s3.low)
        high = min(s1.high, s2.high, s3.high)
        if low <= high:
            start_i = i
            end_i = i + 2
            cur_low, cur_high = low, high
            j = i + 3
            while j < len(strokes):
                sj = strokes[j]
                nlow = max(cur_low, sj.low)
                nhigh = min(cur_high, sj.high)
                if nlow <= nhigh:
                    cur_low, cur_high = nlow, nhigh
                    end_i = j
                    j += 1
                else:
                    break

            centers.append(
                ChanCenter(
                    start_stroke=start_i,
                    end_stroke=end_i,
                    start_date=strokes[start_i].start.date,
                    end_date=strokes[end_i].end.date,
                    low=float(cur_low),
                    high=float(cur_high),
                )
            )
            i = end_i
        else:
            i += 1

    return centers


def compute_chanlun_structure(df, *, min_gap: int = 4) -> dict[str, Any]:
    df_merged = remove_inclusion(df)
    fractals = detect_fractals(df_merged)
    fractals2 = filter_fractals(fractals, min_gap=min_gap)
    strokes = build_strokes(fractals2)
    centers = detect_centers(strokes)

    last_close = _to_float(df["close"].iloc[-1]) if "close" in df.columns and not df.empty else None
    last_date = df["date"].iloc[-1] if "date" in df.columns and not df.empty else None

    last_center = centers[-1] if centers else None
    position_vs_center = "none"
    if last_center and last_close is not None:
        if last_close > last_center.high:
            position_vs_center = "above"
        elif last_close < last_center.low:
            position_vs_center = "below"
        else:
            position_vs_center = "inside"

    summary: dict[str, Any] = {
        "raw_bars": int(len(df)),
        "merged_bars": int(len(df_merged)),
        "fractals": int(len(fractals2)),
        "strokes": int(len(strokes)),
        "centers": int(len(centers)),
        "last_date": last_date.strftime("%Y-%m-%d") if isinstance(last_date, datetime) else str(last_date),
        "last_close": float(last_close) if last_close is not None else None,
        "last_stroke_direction": strokes[-1].direction if strokes else None,
        "position_vs_last_center": position_vs_center,
        "last_center": (
            {
                "start_date": last_center.start_date.strftime("%Y-%m-%d"),
                "end_date": last_center.end_date.strftime("%Y-%m-%d"),
                "low": float(last_center.low),
                "high": float(last_center.high),
            }
            if last_center
            else None
        ),
    }

    def fx_to_dict(fx: ChanFractal) -> dict[str, Any]:
        return {"index": fx.index, "date": fx.date.strftime("%Y-%m-%d"), "kind": fx.kind, "price": float(fx.price)}

    def stroke_to_dict(st: ChanStroke) -> dict[str, Any]:
        return {
            "start": fx_to_dict(st.start),
            "end": fx_to_dict(st.end),
            "direction": st.direction,
            "low": float(st.low),
            "high": float(st.high),
            "length": int(st.end.index - st.start.index),
        }

    def center_to_dict(c: ChanCenter) -> dict[str, Any]:
        return {
            "start_stroke": int(c.start_stroke),
            "end_stroke": int(c.end_stroke),
            "start_date": c.start_date.strftime("%Y-%m-%d"),
            "end_date": c.end_date.strftime("%Y-%m-%d"),
            "low": float(c.low),
            "high": float(c.high),
        }

    return {
        "params": {"min_gap": int(min_gap)},
        "summary": summary,
        "fractals": [fx_to_dict(x) for x in fractals2],
        "strokes": [stroke_to_dict(x) for x in strokes],
        "centers": [center_to_dict(x) for x in centers],
    }

