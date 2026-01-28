from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True, slots=True)
class StockQualityGate:
    """
    股票质量闸门（硬过滤，研究用途）：
    - 目的：把“杂毛/妖股/不可交易标的”挡在系统外面，避免风控/回测/执行被一字板+流动性抽风狠狠干碎。

    说明：
    - 这里的阈值是“保守默认值”，不是永恒真理；你要改我们再一起调。
    - ETF 不走这套；只对 stock 生效。
    """

    # 基础黑名单（不依赖K线）
    exclude_bj: bool = True  # 北交所：默认一刀切
    exclude_st: bool = True  # ST/*ST：一刀切
    exclude_delisting: bool = True  # 退市/退：一刀切

    # 量化可交易性（依赖日线；默认用近20日均成交额做流动性闸门）
    min_price: float = 2.0  # 低价股更像杂毛温床；默认 2 元
    min_amount_avg20_yuan: float = 50_000_000.0  # 近20日均成交额 >= 5000万


def _name_has_any(name: str, keys: tuple[str, ...]) -> bool:
    n = str(name or "").strip().upper()
    if not n:
        return False
    return any(k in n for k in keys)


def forbid_by_symbol_name(*, symbol: str, name: str | None, gate: StockQualityGate) -> tuple[bool, list[str]]:
    """
    不依赖K线的硬过滤：北交所 / ST / 退市。
    返回：(ok, reasons)
    """
    sym = str(symbol or "").strip().lower()
    nm = str(name or "").strip()
    reasons: list[str] = []

    if gate.exclude_bj and sym.startswith("bj"):
        reasons.append("exclude_bj")

    if nm:
        if gate.exclude_st and _name_has_any(nm, ("ST", "*ST")):
            reasons.append("exclude_st")
        if gate.exclude_delisting and _name_has_any(nm, ("退", "退市")):
            reasons.append("exclude_delisting")

    return (not reasons), reasons


def _to_float(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError, OverflowError):  # noqa: BLE001
        return None
    return float(v) if math.isfinite(v) else None


def stock_liquidity_snapshot(df_daily: Any) -> dict[str, Any]:
    """
    从日线里抽一个“足够用”的流动性快照。

    输出字段（可能为 None）：
    - close_last
    - amount_avg20_yuan：近20日均成交额（优先用 amount，否则用 close*volume 兜底）
    - amount_last_yuan
    - days_20_valid：近20日可用样本数
    """
    try:
        import pandas as pd
    except ModuleNotFoundError:
        return {}

    if df_daily is None or getattr(df_daily, "empty", True):
        return {}

    dfd = df_daily.copy()
    if "date" in dfd.columns:
        dfd["date"] = pd.to_datetime(dfd["date"], errors="coerce")
        dfd = dfd.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    close_s = pd.to_numeric(dfd.get("close"), errors="coerce") if "close" in dfd.columns else None
    if close_s is None:
        return {}

    if "amount" in dfd.columns:
        amt_s = pd.to_numeric(dfd.get("amount"), errors="coerce")
    else:
        vol_s = pd.to_numeric(dfd.get("volume"), errors="coerce") if "volume" in dfd.columns else None
        if vol_s is None:
            return {}
        amt_s = close_s * vol_s

    tail_close = close_s.tail(1)
    close_last = _to_float(tail_close.iloc[0]) if len(tail_close) else None

    tail20 = amt_s.tail(20)
    valid20 = tail20.dropna()
    days_20_valid = int(len(valid20))
    amount_avg20 = _to_float(valid20.mean()) if days_20_valid > 0 else None
    amount_last = _to_float(tail20.iloc[-1]) if len(tail20) else None

    return {
        "close_last": close_last,
        "amount_avg20_yuan": amount_avg20,
        "amount_last_yuan": amount_last,
        "days_20_valid": days_20_valid,
    }


def passes_stock_quality_gate(*, symbol: str, name: str | None, df_daily: Any, gate: StockQualityGate) -> dict[str, Any]:
    """
    综合判断（硬过滤）：黑名单 + 流动性。
    返回 dict：{ok, reasons, snapshot, gate}
    """
    ok0, reasons = forbid_by_symbol_name(symbol=symbol, name=name, gate=gate)

    snap = stock_liquidity_snapshot(df_daily)
    close_last = _to_float(snap.get("close_last"))
    amt20 = _to_float(snap.get("amount_avg20_yuan"))
    valid20 = int(snap.get("days_20_valid") or 0)

    # 流动性闸门：没数据也算不过（否则就成了漏网之鱼）
    if valid20 < 10:
        reasons.append("liq_insufficient_samples_20d")

    if gate.min_price > 0:
        if close_last is None or close_last <= 0:
            reasons.append("bad_price")
        elif close_last + 1e-12 < float(gate.min_price):
            reasons.append("min_price")

    if gate.min_amount_avg20_yuan > 0:
        if amt20 is None or amt20 <= 0:
            reasons.append("bad_amount_avg20")
        elif amt20 + 1e-6 < float(gate.min_amount_avg20_yuan):
            reasons.append("min_amount_avg20")

    ok = bool(ok0) and (not reasons)
    return {
        "ok": bool(ok),
        "reasons": reasons,
        "snapshot": snap,
        "gate": {
            "exclude_bj": bool(gate.exclude_bj),
            "exclude_st": bool(gate.exclude_st),
            "exclude_delisting": bool(gate.exclude_delisting),
            "min_price": float(gate.min_price),
            "min_amount_avg20_yuan": float(gate.min_amount_avg20_yuan),
        },
    }

